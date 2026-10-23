"""Rankers trained on the weighted slate log-loss of Eq. 3 (paper §3, §6.1, App. B.4).

Every ranker scores (user, item) pairs; P_θ(y | u, S) is Eq. 1 with the ranker's scores as utilities
and a learned leave score. Rankers: MF (the paper's BPR-MF backbone), LightGCN, SASRec and TextMF,
a text-aware MF that stands in for UniSRec.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, A = 0, 1   # channel ids


@dataclass
class Batch:
    users: torch.Tensor       # (B,)
    channels: torch.Tensor    # (B,)
    slates: torch.Tensor      # (B, K) global item ids
    versions: torch.Tensor    # (B, K) displayed version ids
    choice: torch.Tensor      # (B,) index in [0, K]; K = leave
    weight: torch.Tensor      # (B,) w(e)
    hist: torch.Tensor        # (B, L) previous purchases, -1 padded (SASRec / TextMF)


class Ranker(nn.Module):
    def __init__(self, n_users, n_items, dim=32, channel_heads=False):
        super().__init__()
        self.n_users, self.n_items, self.dim = n_users, n_items, dim
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.item_bias.weight)
        self.leave = nn.Parameter(torch.zeros(1))
        self.channel_heads = channel_heads
        if channel_heads:   # per-channel item offsets; humans are served the human head
            self.head = nn.Embedding(2 * n_items, 1)
            nn.init.zeros_(self.head.weight)

    def prepare(self):
        """Hook run once per training step / scoring call (graph propagation for LightGCN)."""
        return None

    def user_repr(self, users, hist, ctx):
        raise NotImplementedError

    def item_repr(self, items, versions, ctx):
        raise NotImplementedError

    def score(self, users, items, channels=None, versions=None, hist=None, ctx=None):
        """users (B,), items (B, K) -> (B, K) scores."""
        u = self.user_repr(users, hist, ctx)
        v = self.item_repr(items, versions, ctx)
        s = (u.unsqueeze(1) * v).sum(-1) + self.item_bias(items).squeeze(-1)
        if self.channel_heads and channels is not None:
            s = s + self.head(items + channels.unsqueeze(1) * self.n_items).squeeze(-1)
        return s

    def slate_log_prob(self, b: Batch, ctx=None):
        s = self.score(b.users, b.slates, b.channels, b.versions, b.hist, ctx)
        z = torch.cat([s, self.leave.expand(s.shape[0], 1)], dim=1)
        return F.log_softmax(z, dim=1).gather(1, b.choice.unsqueeze(1)).squeeze(1)


def slate_loss(model: Ranker, b: Batch, ctx=None):
    """Eq. 3: L_w(θ) = Σ_e w(e) ℓ_θ(e), ℓ_θ(e) = −log P_θ(y_e | u_e, S_e) (normalized by batch size)."""
    return -(b.weight * model.slate_log_prob(b, ctx)).sum() / max(1, b.users.shape[0])


class MF(Ranker):
    def __init__(self, n_users, n_items, dim=32, channel_heads=False, **_):
        super().__init__(n_users, n_items, dim, channel_heads)
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)

    def user_repr(self, users, hist, ctx):
        return self.U(users)

    def item_repr(self, items, versions, ctx):
        return self.V(items)


class LightGCN(Ranker):
    """LightGCN propagation over the (weighted) user-item purchase graph, layer-mean readout."""

    def __init__(self, n_users, n_items, dim=32, channel_heads=False, layers=2, **_):
        super().__init__(n_users, n_items, dim, channel_heads)
        self.E = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.E.weight, std=0.1)
        self.layers = layers
        self.adj = None

    def set_graph(self, users, items, weights):
        n = self.n_users + self.n_items
        u = torch.as_tensor(users, dtype=torch.long)
        i = torch.as_tensor(items, dtype=torch.long) + self.n_users
        w = torch.as_tensor(weights, dtype=torch.float32)
        keep = w > 0
        u, i, w = u[keep], i[keep], w[keep]
        rows = torch.cat([u, i])
        cols = torch.cat([i, u])
        vals = torch.cat([w, w])
        A_ = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()
        deg = torch.sparse.sum(A_, dim=1).to_dense().clamp_min(1e-8)
        dinv = deg.pow(-0.5)
        idx = A_.indices()
        norm = dinv[idx[0]] * A_.values() * dinv[idx[1]]
        self.adj = torch.sparse_coo_tensor(idx, norm, (n, n)).coalesce().to(self.E.weight.device)

    def prepare(self):
        x = self.E.weight
        if self.adj is None:
            return x
        out, h = x, x
        for _ in range(self.layers):
            h = torch.sparse.mm(self.adj, h)
            out = out + h
        return out / (self.layers + 1)

    def user_repr(self, users, hist, ctx):
        return ctx[users]

    def item_repr(self, items, versions, ctx):
        return ctx[items + self.n_users]


class SASRec(Ranker):
    """Self-attentive sequential recommender over the user's previous purchases (unweighted histories)."""

    def __init__(self, n_users, n_items, dim=32, channel_heads=False, max_len=20, heads=2, **_):
        super().__init__(n_users, n_items, dim, channel_heads)
        self.V = nn.Embedding(n_items + 1, dim, padding_idx=n_items)   # last row = padding
        self.pos = nn.Embedding(max_len + 1, dim)
        self.cls = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.V.weight, std=0.1)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.max_len = max_len

    def user_repr(self, users, hist, ctx):
        h = hist.clone()
        pad = h < 0
        h[pad] = self.n_items
        B, L = h.shape
        x = self.V(h)
        x = torch.cat([x, self.cls.expand(B, 1, -1)], dim=1)       # read out at a final query token
        x = x + self.pos(torch.arange(L + 1, device=h.device)).unsqueeze(0)
        mask = torch.cat([pad, torch.zeros(B, 1, dtype=torch.bool, device=h.device)], dim=1)
        y = self.enc(x, src_key_padding_mask=mask)
        return y[:, -1]

    def item_repr(self, items, versions, ctx):
        return self.V(items)


class TextMF(Ranker):
    """Text-aware MF (UniSRec stand-in): item vector = id embedding + projection of the shown version's
    frozen text features; the user vector adds a mean of history text features (UniSRec-style)."""

    def __init__(self, n_users, n_items, dim=32, channel_heads=False, text_feats=None, max_versions=16, **_):
        super().__init__(n_users, n_items, dim, channel_heads)
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)
        tf = torch.as_tensor(text_feats, dtype=torch.float32)       # (n_items, max_versions, t)
        self.register_buffer("text", tf)
        self.proj = nn.Linear(tf.shape[-1], dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.05)

    def _text(self, items, versions):
        return self.proj(self.text[items, versions.clamp_max(self.text.shape[1] - 1)])

    def user_repr(self, users, hist, ctx):
        u = self.U(users)
        if hist is not None:
            valid = hist >= 0
            h = hist.clamp_min(0)
            t = self.proj(self.text[h, 0]) * valid.unsqueeze(-1)
            u = u + t.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return u

    def item_repr(self, items, versions, ctx):
        v = self.V(items)
        if versions is None:
            versions = torch.zeros_like(items)
        return v + self._text(items, versions)


RANKERS = {"mf": MF, "lightgcn": LightGCN, "sasrec": SASRec, "textmf": TextMF}


def make_ranker(name, n_users, n_items, dim=32, channel_heads=False, **kw):
    return RANKERS[name](n_users, n_items, dim=dim, channel_heads=channel_heads, **kw)


def to_batch(ev, idx, weight, device):
    t = lambda a, dt=torch.long: torch.as_tensor(a[idx], dtype=dt, device=device)
    return Batch(t(ev["user"]), t(ev["channel"]), t(ev["slate"]), t(ev["version"]), t(ev["choice"]),
                 torch.as_tensor(weight[idx], dtype=torch.float32, device=device), t(ev["hist"]))


def train_ranker(model: Ranker, ev, weight, epochs=3, lr=0.01, batch_size=1024, weight_decay=1e-6,
                 seed=0, device="cpu", graph=None):
    """Warm-started fixed-epoch-budget training on L_w (Alg. 2 line 8). ev: dict of numpy arrays."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if graph is not None and isinstance(model, LightGCN):
        model.set_graph(*graph)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    keep = np.nonzero(weight > 0)[0]
    n = len(keep)
    losses = []
    for _ in range(epochs):
        perm = keep[torch.randperm(n, generator=g).numpy()]
        for st in range(0, n, batch_size):
            b = to_batch(ev, perm[st:st + batch_size], weight, device)
            ctx = model.prepare()
            loss = slate_loss(model, b, ctx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    model.eval()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def event_nll(model: Ranker, ev, idx, device="cpu", batch_size=4096):
    """Per-event −log P_θ(y | u, S) (validation loss, LoRec-style user scores)."""
    model.eval()
    ctx = model.prepare()
    out = []
    w1 = np.ones(len(ev["user"]), np.float32)
    for st in range(0, len(idx), batch_size):
        b = to_batch(ev, idx[st:st + batch_size], w1, device)
        out.append(-model.slate_log_prob(b, ctx).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def score_items(model: Ranker, users, items, channels, versions, hist, device="cpu", batch_size=4096):
    """Scores (R, n_cand) for each request's candidate items."""
    model.eval()
    ctx = model.prepare()
    out = []
    for st in range(0, len(users), batch_size):
        sl = slice(st, st + batch_size)
        s = model.score(torch.as_tensor(users[sl], device=device),
                        torch.as_tensor(items[sl], device=device),
                        torch.as_tensor(channels[sl], device=device),
                        torch.as_tensor(versions[sl], device=device),
                        torch.as_tensor(hist[sl], device=device), ctx)
        out.append(s.cpu().numpy())
    return np.concatenate(out)
