"""Public-data marketplaces for the retraining worlds (paper §6.1, App. B.3).

AMZ-E / AMZ-H: Amazon Reviews 2023 [15] (5-core ratings + item metadata) for Electronics and Home and
Kitchen; 5 subcategories (leaf categories) × 60 items, 2,000 users. ML-1M: MovieLens-1M [12], 5 primary
genres × 60 movies, 1,500 users, no listing text (synthetic stress test).

The human choice model of App. B.3 is a within-subcategory conditional logit
    P(i | u, c) ∝ exp(b_i + p_u·q_i)
fitted to the review (rating) events of all items of the selected subcategories; reviews are read as
purchases. The worlds keep b_i (centred per subcategory), q_i of the 60 selected
items and p_u of the selected users.

Output of `prepare_*`: data/processed/<name>/{market.npz, items.json, fit.json}.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
AMAZON = {"amz_e": "Electronics", "amz_h": "Home_and_Kitchen"}


# ---------------------------------------------------------------- loading prepared data

@dataclass
class MarketData:
    name: str
    item_cat: np.ndarray          # (n_items,) subcategory index; items sorted by category
    cat_names: list
    items: list                   # [{id, title, text, facts}] in item order
    quality: np.ndarray           # (n_items,) b_i, centred per category
    Q: np.ndarray                 # (n_items, d)
    P: np.ndarray                 # (n_users, d)
    text_emb: np.ndarray | None   # (n_items, t) frozen sentence embedding of title (+ text)
    llm_emb: np.ndarray | None    # (n_items, e) local-LLM item embedding (LoRec calibrator)
    agent_quality: np.ndarray | None   # (n_items,) agent base utility fitted to LLM harness responses
    fit: dict

    @property
    def n_users(self):
        return self.P.shape[0]

    @property
    def n_cats(self):
        return len(self.cat_names)

    @property
    def items_per_cat(self):
        return int(np.bincount(self.item_cat).max())


@lru_cache(maxsize=8)
def load_market(name: str, root: str | None = None) -> MarketData:
    """name: a prepared market under data/processed, or a directory holding one."""
    d = Path(root) if root else (Path(name) if Path(name).is_dir() else PROCESSED / name)
    z = np.load(d / "market.npz")
    meta = json.loads((d / "items.json").read_text())
    opt = lambda f: np.load(d / f) if (d / f).exists() else None
    return MarketData(name=name, item_cat=z["item_cat"], cat_names=meta["cat_names"], items=meta["items"],
                      quality=z["quality"], Q=z["Q"], P=z["P"], text_emb=opt("text_emb.npy"),
                      llm_emb=opt("llm_emb.npy"), agent_quality=opt("agent_quality.npy"),
                      fit=json.loads((d / "fit.json").read_text()))


# ---------------------------------------------------------------- choice-model fit

def fit_choice_model(users, items, item_group, n_users, n_items, dim=16, l2_grid=(1e-3, 1e-2, 1e-1, 1.0), epochs=30,
                     lr=0.02, batch=8192, val_frac=0.1, seed=0, device=None, n_focus=None):
    """Within-group conditional logit b_i + p_u·q_i by full softmax over each group's items.

    users/items: review events; item_group: (n_items,) group of each item. The factor penalty
    l2·(‖p_u‖² + ‖q_i‖²) is applied per event; l2 and the epoch are chosen by held-out NLL (early
    stopping) on the held-out events of the n_focus first users (the users kept in the worlds; all
    users if None). Returns (b, P, Q, diag); diag compares held-out NLL with the popularity-only (b_i)
    model and uniform choice, overall and for the focus users.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(users)
    val = rng.random(n) < val_frac
    focus = users < (n_focus if n_focus is not None else n_users)
    groups = [np.where(item_group == g)[0] for g in range(item_group.max() + 1)]
    pos_in_group = np.zeros(n_items, np.int64)
    for gi in groups:
        pos_in_group[gi] = np.arange(len(gi))
    groups_t = [torch.as_tensor(gi, device=device) for gi in groups]
    ev_group = item_group[items]
    tgt = torch.as_tensor(pos_in_group[items], device=device)
    u_t = torch.as_tensor(users, device=device)

    def run(use_pref, l2=0.0):
        torch.manual_seed(seed)
        b = torch.zeros(n_items, device=device, requires_grad=True)
        P = (0.01 * torch.randn(n_users, dim, device=device)).requires_grad_(use_pref)
        Q = (0.01 * torch.randn(n_items, dim, device=device)).requires_grad_(use_pref)
        params = [b] + ([P, Q] if use_pref else [])
        opt = torch.optim.Adam(params, lr=lr)

        def nll(g, idx, reg=False):   # events idx all in group g
            idx = torch.as_tensor(idx, device=device)
            gi = groups_t[g]
            logit = b[gi].expand(len(idx), -1)
            pen = 0.0
            if use_pref:
                pu = P[u_t[idx]]
                logit = logit + pu @ Q[gi].T
                if reg:
                    pen = l2 * (pu.pow(2).sum(1) + Q[gi[tgt[idx]]].pow(2).sum(1))
            return torch.nn.functional.cross_entropy(logit, tgt[idx], reduction="none") + pen

        def val_nll(sel=None):
            sel = val if sel is None else sel
            with torch.no_grad():
                return float(torch.cat([nll(g, np.where(sel & (ev_group == g))[0]) for g in range(len(groups))]).mean())

        tr = [np.where(~val & (ev_group == g))[0] for g in range(len(groups))]
        best = (float("inf"), None, -1)
        for ep in range(epochs):
            chunks = []
            for g, ev in enumerate(tr):
                ev = ev[rng.permutation(len(ev))]
                chunks += [(g, ev[st:st + batch]) for st in range(0, len(ev), batch)]
            for k in rng.permutation(len(chunks)):
                g, idx = chunks[k]
                loss = nll(g, idx, reg=True).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = val_nll(val & focus)
            if v < best[0]:
                best = (v, [x.detach().cpu().numpy().copy() for x in (b, P, Q)], ep, val_nll())
            elif ep - best[2] >= 3:
                break
        bb, PP, QQ = best[1]
        return bb, PP, QQ, best[3], best[2] + 1, best[0]

    b0, _, _, v_pop, _, v_pop_focus = run(False)
    fits = {l2: run(True, l2) for l2 in l2_grid}
    l2 = min(fits, key=lambda k: fits[k][5])
    b, P, Q, v_full, ep_best, v_full_focus = fits[l2]
    sizes = np.array([len(g) for g in groups])
    v_unif = float(np.mean(np.log(sizes[item_group[items[val]]])))
    diag = dict(n_events=int(n), n_val=int(val.sum()), val_nll_full=v_full, val_nll_popularity=v_pop,
                val_nll_uniform=v_unif, dim=dim, l2=l2, epochs=ep_best,
                val_nll_focus_full=v_full_focus, val_nll_focus_popularity=v_pop_focus,
                n_val_focus=int((val & focus).sum()),
                val_nll_focus_by_l2={str(k): v[5] for k, v in fits.items()})
    return b, P, Q, diag


def _save(name, item_cat, cat_names, items, b, P, Q, diag, extra=None):
    d = PROCESSED / name
    d.mkdir(parents=True, exist_ok=True)
    q = b.astype(np.float64).copy()
    for c in range(len(cat_names)):
        m = item_cat == c
        q[m] -= q[m].mean()
    np.savez(d / "market.npz", item_cat=item_cat, quality=q, Q=Q, P=P)
    (d / "items.json").write_text(json.dumps(dict(cat_names=cat_names, items=items), indent=1))
    diag = dict(diag, **(extra or {}), n_users=int(P.shape[0]), n_items=int(len(items)),
                quality_sd=float(q.std()), pref_sd=float(np.std(P @ Q.T)))
    (d / "fit.json").write_text(json.dumps(diag, indent=1))
    return d


# ---------------------------------------------------------------- Amazon Reviews 2023

def _num(x):
    try:
        v = float(str(x).replace("$", "").replace(",", "").split()[0])
        return v if math.isfinite(v) else None
    except (ValueError, IndexError):
        return None


def prepare_amazon(name, n_cats=5, items_per_cat=60, n_users=2000, min_item_reviews=5, dim=16, seed=0,
                   max_text_chars=400):
    import pandas as pd
    cat = AMAZON[name]
    df = pd.read_csv(RAW / f"{cat}.csv.gz", usecols=["user_id", "parent_asin", "rating", "timestamp"])
    counts = df["parent_asin"].value_counts()
    keep = set(counts.index[counts >= min_item_reviews])
    meta = {}
    with gzip.open(RAW / f"meta_{cat}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            a = r.get("parent_asin")
            if a not in keep:
                continue
            cats = r.get("categories") or []
            if len(cats) < 2 or not r.get("title"):
                continue
            feats = " ".join(r.get("features") or [])
            desc = " ".join(r.get("description") or [])
            meta[a] = dict(leaf=" > ".join(cats[1:]), top=cats[1], title=r["title"][:200],
                           text=(feats or desc)[:max_text_chars], price=_num(r.get("price")),
                           rating=r.get("average_rating"), reviews=r.get("rating_number"), store=r.get("store"))
    df = df[df["parent_asin"].isin(meta.keys())]
    leaf = df["parent_asin"].map(lambda a: meta[a]["leaf"])
    # Leaf categories ranked by review volume among items with a known price; distinct parents.
    priced = df[df["parent_asin"].map(lambda a: meta[a]["price"] is not None)]
    vol = priced.groupby(leaf[priced.index]).size().sort_values(ascending=False)
    chosen, parents = [], set()
    for lf in vol.index:
        items_lf = [a for a, m in meta.items() if m["leaf"] == lf and m["price"] is not None]
        if len(items_lf) < items_per_cat or lf.split(" > ")[0] in parents:
            continue
        chosen.append(lf)
        parents.add(lf.split(" > ")[0])
        if len(chosen) == n_cats:
            break
    fit_df = df[leaf.isin(chosen)].copy()
    fit_df["leaf"] = leaf[fit_df.index]
    fit_items = sorted(fit_df["parent_asin"].unique(), key=lambda a: (chosen.index(meta[a]["leaf"]), a))
    item_idx = {a: i for i, a in enumerate(fit_items)}
    group = np.array([chosen.index(meta[a]["leaf"]) for a in fit_items])
    # Selected items: the 60 most-reviewed priced items of each chosen leaf.
    rc = fit_df["parent_asin"].value_counts()
    selected = []
    for lf in chosen:
        cands = [a for a in rc.index if meta[a]["leaf"] == lf and meta[a]["price"] is not None]
        selected += cands[:items_per_cat]
    # Users: the n_users most active in the chosen leaves (fit uses every reviewer).
    uc = fit_df["user_id"].value_counts()
    top_users = list(uc.index[:n_users])
    all_users = {u: i for i, u in enumerate(top_users + [u for u in uc.index[n_users:]])}
    b, P, Q, diag = fit_choice_model(fit_df["user_id"].map(all_users).to_numpy(),
                                     fit_df["parent_asin"].map(item_idx).to_numpy(), group,
                                     len(all_users), len(fit_items), dim=dim, seed=seed, n_focus=n_users)
    sel_idx = np.array([item_idx[a] for a in selected])
    items = [dict(id=a, title=meta[a]["title"], text=meta[a]["text"],
                  facts=dict(price=meta[a]["price"], rating=meta[a]["rating"], reviews=meta[a]["reviews"],
                             brand=meta[a]["store"]), category=meta[a]["leaf"]) for a in selected]
    sel_reviews = fit_df[fit_df["parent_asin"].isin(set(selected)) & fit_df["user_id"].isin(set(top_users))]
    extra = dict(source=f"Amazon Reviews 2023 {cat} 5-core", categories=chosen, fit_items=len(fit_items),
                 fit_users=len(all_users), selected_user_reviews=int(len(sel_reviews)),
                 selected_item_reviews=int(fit_df["parent_asin"].isin(set(selected)).sum()))
    return _save(name, group[sel_idx], chosen, items, b[sel_idx], P[:n_users], Q[sel_idx], diag, extra)


# ---------------------------------------------------------------- MovieLens-1M

ML_GENRES = ("Comedy", "Drama", "Action", "Thriller", "Horror")


def prepare_movielens(name="ml_1m", genres=ML_GENRES, items_per_cat=60, n_users=1500, dim=16, seed=0):
    """Subcategories = primary (first-listed) genre; 60 most-rated movies each; the 1,500 users with
    most ratings of them. Ratings of every movie of these genres fit the choice model."""
    import pandas as pd
    d = RAW / "ml-1m"
    mv = pd.read_csv(d / "movies.dat", sep="::", engine="python", names=["mid", "title", "genres"],
                     encoding="latin-1")
    rt = pd.read_csv(d / "ratings.dat", sep="::", engine="python", names=["uid", "mid", "rating", "ts"])
    mv["primary"] = mv["genres"].str.split("|").str[0]
    mv = mv[mv["primary"].isin(genres)]
    rt = rt[rt["mid"].isin(set(mv["mid"]))]
    g_of = dict(zip(mv["mid"], mv["primary"]))
    title = dict(zip(mv["mid"], mv["title"]))
    rc = rt["mid"].value_counts()
    selected = []
    for g in genres:
        selected += [m for m in rc.index if g_of[m] == g][:items_per_cat]
    sel = set(selected)
    uc = rt[rt["mid"].isin(sel)]["uid"].value_counts()
    top_users = list(uc.index[:n_users])
    others = [u for u in rt["uid"].value_counts().index if u not in set(top_users)]
    all_users = {u: i for i, u in enumerate(top_users + others)}
    fit_items = sorted(rt["mid"].unique(), key=lambda m: (genres.index(g_of[m]), m))
    item_idx = {m: i for i, m in enumerate(fit_items)}
    group = np.array([genres.index(g_of[m]) for m in fit_items])
    b, P, Q, diag = fit_choice_model(rt["uid"].map(all_users).to_numpy(), rt["mid"].map(item_idx).to_numpy(),
                                     group, len(all_users), len(fit_items), dim=dim, seed=seed, n_focus=n_users)
    sel_idx = np.array([item_idx[m] for m in selected])
    items = [dict(id=str(m), title=title[m], text="", facts=dict(), category=g_of[m]) for m in selected]
    n_sel = int(rt[rt["mid"].isin(sel) & rt["uid"].isin(set(top_users))].shape[0])
    extra = dict(source="MovieLens-1M", categories=list(genres), fit_items=len(fit_items),
                 fit_users=len(all_users), selected_user_reviews=n_sel)
    return _save(name, group[sel_idx], list(genres), items, b[sel_idx], P[:n_users], Q[sel_idx], diag, extra)
