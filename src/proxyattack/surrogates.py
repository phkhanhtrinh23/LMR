"""Text-aware human-response surrogates for GPA.

Each surrogate h ∈ H is a conditional-logit choice model (Eq. 1) whose item utility is linear in fact
features and hashed bag-of-words text features, fitted to public choice records with alternative
texts and frozen before search. The family holds alternative fits (L2 strength, hash seed). GPA
uses the family only if it passes a predeclared revision-effect calibration on held-out records.

Choice record: {"items": [{"facts": {...}, "text": str}, ...], "choice": index or -1 (left)}.
Calibration pair: {"pair_id": str, "variant": 0|1, "target": index, ...record}.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

_TOK = re.compile(r"[a-z0-9]+")


def tokens(text):
    return _TOK.findall((text or "").lower())


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


@dataclass
class TextLogit:
    hash_dim: int = 256
    hash_seed: int = 0
    l2: float = 1.0
    fact_keys: tuple = ("price", "rating", "reviews")
    theta: np.ndarray | None = None
    leave: float = 0.0
    vocab: set = field(default_factory=set)
    len_range: tuple = (0, 10 ** 9)

    def features(self, item):
        f = np.zeros(len(self.fact_keys) + 1 + self.hash_dim)
        for i, k in enumerate(self.fact_keys):
            v = _num(item.get("facts", {}).get(k))
            if v is not None:
                f[i] = math.log1p(abs(v)) * (1 if v >= 0 else -1) if k in ("price", "reviews") else v
        toks = tokens(item.get("text", ""))
        f[len(self.fact_keys)] = math.log1p(len(toks))
        off = len(self.fact_keys) + 1
        for t in toks:
            h = int(hashlib.md5(f"{self.hash_seed}:{t}".encode()).hexdigest()[:8], 16)
            f[off + h % self.hash_dim] += 1.0
        if toks:
            f[off:] /= math.sqrt(len(toks))
        return f

    def _design(self, records):
        X = [np.stack([self.features(it) for it in r["items"]]) for r in records]
        y = [r["choice"] for r in records]
        return X, y

    def fit(self, records):
        X, y = self._design(records)
        d = X[0].shape[1]

        def nll(w):
            th, lv = w[:-1], w[-1]
            loss, grad = 0.0, np.zeros_like(w)
            for Xi, yi in zip(X, y):
                u = np.append(Xi @ th, lv)
                u -= u.max()
                p = np.exp(u); p /= p.sum()
                k = len(Xi) if yi < 0 else yi
                loss -= math.log(p[k] + 1e-300)
                g = p.copy(); g[k] -= 1
                grad[:-1] += Xi.T @ g[:-1]
                grad[-1] += g[-1]
            loss += 0.5 * self.l2 * th @ th
            grad[:-1] += self.l2 * th
            return loss, grad

        res = minimize(nll, np.zeros(d + 1), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.theta, self.leave = res.x[:-1], float(res.x[-1])
        self.vocab = {t for r in records for it in r["items"] for t in tokens(it.get("text", ""))}
        lens = [len(tokens(it.get("text", ""))) for r in records for it in r["items"]]
        self.len_range = (min(lens), max(lens))
        return self

    def probs(self, items):
        u = np.append(np.stack([self.features(it) for it in items]) @ self.theta, self.leave)
        u -= u.max()
        p = np.exp(u)
        return p / p.sum()

    def target_prob(self, items, target):
        return float(self.probs(items)[target])

    def in_support(self, text, min_known=0.8, slack=1.5):
        """Support rule: most tokens seen in training and length within the training range (with slack)."""
        toks = tokens(text)
        if not toks:
            return False
        known = np.mean([t in self.vocab for t in toks])
        lo, hi = self.len_range
        return bool(known >= min_known and lo / slack <= len(toks) <= hi * slack)


def fit_family(records, settings=((1.0, 0), (0.3, 1), (3.0, 2))):
    """|H| alternative fits (paper: 3 text-aware logit fits)."""
    return [TextLogit(l2=l2, hash_seed=hs).fit(records) for l2, hs in settings]


def revision_effect_calibration(model: TextLogit, pairs):
    """Aggregate revision-effect calibration error (log odds) on held-out pairs: |mean predicted − mean
    observed| of logit(target share | variant 1) − logit(target share | variant 0) across pairs."""
    groups = {}
    for r in pairs:
        groups.setdefault(r["pair_id"], {0: [], 1: []})[r["variant"]].append(r)
    pred, obs = [], []
    for g in groups.values():
        if not g[0] or not g[1]:
            continue
        sh = [np.mean([r["choice"] == r["target"] for r in g[v]]) for v in (0, 1)]
        sh = [(s * len(g[v]) + 0.5) / (len(g[v]) + 1) for s, v in zip(sh, (0, 1))]
        obs.append(math.log(sh[1] / (1 - sh[1])) - math.log(sh[0] / (1 - sh[0])))
        pp = [np.mean([model.target_prob(r["items"], r["target"]) for r in g[v]]) for v in (0, 1)]
        pred.append(math.log(pp[1] / (1 - pp[1])) - math.log(pp[0] / (1 - pp[0])))
    if not obs:
        return float("inf")
    return float(abs(np.mean(pred) - np.mean(obs)))


def family_passes(family, pairs, tolerance=0.15):
    errs = [revision_effect_calibration(h, pairs) for h in family]
    return all(e <= tolerance for e in errs), errs


PHRASE_BANK = ["Built for everyday use and easy to carry.", "A practical choice for busy days.",
               "Designed with a clean, simple look.", "Comfortable to hold and simple to fill.",
               "Great for travel, work and the outdoors.", "Easy to clean after use."]


def synthetic_records(listings, n_records, rng, text_effect=None, slate=8, alt_texts=None, pair_mode=False,
                      alt_prob=0.5):
    """Toy public preference records from a known human utility. Each listing shows its original text or, with probability alt_prob,
    an alternative text (sentences reordered plus generic phrases). text_effect(text) -> utility."""
    text_effect = text_effect or (lambda t: 0.0)

    def alt_text(text):
        if rng.random() >= alt_prob:
            return text
        sents = [x for x in re.split(r"(?<=\.)\s+", text) if x]
        rng.shuffle(sents)
        k = int(rng.integers(0, 3))
        return " ".join(sents + list(rng.choice(PHRASE_BANK, k, replace=False)))

    out = []
    for i in range(n_records):
        idx = rng.choice(len(listings), min(slate, len(listings)), replace=False)
        items = [dict(facts=listings[j]["facts"], text=alt_text(listings[j]["text"])) for j in idx]
        target, variant = 0, 0
        if pair_mode:
            variant = i % 2
            if variant and alt_texts:
                items[0] = dict(items[0], text=alt_texts[listings[idx[0]]["id"]])
        u = np.array([float(it["facts"].get("rating", 4)) - 0.3 * math.log1p(float(it["facts"].get("price", 20)))
                      + text_effect(it["text"]) for it in items] + [0.5])
        p = np.exp(u - u.max()); p /= p.sum()
        c = int(rng.choice(len(p), p=p))
        rec = dict(items=items, choice=-1 if c == len(items) else c)
        if pair_mode:
            rec.update(pair_id=f"p{i // 2 % 20}", variant=variant, target=target)
        out.append(rec)
    return out
