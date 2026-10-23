"""Lift-Matched Reweighting (paper §5, Eq. 9-13, Alg. 2).

C1 version holdout, C2 shrunken agent-human gap, C3 chained lift-matched weights.
Counts per comparison use the keys nA0, bA0, nA1, bA1, nH0, bH0, nH1, bH1
(channel A/H, version 0 = control, 1 = current; n = impressions, b = purchases).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COUNT_KEYS = ("nA0", "bA0", "nA1", "bA1", "nH0", "bH0", "nH1", "bH1")


@dataclass
class LMRConfig:
    eps_A: float = 0.50        # agent control share ε_A
    eps_H: float = 0.05        # human control share ε_H
    w_min: float = 0.1         # weight floor
    tau_min: float = 0.1       # prior-spread floor τ_min
    sigma_max: float = 1.0     # skip a comparison with σ_j > σ_max
    anchor_every: int = 4      # M_a; 0 disables re-anchoring
    # Ablation switches (Tab. 4/5); defaults are the paper's LMR.
    shrink: bool = True
    human_term: bool = True
    tau_floor: bool = True
    cap: bool = True
    floor: bool = True
    chain: bool = True
    prior: str = "ref"          # "ref" (fixed attack-free reference) or "current" (drifting alternative)


# ---------------------------------------------------------------- C1: version holdout

def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x ^ (x >> np.uint64(31))


def holdout_uniform(user, item, comparison, salt=0):
    """Deterministic U[0,1) hashed on (user, item, comparison) so later revisions are re-randomized."""
    with np.errstate(over="ignore"):
        h = _splitmix64(np.asarray(user, dtype=np.uint64) ^ np.uint64(salt))
        h = _splitmix64(h ^ np.asarray(item, dtype=np.uint64) * np.uint64(0x100000001B3))
        h = _splitmix64(h ^ np.asarray(comparison, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def holdout_assign(user, item, comparison, channel_is_agent, cfg: LMRConfig, salt=0):
    """True where the request sees the control version: probability ε_A for agents, ε_H for humans."""
    eps = np.where(np.asarray(channel_is_agent, bool), cfg.eps_A, cfg.eps_H)
    return holdout_uniform(user, item, comparison, salt) < eps


def gap_variance_approx(N_A, N_H, eps_A, eps_H, pi_A, pi_H):
    """Eq. 9: σ² ≈ 1/(N_A ε_A(1−ε_A) ω_A) + 1/(N_H ε_H(1−ε_H) ω_H), ω_c = π_c(1−π_c)."""
    wa, wh = pi_A * (1 - pi_A), pi_H * (1 - pi_H)
    return 1.0 / (N_A * eps_A * (1 - eps_A) * wa) + 1.0 / (N_H * eps_H * (1 - eps_H) * wh)


# ---------------------------------------------------------------- C2: shrunken gap

def haldane_logodds(b, n):
    """λ(b, n) = log((b + 1/2) / (n − b + 1/2))."""
    b, n = np.asarray(b, float), np.asarray(n, float)
    return np.log((b + 0.5) / (n - b + 0.5))


def haldane_var(b, n):
    b, n = np.asarray(b, float), np.asarray(n, float)
    return 1.0 / (b + 0.5) + 1.0 / (n - b + 0.5)


def gap_estimate(counts, human_term=True):
    """Eq. 10: ĝ = δ̂_A − δ̂_H with δ̂_c = λ(b_c1, n_c1) − λ(b_c0, n_c0), and σ² summed over (c, z)."""
    g = haldane_logodds(counts["bA1"], counts["nA1"]) - haldane_logodds(counts["bA0"], counts["nA0"])
    v = haldane_var(counts["bA1"], counts["nA1"]) + haldane_var(counts["bA0"], counts["nA0"])
    if human_term:
        g = g - (haldane_logodds(counts["bH1"], counts["nH1"]) - haldane_logodds(counts["bH0"], counts["nH0"]))
        v = v + haldane_var(counts["bH1"], counts["nH1"]) + haldane_var(counts["bH0"], counts["nH0"])
    return g, v


@dataclass
class ShrinkResult:
    g_tilde: np.ndarray     # posterior means (0 where unusable)
    usable: np.ndarray      # σ_j ≤ σ_max
    B: np.ndarray           # shrinkage factors
    mu: float
    tau2: float | None      # None when no current comparison is eligible


def shrink_gaps(g_hat, var, g_ref, var_ref, cfg: LMRConfig) -> ShrinkResult:
    """Eq. 11 with the prior of §5.2 and Alg. 2 lines 2-4.

    μ_G: median of eligible reference gaps (0 if none) — fixed attack-free period (cfg.prior="ref"),
    or the current-period median (cfg.prior="current", the drifting alternative of Tab. 5).
    τ²_G = max{τ²_min, mean_j[(ĝ_j − μ_G)² − σ²_j]} over eligible current comparisons.
    """
    g_hat, var = np.asarray(g_hat, float), np.asarray(var, float)
    g_ref = np.asarray(g_ref, float)
    var_ref = np.asarray(var_ref, float) if var_ref is not None else np.zeros_like(g_ref)
    smax2 = cfg.sigma_max ** 2
    usable = var <= smax2
    ok_ref = var_ref <= smax2
    if not usable.any():
        z = np.zeros_like(g_hat)
        return ShrinkResult(z, usable, z.copy(), 0.0, None)
    if cfg.prior == "ref":
        mu = float(np.median(g_ref[ok_ref])) if ok_ref.any() else 0.0
    elif cfg.prior == "current":
        mu = float(np.median(g_hat[usable]))
    else:
        raise ValueError(f"unknown prior {cfg.prior!r}")
    tau2 = float(np.mean((g_hat[usable] - mu) ** 2 - var[usable]))
    tau2 = max(tau2, cfg.tau_min ** 2 if cfg.tau_floor else 1e-12)
    B = tau2 / (tau2 + var) if cfg.shrink else np.ones_like(var)
    g_tilde = np.where(usable, mu + B * (g_hat - mu), 0.0)
    return ShrinkResult(g_tilde, usable, np.where(usable, B, 0.0), mu, tau2)


# ---------------------------------------------------------------- C3: chained weights

def weight_from_sum(sigma, cfg: LMRConfig):
    """Eq. 12: w = min{1, max{w_min, e^{−Σ}}} (cap and floor switchable for ablations)."""
    w = np.exp(-np.asarray(sigma, float))
    if cfg.cap:
        w = np.minimum(w, 1.0)
    if cfg.floor:
        w = np.maximum(w, cfg.w_min)
    return w


@dataclass
class Comparison:
    comparison_id: int
    item: int
    control_version: int
    current_version: int
    is_anchor: bool
    category: int


@dataclass
class _ItemChain:
    sigma: dict = field(default_factory=lambda: {0: 0.0})   # Σ_{j,r} per version
    current: int = 0
    anchor: int = 0
    retry_anchor: bool = False
    last_gt: float = 0.0       # latest shrunken gap (used when chaining is disabled)


class LMRState:
    """Per-item version chains, anchors and retries (Eq. 12, §5.3, Alg. 2 lines 5-9)."""

    def __init__(self, cfg: LMRConfig):
        self.cfg = cfg
        self.items: dict[int, _ItemChain] = {}
        self._next_id = 1

    def chain(self, item) -> _ItemChain:
        if item not in self.items:
            self.items[item] = _ItemChain()
        return self.items[item]

    def open_comparison(self, item, category, new_version=None) -> Comparison:
        """C1: open a holdout for item's current version (after a revision, or an anchor retry).

        The control is the predecessor, or the retained anchor every M_a versions / on a retry.
        """
        ch = self.chain(item)
        if new_version is not None:
            if new_version != ch.current + 1:
                raise ValueError("versions must be consecutive")
            ch.current = new_version
        r = ch.current
        scheduled = bool(self.cfg.anchor_every) and r % self.cfg.anchor_every == 0
        is_anchor = bool(self.cfg.anchor_every) and (scheduled or ch.retry_anchor) and r > 0
        control = ch.anchor if is_anchor else r - 1
        cmp_ = Comparison(self._next_id, item, control, r, is_anchor, category)
        self._next_id += 1
        return cmp_

    def close(self, comparisons, shrink: ShrinkResult):
        """Update Σ for closing comparisons; returns the list of items needing an anchor retry."""
        retries = []
        for k, c in enumerate(comparisons):
            ch = self.chain(c.item)
            usable = bool(shrink.usable[k])
            gt = float(shrink.g_tilde[k])
            ch.last_gt = gt if usable else 0.0   # per-revision (unchained) weight of the latest version
            if usable:
                ch.sigma[c.current_version] = ch.sigma.get(c.control_version, 0.0) + gt
                if c.is_anchor:
                    ch.anchor = c.current_version
                    ch.retry_anchor = False
            else:
                # A skipped revision retains the preceding sum.
                ch.sigma[c.current_version] = ch.sigma.get(c.current_version - 1, 0.0)
                if c.is_anchor:
                    ch.retry_anchor = True
                    retries.append(c.item)
        return retries

    def weight(self, item, version):
        """w_{j,r} for an agent purchase of item j shown version r (1 for never-revised items)."""
        ch = self.items.get(item)
        if ch is None:
            return 1.0
        if self.cfg.chain:
            v = version
            while v not in ch.sigma and v > 0:   # pending versions inherit their predecessor
                v -= 1
            s = ch.sigma.get(v, 0.0)
        else:
            s = ch.last_gt if version > 0 else 0.0
        return float(weight_from_sum(s, self.cfg))


def run_round(comparisons, counts, reference, state: LMRState):
    """Alg. 2 lines 1-6 for one round: estimate, shrink per category, chain.

    reference: {category: (ĝ, σ²)} of the fixed attack-free reference revisions R^ref_G.
    """
    cfg = state.cfg
    g_hat, var = gap_estimate(counts, cfg.human_term)
    out_gt = np.zeros(len(comparisons))
    out_ok = np.zeros(len(comparisons), bool)
    cats = np.array([c.category for c in comparisons])
    retries = []
    for cat in np.unique(cats):
        idx = np.where(cats == cat)[0]
        rg, rv = reference.get(int(cat), (np.array([]), np.array([])))
        res = shrink_gaps(g_hat[idx], var[idx], rg, rv, cfg)
        retries += state.close([comparisons[i] for i in idx], res)
        out_gt[idx], out_ok[idx] = res.g_tilde, res.usable
    return dict(g_hat=g_hat, var=var, g_tilde=out_gt, usable=out_ok, retries=retries)


# ---------------------------------------------------------------- Proposition 2

def lift_matched_check(agent_utils, leave_utility, revised_mask, g, loss):
    """Eq. 13 on one context ξ: returns (E_{P_A}[w(Y) ℓ(Y)], C(ξ) E_Q[ℓ(Y)]) with w_j = e^{−g_j}.

    agent_utils: (K,) agent utilities; revised_mask: (K,) bool; g: (K,) true gaps (ignored where
    not revised); loss: (K+1,) losses ℓ(ξ, y) with the leave option last.
    """
    u = np.asarray(agent_utils, float)
    w = np.where(revised_mask, np.exp(-np.asarray(g, float)), 1.0)
    z = np.append(u, leave_utility)
    pa = np.exp(z - z.max())
    pa /= pa.sum()
    wy = np.append(w, 1.0)
    lhs = float(np.sum(pa * wy * loss))
    zq = np.append(u - np.where(revised_mask, g, 0.0), leave_utility)   # human-matched utilities
    q = np.exp(zq - zq.max())
    q /= q.sum()
    C = 1.0 - float(np.sum((1 - w) * pa[:-1]))
    return lhs, C * float(np.sum(q * loss))
