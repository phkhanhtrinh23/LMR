"""Gap-Targeted Proxy Attack: fixed-pool surrogate-gap selection (paper §4, Eq. 5-8, Alg. 1).

The selector consumes frozen response banks only: for every text (original + frozen candidates),
per-agent target-choice counts over the T matched contexts and per-surrogate predicted
probabilities. Generation, validation and querying live in `gpa_pipeline.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GPAConfig:
    n: int = 2048              # responses per agent/context/text cell
    zeta: float = 0.05         # clipping level ζ of ψ_ζ, 0 < ζ < 1/2
    alpha: float = 0.05        # simultaneous miscoverage
    eta_A: float = 0.05        # agent-lift gate η_A > 0
    gamma: float = 0.1         # surrogate-gap gate γ > 0

    def validate(self):
        if not (isinstance(self.n, int) and self.n > 0):
            raise ValueError("n must be a positive integer")
        if not 0 < self.zeta < 0.5:
            raise ValueError("zeta must lie in (0, 1/2)")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if self.eta_A <= 0 or self.gamma <= 0:
            raise ValueError("eta_A and gamma must be positive")


@dataclass
class ResponseBank:
    """Frozen responses for one target item.

    agent_successes[text_id]: int array (|A|, T) of target choices out of n, or None if incomplete.
    surrogate_probs[text_id]: float array (|H|, T) of surrogate target-choice probabilities.
    """
    original: str
    candidates: list[str]               # frozen pool C, in the fixed tie order
    agents: list[str]
    surrogates: list[str]
    context_weights: np.ndarray         # q_t, nonnegative, sum to one
    agent_successes: dict[str, np.ndarray | None]
    surrogate_probs: dict[str, np.ndarray | None]
    n: int


@dataclass
class CandidateScore:
    text_id: str
    eligible: bool = False
    reasons: list[str] = field(default_factory=list)
    agent_point: np.ndarray | None = None   # d_a(x), point estimates
    agent_lower: np.ndarray | None = None   # L_a
    agent_upper: np.ndarray | None = None   # U_a
    surrogate_lift: np.ndarray | None = None  # d_h(x) = L_h = U_h
    gap_point: float = float("nan")         # G(x), Eq. 5
    gap_lower: float = float("nan")         # G̲(x), Eq. 7
    gap_upper: float = float("nan")         # Ḡ(x), Eq. 7


@dataclass
class GPAResult:
    selected: str
    accepted: bool
    radius: float
    num_cells: int
    scores: list[CandidateScore]
    reasons: list[str] = field(default_factory=list)
    regret_bound: float | None = None   # max_{x∈E}[Ḡ(x) − G̲(x)], Prop. 1


def psi(p, zeta):
    """ψ_ζ(p) = logit(clip(p, ζ, 1 − ζ))."""
    p = np.clip(np.asarray(p, dtype=float), zeta, 1 - zeta)
    return np.log(p) - np.log1p(-p)


def contrast(p_text, p_orig, q, zeta):
    """Eq. 5: d_m(x) = Σ_t q_t [ψ_ζ(p_{m,t}(x)) − ψ_ζ(p_{m,t}(x_0))], vectorized over leading axes."""
    return np.sum(np.asarray(q) * (psi(p_text, zeta) - psi(p_orig, zeta)), axis=-1)


def surrogate_gap(agent_lifts, surrogate_lifts):
    """Eq. 5: G(x) = min_a d_a(x) − max_h d_h(x)."""
    return float(np.min(agent_lifts) - np.max(surrogate_lifts))


def hoeffding_radius(num_cells, n, alpha):
    """Eq. 6: r = sqrt(log(2M/α) / (2n))."""
    return math.sqrt(math.log(2 * num_cells / alpha) / (2 * n))


def intervals(p_hat, r):
    """Eq. 6: [p⁻, p⁺] = [max(0, p̂ − r), min(1, p̂ + r)]."""
    p_hat = np.asarray(p_hat, dtype=float)
    return np.maximum(0.0, p_hat - r), np.minimum(1.0, p_hat + r)


def _complete(arr, shape, n=None):
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        return False
    if n is not None:
        return bool(np.all((arr >= 0) & (arr <= n)))
    return bool(np.all((arr >= 0) & (arr <= 1)))


def check_bank(bank: ResponseBank):
    q = np.asarray(bank.context_weights, dtype=float)
    if q.ndim != 1 or len(q) == 0 or np.any(q < 0) or not math.isclose(q.sum(), 1.0, abs_tol=1e-9):
        raise ValueError("context weights must be nonnegative and sum to one")
    if not bank.agents or not bank.surrogates:
        raise ValueError("need at least one scoring agent and one surrogate")
    if len(set(bank.candidates)) != len(bank.candidates) or bank.original in bank.candidates:
        raise ValueError("candidate ids must be unique and differ from the original")


def score_candidates(bank: ResponseBank, cfg: GPAConfig):
    """Eq. 5-7 for every frozen candidate. Returns (scores, radius, M, original_ok)."""
    cfg.validate()
    check_bank(bank)
    if bank.n != cfg.n:
        raise ValueError("bank replicate count differs from the configured n")
    A, H, T = len(bank.agents), len(bank.surrogates), len(bank.context_weights)
    q = np.asarray(bank.context_weights, dtype=float)
    M = A * T * (len(bank.candidates) + 1)
    r = hoeffding_radius(M, cfg.n, cfg.alpha)

    base_c = bank.agent_successes.get(bank.original)
    base_h = bank.surrogate_probs.get(bank.original)
    if not (_complete(base_c, (A, T), cfg.n) and _complete(base_h, (H, T))):
        return [], r, M, False
    base_p = np.asarray(base_c, float) / cfg.n
    base_lo, base_hi = intervals(base_p, r)
    base_h = np.asarray(base_h, float)

    scores = []
    for cid in bank.candidates:
        sc = CandidateScore(cid)
        cnt, hp = bank.agent_successes.get(cid), bank.surrogate_probs.get(cid)
        if not (_complete(cnt, (A, T), cfg.n) and _complete(hp, (H, T))):
            sc.reasons.append("incomplete_cells")
            scores.append(sc)
            continue
        p = np.asarray(cnt, float) / cfg.n
        lo, hi = intervals(p, r)
        sc.agent_point = contrast(p, base_p, q, cfg.zeta)
        # Endpoints propagate through the monotone clipped logit:
        # L_a = candidate lower − original upper, U_a = candidate upper − original lower.
        sc.agent_lower = contrast(lo, base_hi, q, cfg.zeta)
        sc.agent_upper = contrast(hi, base_lo, q, cfg.zeta)
        sc.surrogate_lift = contrast(np.asarray(hp, float), base_h, q, cfg.zeta)
        # Frozen surrogates set L_h = U_h = d_h.
        dh_max = float(np.max(sc.surrogate_lift))
        sc.gap_point = float(np.min(sc.agent_point) - dh_max)
        sc.gap_lower = float(np.min(sc.agent_lower) - dh_max)   # min_a L_a − max_h U_h
        sc.gap_upper = float(np.min(sc.agent_upper) - dh_max)   # min_a U_a − max_h L_h
        if np.min(sc.agent_lower) < cfg.eta_A:
            sc.reasons.append("agent_lift_gate")
        if sc.gap_lower < cfg.gamma:
            sc.reasons.append("gap_gate")
        sc.eligible = not sc.reasons
        scores.append(sc)
    return scores, r, M, True


def select(bank: ResponseBank, cfg: GPAConfig) -> GPAResult:
    """Eq. 8: argmax of G̲ over eligible candidates with the fixed tie order, else the original."""
    scores, r, M, ok = score_candidates(bank, cfg)
    if not ok:
        return GPAResult(bank.original, False, r, M, scores, ["incomplete_original_cell"])
    eligible = [s for s in scores if s.eligible]
    if not eligible:
        return GPAResult(bank.original, False, r, M, scores, ["no_eligible_candidate"])
    order = {cid: i for i, cid in enumerate(bank.candidates)}
    best = min(eligible, key=lambda s: (-s.gap_lower, order[s.text_id]))
    regret = max(s.gap_upper - s.gap_lower for s in eligible)
    return GPAResult(best.text_id, True, r, M, scores, [], regret)


def target_share(successes, n, q):
    """Estimated target-choice share of a text: context-weighted, averaged over agents."""
    return float(np.mean(np.sum(np.asarray(q) * np.asarray(successes, float) / n, axis=-1)))


def max_share_select(bank: ResponseBank):
    """§4.1 baseline: the text in C ∪ {x_0} with the highest estimated target-choice share."""
    q = np.asarray(bank.context_weights, float)
    best, best_share = bank.original, target_share(bank.agent_successes[bank.original], bank.n, q)
    for cid in bank.candidates:
        cnt = bank.agent_successes.get(cid)
        if cnt is None:
            continue
        sh = target_share(cnt, bank.n, q)
        if sh > best_share:
            best, best_share = cid, sh
    return best


def point_gap_select(bank: ResponseBank, cfg: GPAConfig):
    """Fig. 7 ablation: the largest point gap G(x) passing the gates on point estimates, no bounds."""
    scores, _, _, ok = score_candidates(bank, cfg)
    if not ok:
        return bank.original
    order = {cid: i for i, cid in enumerate(bank.candidates)}
    ok_rows = [s for s in scores if s.agent_point is not None
               and np.min(s.agent_point) >= cfg.eta_A and s.gap_point >= cfg.gamma]
    if not ok_rows:
        return bank.original
    return min(ok_rows, key=lambda s: (-s.gap_point, order[s.text_id])).text_id


def single_agent_bank(bank: ResponseBank, agent_index: int = 0) -> ResponseBank:
    """Fig. 7 ablation: GPA scoring one agent only."""
    sub = {k: (None if v is None else np.asarray(v)[[agent_index]]) for k, v in bank.agent_successes.items()}
    return ResponseBank(bank.original, list(bank.candidates), [bank.agents[agent_index]],
                        list(bank.surrogates), bank.context_weights, sub, bank.surrogate_probs, bank.n)
