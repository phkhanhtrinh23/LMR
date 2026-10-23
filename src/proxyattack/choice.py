"""Conditional-logit choice model, the agent-human gap and the pooled transfer shift (paper §3)."""
from __future__ import annotations

import numpy as np


def logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_choice_probs(utilities, leave_utility=0.0):
    """Eq. 1: P_c(y | ξ) over a slate plus the leave option.

    utilities: (..., K) utilities v_c(y; ξ) of the shown items.
    Returns (..., K + 1) probabilities; the last column is the leave option y = 0.
    """
    u = np.asarray(utilities, dtype=float)
    leave = np.broadcast_to(np.asarray(leave_utility, dtype=float), u.shape[:-1])[..., None]
    z = np.concatenate([u, leave], axis=-1)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def gap(delta_agent, delta_human):
    """Eq. 2: g_j = δ_Aj − δ_Hj."""
    return np.asarray(delta_agent) - np.asarray(delta_human)


def lift_to_logodds(base, lift):
    """A selection-share lift (probability units) at base share `base` as a log-odds utility shift.

    A target-only utility shift δ in Eq. 1 is exactly the log odds ratio of choosing the target, so
    harness lifts transport to other slates through this conversion (paper §3, §6.5).
    """
    return float(logit(base + lift) - logit(base))


def transfer_shift(pi_h, pi_a, p_theta_h, p_theta_a, s, delta_h, delta_a):
    """Eq. 4: first-order shift of the shared item term b_j under pooled training.

    pi_c: arrays of channel c's true choice probability of j on its logged contexts.
    p_theta_c: the ranker's predicted probability of j on the same contexts.
    Returns (Δb_j, Δb_j − Δb_j^HM, J_j).
    """
    omega_h = np.mean(pi_h * (1 - pi_h))
    omega_a = np.mean(pi_a * (1 - pi_a))
    J = (1 - s) * np.mean(p_theta_h * (1 - p_theta_h)) + s * np.mean(p_theta_a * (1 - p_theta_a))
    if J <= 0:
        raise ValueError("pooled curvature J_j must be positive")
    db = ((1 - s) * omega_h * delta_h + s * omega_a * delta_a) / J
    excess = s * omega_a * (delta_a - delta_h) / J
    return db, excess, J


def fit_pooled_item_term(base_utils_h, base_utils_a, leave_h, leave_a, delta_h, delta_a, s,
                         ranker_utils_h=None, ranker_utils_a=None, iters=100):
    """Exact unregularized pooled optimum of the shared item term b_j (§A.3) by Newton's method.

    Logged contexts are fixed: arrays (n_ctx, K) of channel utilities with item 0 the revised item j.
    The ranker's utilities default to the channel's pre-revision utilities (a ranker calibrated to
    both channels). Returns the b_j shift that zeroes the pooled score equation
    (1−s) E_H[π_H − p_θ] + s E_A[π_A − p_θ] = 0, used to check Eq. 4 numerically.
    """
    uh = np.array(base_utils_h, float)
    ua = np.array(base_utils_a, float)
    rh = uh.copy() if ranker_utils_h is None else np.array(ranker_utils_h, float)
    ra = ua.copy() if ranker_utils_a is None else np.array(ranker_utils_a, float)
    uh[:, 0] += delta_h
    ua[:, 0] += delta_a
    pih = logit_choice_probs(uh, leave_h)[:, 0]
    pia = logit_choice_probs(ua, leave_a)[:, 0]
    b = 0.0
    for _ in range(iters):
        th, ta = rh.copy(), ra.copy()
        th[:, 0] += b
        ta[:, 0] += b
        ph = logit_choice_probs(th, leave_h)[:, 0]
        pa = logit_choice_probs(ta, leave_a)[:, 0]
        score = (1 - s) * np.mean(pih - ph) + s * np.mean(pia - pa)
        curv = (1 - s) * np.mean(ph * (1 - ph)) + s * np.mean(pa * (1 - pa))
        step = score / curv
        b += step
        if abs(step) < 1e-12:
            break
    return b
