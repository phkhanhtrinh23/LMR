import numpy as np

from proxyattack.choice import sigmoid
from proxyattack.lmr import (LMRConfig, LMRState, gap_estimate, gap_variance_approx, haldane_logodds,
                             holdout_assign, holdout_uniform, lift_matched_check, run_round, shrink_gaps,
                             weight_from_sum)


def draw_counts(rng, n_rev, N, s, eA, eH, dA, dH, base=-3.0):
    imp = np.full(n_rev, int(N))
    nA = rng.binomial(imp, s); nH = imp - nA
    nA0 = rng.binomial(nA, eA); nH0 = rng.binomial(nH, eH)
    return dict(nA0=nA0, bA0=rng.binomial(nA0, sigmoid(base)), nA1=nA - nA0, bA1=rng.binomial(nA - nA0, sigmoid(base + dA)),
                nH0=nH0, bH0=rng.binomial(nH0, sigmoid(base)), nH1=nH - nH0, bH1=rng.binomial(nH - nH0, sigmoid(base + dH)))


def test_haldane():
    assert np.isclose(haldane_logodds(0, 10), np.log(0.5 / 10.5))


def test_holdout_shares_and_determinism():
    cfg = LMRConfig()
    users = np.arange(200000)
    ctrl_a = holdout_assign(users, 7, 3, True, cfg)
    ctrl_h = holdout_assign(users, 7, 3, False, cfg)
    assert abs(ctrl_a.mean() - 0.5) < 0.01 and abs(ctrl_h.mean() - 0.05) < 0.003
    assert np.array_equal(holdout_uniform(users[:50], 7, 3), holdout_uniform(users[:50], 7, 3))
    # A new comparison is randomized afresh.
    assert not np.array_equal(holdout_uniform(users[:50], 7, 3), holdout_uniform(users[:50], 7, 4))


def test_gap_estimate_unbiased_at_large_traffic_and_eq9_variance():
    rng = np.random.default_rng(0)
    N, s, eA, eH, dA, dH = 200000, 0.1, 0.5, 0.05, 0.6, 0.1
    c = draw_counts(rng, 400, N, s, eA, eH, dA, dH)
    g, v = gap_estimate(c)
    assert abs(g.mean() - 0.5) < 0.02
    pi = sigmoid(-3.0)
    approx = gap_variance_approx(N * s, N * (1 - s), eA, eH, pi, pi)
    assert 0.7 < g.var() / approx < 1.4
    assert 0.7 < v.mean() / approx < 1.4


def test_eq9_agent_term_dominates_shared_holdout():
    # §5.1: equal shares at s=10% -> agent term ~9x the human term.
    N, s, pi = 1e5, 0.1, 0.05
    a = 1 / (N * s * 0.05 * 0.95 * pi * (1 - pi))
    h = 1 / (N * (1 - s) * 0.05 * 0.95 * pi * (1 - pi))
    assert np.isclose(a / h, 9.0)


def test_shrinkage_formula_and_prior():
    cfg = LMRConfig()
    g = np.array([1.0, 0.0, 0.5]); v = np.array([0.1, 0.1, 2.0])
    ref = np.array([0.1, 0.2, 0.3])
    r = shrink_gaps(g, v, ref, np.zeros(3), cfg)
    assert r.mu == 0.2
    assert not r.usable[2] and r.g_tilde[2] == 0
    tau2 = max(cfg.tau_min ** 2, np.mean((g[:2] - 0.2) ** 2 - v[:2]))
    assert np.isclose(r.tau2, tau2)
    assert np.allclose(r.g_tilde[:2], 0.2 + tau2 / (tau2 + 0.1) * (g[:2] - 0.2))
    # No eligible reference -> mu = 0; no eligible current -> nothing usable.
    assert shrink_gaps(g, v, np.array([]), np.array([]), cfg).mu == 0.0
    assert not shrink_gaps(g, np.full(3, 5.0), ref, None, cfg).usable.any()


def test_weight_cap_and_floor():
    cfg = LMRConfig()
    assert weight_from_sum(-1.0, cfg) == 1.0
    assert np.isclose(weight_from_sum(0.5, cfg), np.exp(-0.5))
    assert weight_from_sum(10.0, cfg) == cfg.w_min


def _close(state, cmp_, gt, usable=True):
    from proxyattack.lmr import ShrinkResult
    return state.close([cmp_], ShrinkResult(np.array([gt]), np.array([usable]), np.array([1.0]), 0.0, 0.1))


def test_chaining_split_edits_cannot_reset():
    st = LMRState(LMRConfig(anchor_every=0))
    c1 = st.open_comparison(5, 0, new_version=1); _close(st, c1, 0.4)
    c2 = st.open_comparison(5, 0, new_version=2); _close(st, c2, 0.3)
    assert np.isclose(st.weight(5, 2), np.exp(-0.7))
    assert np.isclose(st.weight(5, 1), np.exp(-0.4))
    assert st.weight(5, 0) == 1.0 and st.weight(99, 3) == 1.0
    # Skipped revision retains the preceding sum.
    c3 = st.open_comparison(5, 0, new_version=3); _close(st, c3, 9.9, usable=False)
    assert np.isclose(st.weight(5, 3), np.exp(-0.7))


def test_anchor_replaces_intervening_sum_and_retries():
    st = LMRState(LMRConfig(anchor_every=2))
    c1 = st.open_comparison(1, 0, new_version=1); _close(st, c1, 0.2)
    c2 = st.open_comparison(1, 0, new_version=2)
    assert c2.is_anchor and c2.control_version == 0
    retries = _close(st, c2, 0.5, usable=False)
    assert retries == [1]
    retry = st.open_comparison(1, 0)          # anchor retry without a new revision
    assert retry.is_anchor and retry.control_version == 0 and retry.current_version == 2
    _close(st, retry, 0.35)
    assert np.isclose(st.weight(1, 2), np.exp(-0.35))   # anchor-to-current replaces 0.2 + ...
    c3 = st.open_comparison(1, 0, new_version=3)
    assert not c3.is_anchor and c3.control_version == 2


def test_run_round_per_category():
    rng = np.random.default_rng(1)
    st = LMRState(LMRConfig(anchor_every=0))
    comps = [st.open_comparison(i, i % 2, new_version=1) for i in range(40)]
    c = draw_counts(rng, 40, 50000, 0.1, 0.5, 0.05, 0.8, 0.1)
    ref = {0: (np.zeros(10), np.full(10, 0.01)), 1: (np.zeros(10), np.full(10, 0.01))}
    out = run_round(comps, c, ref, st)
    assert out["usable"].all()
    assert 0.4 < out["g_tilde"].mean() < 0.8
    assert all(st.weight(i, 1) < 0.8 for i in range(40))


def test_prop2_lift_matching_identity():
    rng = np.random.default_rng(2)
    for _ in range(50):
        K = 6
        u = rng.normal(size=K)
        revised = rng.random(K) < 0.5
        g = rng.normal(0.5, 0.5, size=K)
        loss = rng.normal(size=K + 1)
        lhs, rhs = lift_matched_check(u, 0.2, revised, g, loss)
        assert np.isclose(lhs, rhs)
