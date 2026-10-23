import numpy as np

from proxyattack.choice import (fit_pooled_item_term, gap, lift_to_logodds, logit, logit_choice_probs,
                                transfer_shift)


def test_logit_probs_sum_to_one_and_leave_last():
    p = logit_choice_probs(np.array([[0.0, 1.0, -1.0]]), leave_utility=0.5)
    assert p.shape == (1, 4)
    assert np.allclose(p.sum(-1), 1)
    e = np.exp([0, 1, -1, 0.5])
    assert np.allclose(p[0], e / e.sum())


def test_target_shift_is_log_odds_ratio_on_every_slate():
    rng = np.random.default_rng(0)
    delta = 0.7
    for _ in range(20):
        u = rng.normal(size=6)
        p0 = logit_choice_probs(u, 0.3)
        u1 = u.copy()
        u1[0] += delta
        p1 = logit_choice_probs(u1, 0.3)
        # Under Eq. 1 the odds of j against any other option (here: leaving) scale by exactly e^δ.
        assert np.isclose(np.log(p1[0] / p1[-1]) - np.log(p0[0] / p0[-1]), delta)
        assert np.isclose(np.log(p1[1] / p1[-1]), np.log(p0[1] / p0[-1]))


def test_gap_and_lift_conversion():
    assert gap(1.0, 0.25) == 0.75
    assert np.isclose(lift_to_logodds(0.2, 0.3), logit(0.5) - logit(0.2))


def test_transfer_shift_matches_exact_pooled_refit_for_small_shifts():
    rng = np.random.default_rng(1)
    n_ctx, K, s = 400, 8, 0.1
    uh = rng.normal(size=(n_ctx, K))
    ua = rng.normal(size=(n_ctx, K))
    dh, da = 0.02, 0.10
    pih = logit_choice_probs(uh, 0.0)[:, 0]
    pia = logit_choice_probs(ua, 0.0)[:, 0]
    # Ranker calibrated to both channels before the revision: p_theta equals the channel's probability.
    db, excess, J = transfer_shift(pih, pia, pih, pia, s, dh, da)
    b_exact = fit_pooled_item_term(uh, ua, 0.0, 0.0, dh, da, s)
    b_hm = fit_pooled_item_term(uh, ua, 0.0, 0.0, dh, dh, s)
    assert np.isclose(db, b_exact, rtol=0.05)
    assert np.isclose(excess, b_exact - b_hm, rtol=0.05)
    assert excess > 0


def test_equal_curvature_reduces_excess_to_s_times_gap():
    pi = np.full(50, 0.2)
    _, excess, _ = transfer_shift(pi, pi, pi, pi, 0.1, 0.0, 0.5)
    assert np.isclose(excess, 0.1 * 0.5)
