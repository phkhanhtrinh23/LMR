import math

import numpy as np
import pytest

from proxyattack.gpa import (GPAConfig, ResponseBank, contrast, hoeffding_radius, max_share_select,
                             point_gap_select, psi, select, single_agent_bank, surrogate_gap)


def bank_from_probs(agent_p, human_p, n, q=(0.5, 0.5), original="x0"):
    cands = [k for k in agent_p if k != original]
    return ResponseBank(original, cands, ["a1", "a2"], ["h1", "h2"], np.array(q),
                        {k: np.round(np.asarray(v) * n).astype(int) for k, v in agent_p.items()},
                        {k: np.asarray(v, float) for k, v in human_p.items()}, n)


def two(p, T=2):
    return np.full((2, T), p)


def test_psi_clips():
    assert np.isclose(psi(0.0, 0.05), math.log(0.05 / 0.95))
    assert np.isclose(psi(0.5, 0.05), 0.0)


def test_paper_example_gap_targeting_vs_max_share():
    # §4.3: from 0.2, agent/surrogate responses (0.6, 0.6) and (0.5, 0.2) give gaps 0 and log 4.
    q = np.array([1.0])
    g_shared = surrogate_gap([contrast([0.6], [0.2], q, 0.05)], [contrast([0.6], [0.2], q, 0.05)])
    g_spec = surrogate_gap([contrast([0.5], [0.2], q, 0.05)], [contrast([0.2], [0.2], q, 0.05)])
    assert np.isclose(g_shared, 0.0)
    assert np.isclose(g_spec, math.log(4))
    n = 4096
    b = bank_from_probs({"x0": two(.2), "shared": two(.6), "specific": two(.5)},
                        {"x0": two(.2), "shared": two(.6), "specific": two(.2)}, n)
    cfg = GPAConfig(n=n)
    assert select(b, cfg).selected == "specific"
    assert max_share_select(b) == "shared"
    assert point_gap_select(b, cfg) == "specific"


def test_hoeffding_radius_formula():
    assert np.isclose(hoeffding_radius(20, 256, 0.05), math.sqrt(math.log(800) / 512))


def test_small_budget_keeps_original():
    n = 16
    b = bank_from_probs({"x0": two(.2), "specific": two(.5)}, {"x0": two(.2), "specific": two(.2)}, n)
    res = select(b, GPAConfig(n=n))
    assert res.selected == "x0" and not res.accepted


def test_agent_lift_gate_blocks_surrogate_decline():
    # Agents unchanged, surrogates decline: G > 0 but no agent lift -> not eligible.
    n = 4096
    b = bank_from_probs({"x0": two(.2), "neg": two(.2)}, {"x0": two(.2), "neg": two(.05)}, n)
    res = select(b, GPAConfig(n=n))
    assert res.selected == "x0"
    assert "agent_lift_gate" in res.scores[0].reasons


def test_incomplete_cells():
    n = 1024
    b = bank_from_probs({"x0": two(.2), "c1": two(.5)}, {"x0": two(.2), "c1": two(.2)}, n)
    b.agent_successes["c1"] = None
    assert select(b, GPAConfig(n=n)).selected == "x0"
    b.agent_successes["c1"] = np.full((2, 2), 512)
    b.agent_successes["x0"] = None
    res = select(b, GPAConfig(n=n))
    assert res.selected == "x0" and "incomplete_original_cell" in res.reasons


def test_tie_order_and_bounds_ordering():
    n = 4096
    b = bank_from_probs({"x0": two(.2), "c1": two(.5), "c2": two(.5)},
                        {"x0": two(.2), "c1": two(.2), "c2": two(.2)}, n)
    res = select(b, GPAConfig(n=n))
    assert res.selected == "c1"
    for s in res.scores:
        assert s.gap_lower <= s.gap_point <= s.gap_upper
    assert res.regret_bound >= 0


def test_certificate_holds_on_covered_draws():
    """Prop. 1: whenever all cells are covered, a returned candidate satisfies both gates on true lifts."""
    rng = np.random.default_rng(3)
    n, cfg = 512, GPAConfig(n=512)
    truth = {"x0": two(.2), "shared": two(.6), "specific": two(.5), "brittle": np.array([[.75, .75], [.22, .22]])}
    human = {"x0": two(.2), "shared": two(.6), "specific": two(.2), "brittle": two(.2)}
    q = np.array([.5, .5])
    violations = 0
    for _ in range(200):
        b = bank_from_probs(truth, human, n)
        b.agent_successes = {k: rng.binomial(n, v) for k, v in truth.items()}
        res = select(b, cfg)
        covered = all(np.all(np.abs(b.agent_successes[k] / n - truth[k]) <= res.radius) for k in truth)
        if res.accepted and covered:
            da = contrast(truth[res.selected], truth["x0"], q, cfg.zeta)
            dh = contrast(human[res.selected], human["x0"], q, cfg.zeta)
            violations += (da.min() < cfg.eta_A) or (da.min() - dh.max() < cfg.gamma)
    assert violations == 0


def test_single_agent_bank():
    n = 64
    b = bank_from_probs({"x0": two(.2), "c": two(.5)}, {"x0": two(.2), "c": two(.2)}, n)
    sb = single_agent_bank(b)
    assert sb.agents == ["a1"] and sb.agent_successes["c"].shape == (1, 2)


def test_config_validation():
    with pytest.raises(ValueError):
        GPAConfig(zeta=0.6).validate()
