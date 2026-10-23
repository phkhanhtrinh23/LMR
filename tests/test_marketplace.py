from dataclasses import replace

import numpy as np
import pytest
import torch

from proxyattack.marketplace import Catalog, World, WorldConfig, excess, run_world
from proxyattack.rankers import Batch, make_ranker, slate_loss

TINY = replace(WorldConfig(), n_users=200, n_cats=2, items_per_cat=30, requests_per_round=1200, rounds=2,
               eval_requests=600, attack_rank=(5, 20), ranker="mf")


def test_catalog_shared_and_attacked_in_rank_window():
    a, b = Catalog(TINY), Catalog(TINY)
    assert np.array_equal(a.attacked, b.attacked) and len(a.attacked) == 4   # 2 per category
    assert np.allclose(a.incA, b.incA)
    assert abs(a.propensity.mean() - TINY.s) < 0.05


def test_hm_world_replaces_agent_response():
    cat = Catalog(TINY)
    hm = World(cat, TINY, "hm")
    act = World(cat, TINY, "actual")
    assert np.allclose(hm.cumA, hm.cumH) and np.allclose(act.cumH, hm.cumH)
    assert (act.cumA[cat.attacked] > act.cumH[cat.attacked]).any()


def test_leave_calibration_hits_purchase_rate():
    cat = Catalog(TINY)
    w = World(cat, TINY)
    w.calibrate_leave()
    w.run_round(0)
    ev = w.stacked()
    assert abs(np.mean(ev["choice"] < TINY.slate_size) - TINY.purchase_rate) < 0.08


@pytest.mark.parametrize("policy", ["drop", "facts_only"])
def test_zero_excess_by_construction(policy):
    cat = Catalog(TINY)
    a = run_world(TINY, "actual", policy, cat)
    h = run_world(TINY, "hm", policy, cat)
    assert all(v == 0.0 for v in excess(a, h).values())


def test_lmr_weights_downweight_attacked_agent_purchases():
    cfg = replace(TINY, rounds=1)
    cat = Catalog(cfg)
    w = World(cat, cfg, "actual", "lmr")
    w.calibrate_leave()
    for r in range(2):
        w.run_round(r)
    weights = [w.lmr.weight(int(j), 1) for j in cat.attacked]
    assert np.mean(weights) < 0.9
    ev = w.stacked()
    wt = w.event_weights(ev, 1)
    assert np.all(wt[ev["channel"] == 0] == 1.0) and wt.min() >= cfg.lmr.w_min


@pytest.mark.parametrize("name", ["mf", "lightgcn", "sasrec", "textmf"])
def test_rankers_slate_loss_backprop_and_weights(name):
    torch.manual_seed(0)
    text = np.random.default_rng(0).standard_normal((20, 4, 8))
    m = make_ranker(name, 10, 20, 8, text_feats=text, max_len=5)
    if name == "lightgcn":
        m.set_graph(np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([1.0, 1.0, 0.5]))
    b = Batch(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([[1, 2, 3], [4, 5, 6]]),
              torch.zeros(2, 3, dtype=torch.long), torch.tensor([0, 3]), torch.tensor([1.0, 0.0]),
              torch.tensor([[-1, -1, -1, 2, 3], [-1, -1, -1, -1, -1]]))
    loss = slate_loss(m, b, m.prepare())
    loss.backward()
    assert torch.isfinite(loss) and m.leave.grad is not None
    # Zero-weight events contribute nothing: loss equals the first event's NLL / batch size.
    lp = m.slate_log_prob(b, m.prepare())
    assert torch.isclose(loss, -lp[0] / 2)
