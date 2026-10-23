import json
from dataclasses import replace

import numpy as np
import torch

from proxyattack.agent_fit import fit_slate_logit, make_slates
from proxyattack.datasets import fit_choice_model, load_market
from proxyattack.marketplace import Catalog, World, config_for, excess, run_world


def test_choice_fit_recovers_item_effects_and_beats_popularity():
    rng = np.random.default_rng(0)
    n_items, n_users, d = 40, 300, 4
    group = np.repeat([0, 1], 20)
    b = rng.normal(0, 1, n_items)
    P, Q = rng.normal(0, 1, (n_users, d)), rng.normal(0, 1, (n_items, d))
    users, items = [], []
    for _ in range(30000):
        u, g = rng.integers(n_users), rng.integers(2)
        it = np.where(group == g)[0]
        v = b[it] + P[u] @ Q[it].T
        p = np.exp(v - v.max()); p /= p.sum()
        users.append(u); items.append(rng.choice(it, p=p))
    bh, Ph, Qh, diag = fit_choice_model(np.array(users), np.array(items), group, n_users, n_items, dim=4,
                                        epochs=40, device="cpu")
    for g in (0, 1):
        m = group == g
        assert np.corrcoef(bh[m], b[m])[0, 1] > 0.8
    assert diag["val_nll_full"] < diag["val_nll_popularity"] < diag["val_nll_uniform"]


def test_slate_logit_recovers_agent_utilities():
    rng = np.random.default_rng(1)
    item_cat = np.repeat([0, 1], 15)
    beta = rng.normal(0, 1, 30)
    slates = make_slates(item_cat, 300, k=6)
    pos = np.array([0.3, 0.1, 0, 0, -0.1, -0.3])
    v = beta[slates] + pos
    probs = np.exp(v) / np.exp(v).sum(1, keepdims=True)
    bh, ph, _ = fit_slate_logit(slates, probs, 30, l2=1e-6)
    for c in (0, 1):
        m = item_cat == c
        assert np.allclose(bh[m] - bh[m].mean(), beta[m] - beta[m].mean(), atol=0.05)
    assert np.allclose(ph, pos - pos.mean(), atol=0.05)


def _tiny_market(tmp_path, n_users=120, n_cats=2, per_cat=24, with_llm=True):
    rng = np.random.default_rng(3)
    n = n_cats * per_cat
    np.savez(tmp_path / "market.npz", item_cat=np.repeat(np.arange(n_cats), per_cat), quality=rng.normal(0, 1, n),
             Q=rng.normal(0, 0.5, (n, 4)), P=rng.normal(0, 0.5, (n_users, 4)))
    items = [dict(id=str(i), title=f"item {i}", text="", facts={}, category=str(i // per_cat)) for i in range(n)]
    (tmp_path / "items.json").write_text(json.dumps(dict(cat_names=["a", "b"][:n_cats], items=items)))
    (tmp_path / "fit.json").write_text("{}")
    np.save(tmp_path / "text_emb.npy", rng.normal(0, 1, (n, 6)))
    np.save(tmp_path / "agent_quality.npy", rng.normal(0, 1, n))
    if with_llm:
        np.save(tmp_path / "llm_emb.npy", rng.normal(0, 1, (n, 10)))
    load_market.cache_clear()
    return str(tmp_path)


def test_gpa_success_scenario_effects(tmp_path):
    cfg = replace(config_for(_tiny_market(tmp_path), "gpa_success"), attack_rank=(3, 15), rounds=2)
    cat = Catalog(cfg)
    att = [(j, v) for (j, v), k in cat.kind.items() if k == "attack"]
    ben = [(j, v) for (j, v), k in cat.kind.items() if k == "benign"]
    assert att and ben
    assert all(cat.incH[j, v] == 0.0 and cat.incA[j, v] > 0.3 for j, v in att)        # δ_H = 0, δ_A > 0
    gaps = np.array([cat.incA[j, v] - cat.incH[j, v] for j, v in ben])
    assert np.all(np.abs(gaps) < 0.3) and np.mean([cat.incH[j, v] for j, v in ben]) > 0.15   # δ_A ≈ δ_H > 0
    assert not np.allclose(cat.agent_quality, cat.quality)                              # harness-fitted β_A
    assert cat.text.shape[-1] == 6


def test_data_world_runs_lmr_logs_estimates_and_lorec_llm(tmp_path):
    cfg = replace(config_for(_tiny_market(tmp_path), "gpa_success"), attack_rank=(3, 15), rounds=2,
                  requests_per_round=1500, eval_requests=500, ranker="mf")
    cat = Catalog(cfg)
    a = run_world(cfg, "actual", "lmr", cat)
    assert a["estimates"] and {"attack", "benign"} <= {e["kind"] for e in a["estimates"]}
    w = World(cat, cfg, "actual", "lorec")
    w.run_round(0); w.train(0); w.run_round(1)
    ev = w.stacked()
    inc = w._llm_incoherence(ev, np.where(ev["round"] == 1)[0], np.bincount(ev["user"], minlength=cfg.n_users) > 0)
    assert inc.shape == (cfg.n_users,) and np.isfinite(inc).all()
    wt = w.event_weights(ev, 1)
    assert wt.min() >= 0.2 and wt.max() <= 1.0
    h = run_world(cfg, "hm", "drop", cat)
    assert all(v == 0.0 for v in excess(run_world(cfg, "actual", "drop", cat), h).values())
