"""GPA selector verification on known Bernoulli banks (paper §C.1, Fig. 7, §6.2 last paragraph).

Two agents, two contexts, two deterministic surrogates; candidates `shared` (raises agents and
surrogates to 0.6), `specific` (agents only, 0.5), `brittle` (one agent only) and `negative`
(lowers both). Scenarios: separation, heterogeneous (unequal context weights/baselines) and
misspecified_human (the real human response to `specific` is 0.8, outside the surrogates).
"""
from __future__ import annotations

import numpy as np

from .gpa import (GPAConfig, ResponseBank, contrast, max_share_select, point_gap_select, select,
                  single_agent_bank)

CANDIDATES = ["shared", "specific", "brittle", "negative"]
METHODS = ["max_share", "point_gap", "GPA", "single_agent_GPA"]


def fixture(scenario):
    het = scenario == "heterogeneous"
    base = [.1, .3] if het else [.2, .2]
    shared = [.45, .7] if het else [.6, .6]
    specific = [.3, .58] if het else [.5, .5]
    brittle1 = [.65, .85] if het else [.75, .75]
    brittle2 = [.12, .34] if het else [.22, .22]
    negative = [.08, .22] if het else [.15, .15]
    agents = {"original": [base, base], "shared": [shared, shared], "specific": [specific, specific],
              "brittle": [brittle1, brittle2], "negative": [negative, negative]}
    human = {"original": [base, base], "shared": [shared, shared], "specific": [base, base],
             "brittle": [base, base], "negative": [[.05, .05], [.05, .05]]}
    actual = {k: [list(r) for r in v] for k, v in human.items()}
    if scenario == "misspecified_human":
        actual["specific"] = [[.8, .8], [.8, .8]]
    q = np.array([.35, .65])
    return ({k: np.array(v) for k, v in agents.items()}, {k: np.array(v) for k, v in human.items()},
            {k: np.array(v) for k, v in actual.items()}, q)


def true_scores(agents, human, actual, q, selected, zeta):
    if selected == "original":
        return 0.0, 0.0, 0.0
    da = contrast(agents[selected], agents["original"], q, zeta)
    dh = contrast(human[selected], human["original"], q, zeta)
    dh_act = contrast(actual[selected], actual["original"], q, zeta)
    return float(da.min() - dh.max()), float(da.min() - dh_act.max()), float(da.min())


def run(ns=(16, 64, 256, 1024, 4096), reps=200, seed=73129,
        scenarios=("separation", "heterogeneous", "misspecified_human")):
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in scenarios:
        agents, human, actual, q = fixture(scenario)
        for n in ns:
            cfg = GPAConfig(n=n, zeta=0.05, alpha=0.05, eta_A=0.05, gamma=0.1)
            agg = {m: dict(gap=[], actual=[], lift=[], chosen=[]) for m in METHODS}
            covered_n = violations = 0
            for _ in range(reps):
                counts = {k: rng.binomial(n, v) for k, v in agents.items()}
                bank = ResponseBank("original", list(CANDIDATES), ["a1", "a2"], ["h1", "h2"], q, counts, human, n)
                res = select(bank, cfg)
                chosen = {"GPA": res.selected, "max_share": max_share_select(bank),
                          "point_gap": point_gap_select(bank, cfg),
                          "single_agent_GPA": select(single_agent_bank(bank), cfg).selected}
                covered = all(np.all(np.abs(counts[k] / n - agents[k]) <= res.radius) for k in agents)
                covered_n += covered
                if res.accepted and covered:
                    g, _, la = true_scores(agents, human, actual, q, res.selected, cfg.zeta)
                    violations += (g < cfg.gamma) or (la < cfg.eta_A)
                for m, item in chosen.items():
                    g, ga, la = true_scores(agents, human, actual, q, item, cfg.zeta)
                    agg[m]["gap"].append(g); agg[m]["actual"].append(ga); agg[m]["lift"].append(la)
                    agg[m]["chosen"].append(item)
            for m, o in agg.items():
                rows.append(dict(scenario=scenario, n=n, method=m,
                                 surrogate_gap=float(np.mean(o["gap"])),
                                 actual_human_gap=float(np.mean(o["actual"])),
                                 min_agent_lift=float(np.mean(o["lift"])),
                                 acceptance=float(np.mean([c != "original" for c in o["chosen"]])),
                                 brittle_or_negative=float(np.mean([c in ("brittle", "negative") for c in o["chosen"]])),
                                 counts={c: o["chosen"].count(c) for c in ["original"] + CANDIDATES},
                                 coverage=covered_n / reps, certificate_violations=int(violations)))
    return rows
