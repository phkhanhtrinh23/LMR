"""LMR estimator study (paper §6.5-6.6, Fig. 5, Fig. 6, Tab. 4, Tab. 5, §C.3, §C.4, §D).

Pure statistics, no recommender: each replicate draws one category's revisions per holdout window
(15 attack, 60 benign, plus 60 attack-free reference revisions), Binomial holdout counts per channel
and version with heavy-tailed traffic, and runs LMR's estimator (Eq. 10-12) through `lmr.py`.

Lift removed = 1 − Σ_j n_j (g_j + log w_j) / Σ_j n_j g_j over attack revisions (n_j: agent
impressions of the current version). Effect sizes are the paper's RQ1 lifts converted to log odds
at the 12.5% base share.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .choice import lift_to_logodds, logit, sigmoid
from .lmr import (LMRConfig, LMRState, ShrinkResult, gap_estimate, gap_variance_approx, shrink_gaps,
                  weight_from_sum)

# RQ1 lifts (percentage points over a 1/8 base share), §6.2 / Fig. 3.
BASE = 0.125
LIFTS_PP = dict(att_A=float(np.mean([9.4, 5.1, 11.8])), att_H=0.9,
                ben_A=float(np.mean([2.3, 1.8, 2.6])), ben_H=1.4)


@dataclass(frozen=True)
class StudyConfig:
    N: float = 5e4            # median impressions per revision per window
    s: float = 0.10           # agent share
    eA: float = 0.50
    eH: float = 0.05
    n_att: int = 15
    n_ben: int = 60
    n_ref: int = 60
    het: float = 0.10         # sd of a revision's increment around its type mean
    pi: float = 0.05          # mean per-impression purchase probability
    pi_sd: float = 0.5        # sd of base log-odds across items
    n_sd: float = 1.0         # sd of log impressions (heavy-tailed traffic)
    steps: str = "single"     # single | twostep | small-k | honest-k | zero-k
    reps: int = 200
    lmr: LMRConfig = field(default_factory=LMRConfig)
    lifts_pp: tuple = tuple(sorted(LIFTS_PP.items()))
    undetected: float = 0.0   # share of agent traffic labelled human (§6.6, §D)
    mislabeled: float = 0.0   # share of human traffic labelled agent
    dup: int = 1              # count each impression dup times (§C.4 "duplication")
    scenario: str = "paper"   # "gpa_success": attacks δ_H = 0 exactly; benign δ_A = δ_H + N(0, ben_gap_sd²)
    ben_gap_sd: float = 0.05


def effects(cfg: StudyConfig):
    return {k: lift_to_logodds(BASE, v / 100) for k, v in dict(cfg.lifts_pp).items()}


GPA_SUCCESS_LIFTS_PP = dict(att_A=LIFTS_PP["att_A"], att_H=0.0, ben_A=3.0, ben_H=3.0)


def incr(rng, cfg: StudyConfig, D, key, n):
    """(agent, human) log-odds increments of n revisions of type key ("att" | "ben")."""
    zA, zH = rng.standard_normal(n), rng.standard_normal(n)
    a, h = D[f"{key}_A"] + cfg.het * zA, D[f"{key}_H"] + cfg.het * zH
    if cfg.scenario == "gpa_success":
        if key == "att":
            h = np.zeros(n)
        else:
            a = h + (D["ben_A"] - D["ben_H"]) + cfg.ben_gap_sd * zA
    return a, h


def draw_counts(rng, dA, dH, cfg: StudyConfig, N=None):
    """Holdout counts by labelled channel for revisions with agent/human increments dA, dH."""
    n = len(dA)
    N = cfg.N if N is None else N
    bA = logit(cfg.pi) + cfg.pi_sd * rng.standard_normal(n)
    bH = logit(cfg.pi) + cfg.pi_sd * rng.standard_normal(n)
    imp = rng.lognormal(np.log(N), cfg.n_sd, n).astype(np.int64)
    tA = rng.binomial(imp, cfg.s)
    tH = imp - tA
    aU = rng.binomial(tA, cfg.undetected)
    aD = tA - aU
    hF = rng.binomial(tH, cfg.mislabeled)
    hT = tH - hF
    c = {k: np.zeros(n, np.int64) for k in ("nA0", "bA0", "nA1", "bA1", "nH0", "bH0", "nH1", "bH1")}
    for m, base, d, ch, e in ((aD, bA, dA, "A", cfg.eA), (hF, bH, dH, "A", cfg.eA),
                              (hT, bH, dH, "H", cfg.eH), (aU, bA, dA, "H", cfg.eH)):
        m0 = rng.binomial(m, e)
        m1 = m - m0
        c[f"n{ch}0"] += m0
        c[f"b{ch}0"] += rng.binomial(m0, sigmoid(base))
        c[f"n{ch}1"] += m1
        c[f"b{ch}1"] += rng.binomial(m1, sigmoid(base + d))
    if cfg.dup > 1:
        c = {k: v * cfg.dup for k, v in c.items()}
    return c, dict(imp=imp, agent_det=aD, agent_und=aU)


def _plan(rng, cfg, D):
    """Per-window (agent, human) increments of the attack-side items."""
    n = cfg.n_att
    dA, dH = incr(rng, cfg, D, "att", n)
    kind, _, k = cfg.steps.partition("-")
    if kind == "single":
        return [(dA, dH)]
    if kind == "twostep":    # the attack, then a cosmetic edit one window later
        return [(dA, dH), (0.5 * cfg.het * rng.standard_normal(n), 0.5 * cfg.het * rng.standard_normal(n))]
    if kind == "small":      # k small steps that add up to the attack
        return [(dA / int(k), dH / int(k)) for _ in range(int(k))]
    if kind == "honest":     # an honest seller making k benign revisions
        return [incr(rng, cfg, D, "ben", n)
                for _ in range(int(k))]
    if kind == "zero":       # an honest seller whose k revisions move agents and humans alike
        out = []
        for _ in range(int(k)):
            x = D["ben_H"] + cfg.het * rng.standard_normal(n)
            out.append((x, x.copy()))
        return out
    raise ValueError(cfg.steps)


def replicate(rng, cfg: StudyConfig):
    D = effects(cfg)
    lc = cfg.lmr
    nat, nbe = cfg.n_att, cfg.n_ben
    plan = _plan(rng, cfg, D)
    # Attack-free reference period, drawn once per replicate and never rolled forward.
    rA, rH = incr(rng, cfg, D, "ben", cfg.n_ref)
    gr, vr = gap_estimate(draw_counts(rng, rA, rH, cfg)[0], lc.human_term)

    state = LMRState(lc)
    cum = {"A": np.zeros((nat, len(plan) + 1)), "H": np.zeros((nat, len(plan) + 1))}
    w_ben, o_ben, B_med, err = [], [], [], []
    last = None
    for i, (a, h) in enumerate(plan):
        cum["A"][:, i + 1] = cum["A"][:, i] + a
        cum["H"][:, i + 1] = cum["H"][:, i] + h
        comps = [state.open_comparison(j, 0, new_version=i + 1) for j in range(nat)]
        ctrl = np.array([c.control_version for c in comps])
        tA = cum["A"][np.arange(nat), i + 1] - cum["A"][np.arange(nat), ctrl]
        tH = cum["H"][np.arange(nat), i + 1] - cum["H"][np.arange(nat), ctrl]
        ca, det = draw_counts(rng, tA, tH, cfg)
        bA, bH = incr(rng, cfg, D, "ben", nbe)
        cb, _ = draw_counts(rng, bA, bH, cfg)
        ga, va = gap_estimate(ca, lc.human_term)
        gb, vb = gap_estimate(cb, lc.human_term)
        res = shrink_gaps(np.concatenate([ga, gb]), np.concatenate([va, vb]), gr, vr, lc)
        state.close(comps, ShrinkResult(res.g_tilde[:nat], res.usable[:nat], res.B[:nat], res.mu, res.tau2))
        err.append(np.concatenate([ga - (tA - tH), gb - (bA - bH)]))
        w_ben.append(weight_from_sum(res.g_tilde[nat:], lc).mean())
        o_ben.append(np.clip(np.exp(-(bA - bH)), lc.w_min, 1).mean())
        B_med.append(np.median(res.B))
        last = (ca, det)
    g_true = cum["A"][:, -1] - cum["H"][:, -1]
    w = np.array([state.weight(j, len(plan)) for j in range(nat)])
    ca, det = last
    if cfg.undetected == 0 and cfg.mislabeled == 0:
        nd, nu = ca["nA1"].astype(float), np.zeros(nat)
    else:   # detected agents get weight w; undetected agents' purchases keep weight 1
        nd, nu = det["agent_det"].astype(float), det["agent_und"].astype(float)
    denom = np.sum((nd + nu) * g_true)
    resid = (np.sum(nd * (g_true + np.log(w))) + np.sum(nu * g_true)) / denom if abs(denom) > 1e-9 else np.nan
    return dict(removed=1 - resid, w_att=w.mean(), o_att=np.clip(np.exp(-g_true), lc.w_min, 1).mean(),
                w_ben=float(np.mean(w_ben)), o_ben=float(np.mean(o_ben)), B=float(np.mean(B_med)),
                err_sd=float(np.std(np.concatenate(err))),
                sd_formula=float(np.sqrt(gap_variance_approx(cfg.N * cfg.s, cfg.N * (1 - cfg.s), cfg.eA, cfg.eH,
                                                              cfg.pi, cfg.pi))),
                imp=det["imp"], w_vec=w, g_vec=g_true, nd=nd)


def replicate_rolling(rng, cfg: StudyConfig, windows=6, roll=True):
    """Tab. 5 row 2: sustained attacks; prior mean from a fixed reference vs a rolling one."""
    D = effects(cfg)
    lc = cfg.lmr
    nat, nbe = cfg.n_att, cfg.n_ben
    rA, rH = incr(rng, cfg, D, "ben", cfg.n_ref)
    g0, v0 = gap_estimate(draw_counts(rng, rA, rH, cfg)[0])
    ref, refv = list(g0), list(v0)
    for _ in range(windows):
        dA, dH = incr(rng, cfg, D, "att", nat)
        bA, bH = incr(rng, cfg, D, "ben", nbe)
        ga, va = gap_estimate(draw_counts(rng, dA, dH, cfg)[0])
        gb, vb = gap_estimate(draw_counts(rng, bA, bH, cfg)[0])
        gh, v = np.concatenate([ga, gb]), np.concatenate([va, vb])
        res = shrink_gaps(gh, v, np.array(ref), np.array(refv), lc)
        if roll:
            ref += list(gh)
            refv += list(v)
    w = weight_from_sum(res.g_tilde, lc)
    g_true = dA - dH
    resid = np.sum(g_true + np.log(w[:nat])) / np.sum(g_true)
    return dict(removed=1 - resid, w_att=w[:nat].mean(), w_ben=w[nat:].mean(), mu=res.mu)


# ---------------------------------------------------------------- §C.3 heterogeneous contexts

def replicate_contexts(rng, cfg: StudyConfig, stratified, strata=5, spread=1.0):
    D = effects(cfg)
    lc = cfg.lmr

    def counts_ctx(dA, dH):
        n = len(dA)
        off = np.linspace(-spread, spread, strata) * np.sqrt(3 / 2)
        bA = logit(cfg.pi) + cfg.pi_sd * rng.standard_normal(n)
        bH = logit(cfg.pi) + cfg.pi_sd * rng.standard_normal(n)
        imp = rng.lognormal(np.log(cfg.N), cfg.n_sd, n).astype(np.int64)
        out = {k: np.zeros((n, strata), np.int64) for k in ("nA0", "bA0", "nA1", "bA1", "nH0", "bH0", "nH1", "bH1")}
        for i in range(n):
            split = rng.multinomial(imp[i], np.full(strata, 1 / strata))
            nA = rng.binomial(split, cfg.s)
            nH = split - nA
            nA0 = rng.binomial(nA, cfg.eA)
            nH0 = rng.binomial(nH, cfg.eH)
            out["nA0"][i], out["nA1"][i], out["nH0"][i], out["nH1"][i] = nA0, nA - nA0, nH0, nH - nH0
            out["bA0"][i] = rng.binomial(nA0, sigmoid(bA[i] + off))
            out["bA1"][i] = rng.binomial(nA - nA0, sigmoid(bA[i] + off + dA[i]))
            out["bH0"][i] = rng.binomial(nH0, sigmoid(bH[i] + off))
            out["bH1"][i] = rng.binomial(nH - nH0, sigmoid(bH[i] + off + dH[i]))
        return out

    def est(c):
        if not stratified:
            return gap_estimate({k: v.sum(axis=1) for k, v in c.items()})
        from .lmr import haldane_logodds, haldane_var
        g, v = 0.0, 0.0
        for ch, sign in (("A", 1), ("H", -1)):
            d = haldane_logodds(c[f"b{ch}1"], c[f"n{ch}1"]) - haldane_logodds(c[f"b{ch}0"], c[f"n{ch}0"])
            vd = haldane_var(c[f"b{ch}1"], c[f"n{ch}1"]) + haldane_var(c[f"b{ch}0"], c[f"n{ch}0"])
            wt = 1 / vd
            g = g + sign * (wt * d).sum(axis=1) / wt.sum(axis=1)
            v = v + 1 / wt.sum(axis=1)
        return g, v

    rA, rH = incr(rng, cfg, D, "ben", cfg.n_ref)
    gr, vr = est(counts_ctx(rA, rH))
    dA, dH = incr(rng, cfg, D, "att", cfg.n_att)
    bA, bH = incr(rng, cfg, D, "ben", cfg.n_ben)
    ca = counts_ctx(dA, dH)
    ga, va = est(ca)
    gb, vb = est(counts_ctx(bA, bH))
    res = shrink_gaps(np.concatenate([ga, gb]), np.concatenate([va, vb]), gr, vr, lc)
    w = weight_from_sum(res.g_tilde, lc)
    g_true = dA - dH
    nA = ca["nA1"].sum(axis=1)
    resid = np.sum(nA * (g_true + np.log(w[:cfg.n_att]))) / np.sum(nA * g_true)
    return dict(removed=1 - resid, ratio=float(np.mean(ga) / np.mean(g_true)), w_ben=w[cfg.n_att:].mean())


# ---------------------------------------------------------------- runners

def _summ(rows, keys):
    """Mean and 95% band; removal is undefined (NaN) when the true agent-only lift is zero."""
    out = {}
    for k in keys:
        x = np.array([r[k] for r in rows], float)
        if np.isnan(x).all():
            out[k] = dict(mean=float("nan"), lo=float("nan"), hi=float("nan"))
            continue
        out[k] = dict(mean=float(np.nanmean(x)), lo=float(np.nanquantile(x, 0.025)), hi=float(np.nanquantile(x, 0.975)))
    return out


SCALAR_KEYS = ("removed", "w_att", "o_att", "w_ben", "o_ben", "B", "err_sd", "sd_formula")


def run(cfg: StudyConfig, seed: int, keep_rows=False):
    rng = np.random.default_rng(seed)
    rows = [replicate(rng, cfg) for _ in range(cfg.reps)]
    out = _summ(rows, SCALAR_KEYS)
    if keep_rows:
        out["_rows"] = rows
    return out


def run_rolling(cfg, seed, roll):
    rng = np.random.default_rng(seed)
    return _summ([replicate_rolling(rng, cfg, roll=roll) for _ in range(cfg.reps)], ("removed", "w_att", "w_ben", "mu"))


def run_contexts(cfg, seed, stratified):
    rng = np.random.default_rng(seed)
    return _summ([replicate_contexts(rng, cfg, stratified) for _ in range(cfg.reps)], ("removed", "ratio", "w_ben"))


def C(base: StudyConfig, **kw):
    lmr_kw = {k[4:]: kw.pop(k) for k in list(kw) if k.startswith("lmr_")}
    cfg = replace(base, **kw)
    if lmr_kw:
        cfg = replace(cfg, lmr=replace(cfg.lmr, **lmr_kw))
    return cfg


def _job(args):
    name, kind, cfg, seed, extra = args
    t0 = time.perf_counter()
    if kind == "run":
        res = run(cfg, seed)
    elif kind == "rolling":
        res = run_rolling(cfg, seed, extra)
    elif kind == "contexts":
        res = run_contexts(cfg, seed, extra)
    elif kind == "tail":
        res = run(cfg, seed, keep_rows=True)
        rows = res.pop("_rows")
        imp = np.concatenate([r["imp"] for r in rows])
        w = np.concatenate([r["w_vec"] for r in rows])
        g = np.concatenate([r["g_vec"] for r in rows])
        nd = np.concatenate([r["nd"] for r in rows])
        res = {}
        for lo, hi in ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, np.inf)):
            m = (imp >= lo) & (imp < hi)
            key = f"{int(lo)}-{hi:g}"
            res[key] = dict(share=float(m.mean()))
            if m.any():
                res[key].update(removed=float(1 - np.sum(nd[m] * (g[m] + np.log(w[m]))) / np.sum(nd[m] * g[m])),
                                mean_weight=float(w[m].mean()))
    else:
        raise ValueError(kind)
    return name, res, time.perf_counter() - t0


def build_jobs(base: StudyConfig, seed=20261013, quick=False):
    """All configurations behind Fig. 5, Fig. 6a, Tab. 4, Tab. 5, §C.3, §C.4, §D."""
    jobs = []
    traffic = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5]
    k = 0

    def add(name, kind, cfg, extra=None):
        nonlocal k
        k += 1
        jobs.append((name, kind, cfg, seed + 7 * k, extra))

    for N in traffic:   # Fig. 5
        add(f"traffic/shared/{int(N)}", "run", C(base, N=N, eA=0.05, eH=0.05))
        add(f"traffic/per_channel/{int(N)}", "run", C(base, N=N))
    for N in (1e4, 5e4):   # Tab. 4
        tag = int(N)
        add(f"ablation/{tag}/lmr", "run", C(base, N=N))
        add(f"ablation/{tag}/shared", "run", C(base, N=N, eA=0.05))
        add(f"ablation/{tag}/noshrink", "run", C(base, N=N, lmr_shrink=False))
        add(f"ablation/{tag}/nohuman", "run", C(base, N=N, lmr_human_term=False))
        add(f"ablation/{tag}/notaufloor", "run", C(base, N=N, lmr_tau_floor=False))
        add(f"ablation/{tag}/nocap", "run", C(base, N=N, lmr_cap=False))
    # Tab. 5 stress tests.
    add("stress/prevalence50/ref", "run", C(base, n_att=150, n_ben=60))
    add("stress/prevalence50/current", "run", C(base, n_att=150, n_ben=60, lmr_prior="current"))
    add("stress/prevalence20/ref", "run", C(base, n_att=60, n_ben=60))
    add("stress/prevalence20/current", "run", C(base, n_att=60, n_ben=60, lmr_prior="current"))
    add("stress/rolling50/fixed", "rolling", C(base, n_att=150, n_ben=60), False)
    add("stress/rolling50/rolling", "rolling", C(base, n_att=150, n_ben=60), True)
    add("stress/twostep/lmr", "run", C(base, steps="twostep"))
    add("stress/twostep/per_revision", "run", C(base, steps="twostep", lmr_chain=False, lmr_anchor_every=0))
    add("stress/small4/lmr", "run", C(base, steps="small-4"))
    add("stress/small4/per_revision", "run", C(base, steps="small-4", lmr_chain=False, lmr_anchor_every=0))
    add("stress/zero10/lmr", "run", C(base, steps="zero-10"))
    add("stress/zero10/plain_chain", "run", C(base, steps="zero-10", lmr_anchor_every=0))
    add("stress/honest4/lmr", "run", C(base, steps="honest-4"))
    add("stress/costmatched_shared", "run", C(base, eA=0.095, eH=0.095))
    # Fig. 6a regime grid and §C.4 share sweep.
    for s in (0.01, 0.05, 0.10, 0.20):
        for N in traffic:
            add(f"regime/{int(round(s * 100))}/{int(N)}", "run", C(base, s=s, N=N))
    for s in (0.01, 0.05, 0.20):   # §C.4 share sweep at the default traffic
        add(f"share/{int(round(s * 100))}", "run", C(base, s=s))
    # §C.4: own traffic (250 impressions), 20x and 200x.
    add("multiplier/1", "run", C(base, N=250))
    add("multiplier/20", "run", C(base, N=5000))
    # §6.6/§D: undetected agents and mislabelled humans; §D duplication.
    for u, r in ((0.1, 0), (0.25, 0), (0.5, 0), (0, 0.01), (0, 0.05)):
        add(f"channel/u{int(u * 100)}_r{int(r * 100)}", "run", C(base, undetected=u, mislabeled=r))
    add("duplication/250x200", "run", C(base, N=250, dup=200))
    add("tail/default", "tail", base)
    # §C.3 heterogeneous contexts.
    for pi in (0.05, 0.20):
        for strat in (False, True):
            add(f"contexts/{int(pi * 100)}/{'strat' if strat else 'pooled'}", "contexts", C(base, pi=pi), strat)
    if quick:
        jobs = [j for j in jobs if j[0].startswith(("traffic/per_channel/50000", "ablation/10000/lmr"))] or jobs[:2]
    return jobs


def timing(R_values=(10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6), seed=0):
    """Fig. 6b: wall-clock time of one estimation round (gap, shrinkage, weights)."""
    base = StudyConfig()
    D = effects(base)
    out = {}
    for R in R_values:
        rng = np.random.default_rng(seed)
        c, _ = draw_counts(rng, np.full(R, D["att_A"]), np.full(R, D["att_H"]), base)
        cr, _ = draw_counts(rng, np.full(R, D["ben_A"]), np.full(R, D["ben_H"]), base)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            g, v = gap_estimate(c)
            gr, vr = gap_estimate(cr)
            res = shrink_gaps(g, v, gr, vr, base.lmr)
            weight_from_sum(res.g_tilde, base.lmr)
            ts.append(time.perf_counter() - t0)
        out[str(R)] = float(np.median(ts))
    return out


def run_all(reps=200, workers=8, seed=20261013, quick=False, scenario="paper"):
    base = StudyConfig(reps=reps)
    if scenario == "gpa_success":
        base = StudyConfig(reps=reps, scenario=scenario, lifts_pp=tuple(sorted(GPA_SUCCESS_LIFTS_PP.items())))
    jobs = build_jobs(base, seed, quick)
    results, secs = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, res, dt in ex.map(_job, jobs):
            results[name] = res
            secs[name] = dt
    results["timing_seconds"] = timing()
    results["_meta"] = dict(reps=reps, seed=seed, scenario=scenario, lifts_pp=dict(base.lifts_pp), logodds=effects(base),
                            base=asdict(base) | {"lmr": asdict(base.lmr)}, job_seconds=secs)
    return results


# Paper-reported values (all measured in the paper) for side-by-side comparison.
PAPER = {
    "Tab4 LMR 1e4 removed": 0.55, "Tab4 LMR 1e4 benign": 0.91,
    "Tab4 LMR 5e4 removed": 0.85, "Tab4 LMR 5e4 benign": 0.91,
    "Tab4 shared 1e4 removed": 0.08, "Tab4 shared 1e4 benign": 0.99,
    "Tab4 shared 5e4 removed": 0.51, "Tab4 shared 5e4 benign": 0.94,
    "Tab4 noshrink 1e4 removed": 1.03, "Tab4 noshrink 1e4 benign": 0.85,
    "Tab4 noshrink 5e4 removed": 0.99, "Tab4 noshrink 5e4 benign": 0.89,
    "Tab4 nohuman 1e4 removed": 0.79, "Tab4 nohuman 1e4 benign": 0.83,
    "Tab4 nohuman 5e4 removed": 1.03, "Tab4 nohuman 5e4 benign": 0.83,
    "Tab5 50% attacked benign weight (LMR)": 0.90, "Tab5 50% attacked benign weight (current prior)": 0.82,
    "Tab5 rolling mu (LMR fixed)": 0.07, "Tab5 rolling mu (rolling)": 0.44,
    "Tab5 twostep removed (LMR)": 0.79, "Tab5 twostep removed (per-revision)": 0.09,
    "Tab5 small4 removed (LMR)": 0.84, "Tab5 small4 removed (per-revision)": 0.20,
    "Tab5 zero10 weight (LMR)": 0.85, "Tab5 zero10 weight (plain chain)": 0.69,
    "Tab5 equal holdout removed (LMR)": 0.85, "Tab5 equal holdout removed (shared 9.5%)": 0.68,
    "Fig5 attack weight 5e4": 0.68, "Fig5 oracle attack weight": 0.58,
    "C4 share 1% removed": 0.49, "C4 share 5% removed": 0.80, "C4 share 10% removed": 0.85,
    "C4 share 20% removed": 0.88,
    "C4 own traffic removed": 0.19, "C4 20x removed": 0.43, "C4 200x removed": 0.85,
    "D undetected 10% removed": 0.74, "D undetected 25% removed": 0.58, "D undetected 50% removed": 0.35,
    "D mislabeled 1% removed": 0.77, "D mislabeled 5% removed": 0.56,
    "C3 pooled 5% removed": 0.82, "C3 strat 5% removed": 0.89,
    "Fig6a s=1% N=1e4 removed": 0.21, "Fig6a s=1% N=1e5 removed": 0.66, "Fig6a s=20% N=1e4 removed": 0.61,
}


def comparison_table(res):
    def g(key, metric="removed"):
        r = res.get(key)
        return None if r is None else r[metric]["mean"]
    ours = {
        "Tab4 LMR 1e4 removed": g("ablation/10000/lmr"), "Tab4 LMR 1e4 benign": g("ablation/10000/lmr", "w_ben"),
        "Tab4 LMR 5e4 removed": g("ablation/50000/lmr"), "Tab4 LMR 5e4 benign": g("ablation/50000/lmr", "w_ben"),
        "Tab4 shared 1e4 removed": g("ablation/10000/shared"), "Tab4 shared 1e4 benign": g("ablation/10000/shared", "w_ben"),
        "Tab4 shared 5e4 removed": g("ablation/50000/shared"), "Tab4 shared 5e4 benign": g("ablation/50000/shared", "w_ben"),
        "Tab4 noshrink 1e4 removed": g("ablation/10000/noshrink"), "Tab4 noshrink 1e4 benign": g("ablation/10000/noshrink", "w_ben"),
        "Tab4 noshrink 5e4 removed": g("ablation/50000/noshrink"), "Tab4 noshrink 5e4 benign": g("ablation/50000/noshrink", "w_ben"),
        "Tab4 nohuman 1e4 removed": g("ablation/10000/nohuman"), "Tab4 nohuman 1e4 benign": g("ablation/10000/nohuman", "w_ben"),
        "Tab4 nohuman 5e4 removed": g("ablation/50000/nohuman"), "Tab4 nohuman 5e4 benign": g("ablation/50000/nohuman", "w_ben"),
        "Tab5 50% attacked benign weight (LMR)": g("stress/prevalence50/ref", "w_ben"),
        "Tab5 50% attacked benign weight (current prior)": g("stress/prevalence50/current", "w_ben"),
        "Tab5 rolling mu (LMR fixed)": g("stress/rolling50/fixed", "mu"),
        "Tab5 rolling mu (rolling)": g("stress/rolling50/rolling", "mu"),
        "Tab5 twostep removed (LMR)": g("stress/twostep/lmr"), "Tab5 twostep removed (per-revision)": g("stress/twostep/per_revision"),
        "Tab5 small4 removed (LMR)": g("stress/small4/lmr"), "Tab5 small4 removed (per-revision)": g("stress/small4/per_revision"),
        "Tab5 zero10 weight (LMR)": g("stress/zero10/lmr", "w_att"), "Tab5 zero10 weight (plain chain)": g("stress/zero10/plain_chain", "w_att"),
        "Tab5 equal holdout removed (LMR)": g("ablation/50000/lmr"), "Tab5 equal holdout removed (shared 9.5%)": g("stress/costmatched_shared"),
        "Fig5 attack weight 5e4": g("ablation/50000/lmr", "w_att"), "Fig5 oracle attack weight": g("ablation/50000/lmr", "o_att"),
        "C4 share 1% removed": g("share/1"), "C4 share 5% removed": g("share/5"),
        "C4 share 10% removed": g("ablation/50000/lmr"), "C4 share 20% removed": g("share/20"),
        "C4 own traffic removed": g("multiplier/1"), "C4 20x removed": g("multiplier/20"), "C4 200x removed": g("ablation/50000/lmr"),
        "D undetected 10% removed": g("channel/u10_r0"), "D undetected 25% removed": g("channel/u25_r0"),
        "D undetected 50% removed": g("channel/u50_r0"), "D mislabeled 1% removed": g("channel/u0_r1"),
        "D mislabeled 5% removed": g("channel/u0_r5"),
        "C3 pooled 5% removed": g("contexts/5/pooled"), "C3 strat 5% removed": g("contexts/5/strat"),
        "Fig6a s=1% N=1e4 removed": g("regime/1/10000"), "Fig6a s=1% N=1e5 removed": g("regime/1/100000"),
        "Fig6a s=20% N=1e4 removed": g("regime/20/10000"),
    }
    rows = []
    for k, p in PAPER.items():
        o = ours.get(k)
        rows.append(dict(metric=k, paper=p, ours=None if o is None else round(float(o), 3)))
    return rows


def save(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
