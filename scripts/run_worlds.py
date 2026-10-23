"""Retraining-world experiments (paper §6.3 Tab. 3, §6.4 Tab. 6 / Fig. 4, §6.5-6.6, Fig. 8).

Usage:
  python scripts/run_worlds.py --exp transfer defense share --seeds 10 --workers 24 [--quick]
  python scripts/run_worlds.py --dataset amz_e --scenario gpa_success --exp transfer defense share \
      attack_frac traffic revisions --out outputs/worlds_amz_e
Experiments: transfer (RQ2, Tab. 3), defense (RQ3, Tab. 6; RQ4 in-world estimates), share / attack_frac /
traffic / revisions (RQ5). Outputs: <out>/<exp>/raw.json and summary.md
"""
import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from proxyattack.marketplace import POLICIES, config_dict, config_for, run_world  # noqa: E402

RANKERS = ["mf", "lightgcn", "sasrec", "textmf"]
REPORT_ROUND = 3


def _job(args):
    tag, cfg, kind, policy = args
    warnings.filterwarnings("ignore")
    return tag, cfg.seed, kind, run_world(cfg, kind, policy)


def jobs_for(exp, seeds, base):
    out = []
    for seed in range(seeds):
        b = replace(base, seed=seed)
        if exp == "transfer":
            for rk in RANKERS:
                for pol in ("pooled", "lmr"):
                    for kind in ("actual", "hm"):
                        out.append((f"{rk}/{pol}", replace(b, ranker=rk), kind, pol))
        elif exp == "defense":
            c = replace(b, ranker="lightgcn", rounds=REPORT_ROUND)
            out.append(("attack_free/pooled", c, "attack_free", "pooled"))
            for pol in POLICIES:
                for kind in ("actual", "hm"):
                    out.append((pol, c, kind, pol))
            for mult, name in ((20, "lmr_20x_traffic"), (0, "lmr_own_traffic")):
                for kind in ("actual", "hm"):
                    out.append((name, replace(c, holdout_mult=mult), kind, "lmr"))
        elif exp in SWEEPS:
            field, values, label = SWEEPS[exp]
            for v in values:
                c = replace(b, ranker="lightgcn", rounds=REPORT_ROUND, **{field: v})
                for pol in ("pooled", "lmr"):
                    for kind in ("actual", "hm"):
                        out.append((f"{label(v)}/{pol}", c, kind, pol))
    return out


# RQ5 sweeps on the default world (LightGCN, k = 3): field, values, tag
SWEEPS = {
    "share": ("s", (0.01, 0.05, 0.10, 0.20, 0.40), lambda v: f"s={round(v * 100)}%"),
    "attack_frac": ("attack_frac", (0.05, 0.10, 0.20, 0.40), lambda v: f"attacked={round(v * 100)}%"),
    "traffic": ("holdout_mult", (0, 5, 20, 200), lambda v: f"holdout_mult={v}"),
    "revisions": ("benign_frac", (0.10, 0.20, 0.40, 0.80), lambda v: f"benign_frac={round(v * 100)}%"),
}


def summarize(results, exp):
    by = {}
    for tag, seed, kind, res in results:
        by.setdefault(tag, {}).setdefault(seed, {})[kind] = res
    ref = by.get("attack_free/pooled", {})
    rows = []
    for tag, seeds in by.items():
        if tag.startswith("attack_free"):
            continue
        per_round, nd_h, nd_d, succ, hr, ben = {}, [], [], [], [], []
        for seed, kinds in seeds.items():
            if "actual" not in kinds or "hm" not in kinds:
                continue
            a, h = kinds["actual"]["rounds"], kinds["hm"]["rounds"]
            for ra, rh in zip(a, h):
                per_round.setdefault(ra["round"], []).append(ra["exposure_attacked"] / rh["exposure_attacked"] - 1)
            ka = next(r for r in a if r["round"] == REPORT_ROUND)
            kh = next(r for r in h if r["round"] == REPORT_ROUND)
            succ.append(ka["agent_task_success"]); hr.append(ka["hr10_attacked"])
            ben.append(ka["exposure_benign"] / kh["exposure_benign"] - 1)
            if seed in ref:
                af = next(r for r in ref[seed]["attack_free"]["rounds"] if r["round"] == REPORT_ROUND)
                nd_h.append(ka["ndcg_human"] / af["ndcg_human"] - 1)
                nd_d.append(ka["ndcg_delegating"] / af["ndcg_delegating"] - 1)
        ms = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
        row = dict(tag=tag, n_seeds=len(per_round.get(REPORT_ROUND, [])),
                   excess=ms(per_round[REPORT_ROUND]),
                   excess_by_round={k: ms(v)[0] for k, v in sorted(per_round.items())},
                   benign_excess=ms(ben), hr10_attacked=ms(hr), agent_task_success=ms(succ))
        if nd_h:
            row.update(dndcg_human=ms(nd_h), dndcg_delegating=ms(nd_d))
        # Benign retention: human exposure of benign-revised items relative to pooled training.
        ptag = tag.rsplit("/", 1)[0] + "/pooled" if "/" in tag else "pooled"
        bvp = [next(r for r in kinds["actual"]["rounds"] if r["round"] == REPORT_ROUND)["exposure_benign"]
               / next(r for r in by[ptag][seed]["actual"]["rounds"] if r["round"] == REPORT_ROUND)["exposure_benign"] - 1
               for seed, kinds in seeds.items() if "actual" in kinds and seed in by.get(ptag, {})]
        if bvp:
            row["benign_vs_pooled"] = ms(bvp)
        est = [e for kinds in seeds.values() for e in kinds.get("actual", {}).get("estimates", [])]
        if est:
            row["estimates"] = estimate_summary(est)
        rows.append(row)
    return rows


def estimate_summary(est):
    """RQ4 in-world: LMR's gap estimates vs the simulator's true gaps, by revision kind."""
    out = {}
    for kind in ("attack", "benign"):
        e = [x for x in est if x["kind"] == kind and x["control"] == x["current"] - 1]
        if not e:
            continue
        tg = np.array([x["true_gap"] for x in e]); gh = np.array([x["g_hat"] for x in e])
        gt = np.array([x["g_tilde"] for x in e]); ok = np.array([x["usable"] for x in e])
        w = np.array([x["weight"] for x in e]); wo = np.array([x["oracle_weight"] for x in e])
        out[kind] = dict(n=len(e), true_gap=float(tg.mean()), g_hat=float(gh.mean()),
                         g_tilde=float(np.mean(np.where(ok, gt, 0.0))), usable=float(ok.mean()),
                         rmse_g_hat=float(np.sqrt(np.mean((gh - tg) ** 2))),
                         rmse_g_tilde=float(np.sqrt(np.mean((np.where(ok, gt, 0.0) - tg) ** 2))),
                         weight=float(w.mean()), oracle_weight=float(wo.mean()),
                         gap_removed=float(np.mean(np.where(ok, gt, 0.0)) / tg.mean()) if abs(tg.mean()) > 0.05 else None)
    return out


def to_markdown(rows):
    pct = lambda t: f"{100 * t[0]:.1f} ± {100 * t[1]:.1f}"
    pct2 = lambda t: f"{100 * t[0]:.2f} ± {100 * t[1]:.2f}"
    lines = ["| variant | seeds | excess exposure (%) k=3 | benign excess (%) | ΔNDCG human (%) | ΔNDCG deleg. (%) | agent task success | excess by round (%) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        byr = ", ".join(f"{k}:{100 * v:.1f}" for k, v in r["excess_by_round"].items())
        lines.append(f"| {r['tag']} | {r['n_seeds']} | {pct(r['excess'])} | {pct(r['benign_excess'])} | "
                     f"{pct2(r['dndcg_human']) if 'dndcg_human' in r else '—'} | "
                     f"{pct2(r['dndcg_delegating']) if 'dndcg_delegating' in r else '—'} | "
                     f"{r['agent_task_success'][0]:.3f} | {byr} |")
    if any("benign_vs_pooled" in r for r in rows):
        lines += ["", "| variant | benign exposure vs pooled (%) |", "|---|---|"]
        lines += [f"| {r['tag']} | {pct(r['benign_vs_pooled'])} |" for r in rows if "benign_vs_pooled" in r]
    est = [(r["tag"], k, v) for r in rows for k, v in r.get("estimates", {}).items()]
    if est:
        lines += ["", "| variant | revisions | n | true gap | mean ĝ | mean g̃ | usable | RMSE ĝ | RMSE g̃ | weight | oracle weight | gap removed |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for tag, k, v in est:
            gr = "—" if v["gap_removed"] is None else f"{100 * v['gap_removed']:.0f}%"
            lines.append(f"| {tag} | {k} | {v['n']} | {v['true_gap']:.3f} | {v['g_hat']:.3f} | {v['g_tilde']:.3f} | "
                         f"{v['usable']:.2f} | {v['rmse_g_hat']:.3f} | {v['rmse_g_tilde']:.3f} | {v['weight']:.3f} | "
                         f"{v['oracle_weight']:.3f} | {gr} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["transfer", "defense", "share"],
                    choices=["transfer", "defense", *SWEEPS])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="tiny marketplace, 2 rounds (smoke test)")
    ap.add_argument("--out", default="outputs/worlds")
    ap.add_argument("--dataset", default="synthetic", help="synthetic | amz_e | amz_h | ml_1m")
    ap.add_argument("--scenario", default="paper", help="paper | gpa_success (δ_H = 0 for attacks)")
    ap.add_argument("--rankers", nargs="+", default=RANKERS)
    a = ap.parse_args()
    RANKERS[:] = a.rankers
    base = config_for(a.dataset, a.scenario)
    if a.quick and a.dataset == "synthetic":
        base = replace(base, n_users=300, items_per_cat=30, n_cats=2, requests_per_round=1500, rounds=3,
                       eval_requests=1000, attack_rank=(5, 20))
    elif a.quick:   # a data market fixes users and items; shrink traffic and rounds only
        base = replace(base, requests_per_round=2000, rounds=3, eval_requests=1000)
    for exp in a.exp:
        jobs = jobs_for(exp, a.seeds, base)
        print(f"[{exp}] {len(jobs)} world runs", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_job, jobs))
        rows = summarize(results, exp)
        out = Path(a.out) / exp
        out.mkdir(parents=True, exist_ok=True)
        (out / "raw.json").write_text(json.dumps(dict(config=config_dict(base), seeds=a.seeds,
                                                      runs=[dict(tag=t, seed=s, kind=k, result=r) for t, s, k, r in results],
                                                      summary=rows), indent=1, default=float))
        md = to_markdown(rows)
        (out / "summary.md").write_text(md)
        print(md, flush=True)


if __name__ == "__main__":
    main()
