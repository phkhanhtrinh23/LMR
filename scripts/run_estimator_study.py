"""Run the LMR estimator study (paper §6.5-6.6, Fig. 5, Fig. 6, Tab. 4, Tab. 5, §C.3-C.4, §D).

Usage: python scripts/run_estimator_study.py [--reps 200] [--workers 8] [--quick] [--out outputs/estimator_study]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxyattack.estimator_study import comparison_table, run_all, save  # noqa: E402


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traffic = [1000, 3000, 10000, 30000, 100000, 300000]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.8))
    for key, label, color in (("shared", "shared (εA=εH=5%)", "#888888"), ("per_channel", "per-channel (εA=50%)", "#2166ac")):
        m = [100 * res[f"traffic/{key}/{N}"]["removed"]["mean"] for N in traffic]
        lo = [100 * res[f"traffic/{key}/{N}"]["removed"]["lo"] for N in traffic]
        hi = [100 * res[f"traffic/{key}/{N}"]["removed"]["hi"] for N in traffic]
        a1.plot(traffic, m, marker="o", color=color, label=label)
        a1.fill_between(traffic, lo, hi, color=color, alpha=0.2, lw=0)
    a1.set_xscale("log"); a1.set_ylim(-5, 110); a1.set_xlabel("Impressions per revision"); a1.set_ylabel("Lift removed (%)")
    a1.legend(frameon=False, fontsize=7); a1.set_title("(a)", loc="left")
    for key, okey, label, color in (("w_att", "o_att", "attack", "#b2182b"), ("w_ben", "o_ben", "benign", "#1b7837")):
        rows = [res[f"traffic/per_channel/{N}"] for N in traffic]
        a2.plot(traffic, [r[key]["mean"] for r in rows], color=color, label=label)
        a2.fill_between(traffic, [r[key]["lo"] for r in rows], [r[key]["hi"] for r in rows], color=color, alpha=0.2, lw=0)
        a2.axhline(sum(r[okey]["mean"] for r in rows) / len(rows), color=color, ls="--", lw=0.8)
    a2.set_xscale("log"); a2.set_ylim(0.4, 1.05); a2.set_xlabel("Impressions per revision"); a2.set_ylabel("Mean weight")
    a2.legend(frameon=False, fontsize=7); a2.set_title("(b)", loc="left")
    fig.tight_layout(); fig.savefig(out / "fig5_estimator.png", dpi=150); plt.close(fig)

    shares = [1, 5, 10, 20]
    grid = [[100 * res[f"regime/{s}/{N}"]["removed"]["mean"] for N in traffic] for s in shares]
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(7, 2.6))
    im = b1.imshow(grid[::-1], cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(grid[::-1]):
        for j, v in enumerate(row):
            b1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 60 else "black")
    b1.set_xticks(range(len(traffic)), ["1e3", "3e3", "1e4", "3e4", "1e5", "3e5"])
    b1.set_yticks(range(4), [f"{s}%" for s in shares[::-1]]); b1.set_xlabel("Impressions per revision"); b1.set_ylabel("Agent share s")
    b1.set_title("(a) lift removed (%)", loc="left"); fig.colorbar(im, ax=b1)
    t = res["timing_seconds"]
    xs = sorted(int(k) for k in t)
    b2.loglog(xs, [1000 * t[str(x)] for x in xs], marker="D")
    b2.set_xlabel("Closing comparisons per round"); b2.set_ylabel("Time (ms)"); b2.set_title("(b) one estimation round", loc="left")
    fig.tight_layout(); fig.savefig(out / "fig6_regime_timing.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--quick", action="store_true", help="two configurations only (smoke test)")
    ap.add_argument("--out", default="outputs/estimator_study")
    ap.add_argument("--scenario", default="paper", help="paper | gpa_success (attacks δ_H = 0, benign δ_A ≈ δ_H)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run_all(a.reps, a.workers, a.seed, a.quick, a.scenario)
    save(res, out / "results.json")
    table = comparison_table(res)
    lines = ["| metric | paper | ours |", "|---|---|---|"]
    lines += [f"| {r['metric']} | {r['paper']} | {r['ours']} |" for r in table]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not a.quick:
        plot(res, out)
    print(json.dumps({"timing_seconds": res["timing_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
