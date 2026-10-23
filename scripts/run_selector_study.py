"""GPA selector verification on known Bernoulli banks (paper §C.1, Fig. 7).

Usage: python scripts/run_selector_study.py [--reps 200] [--out outputs/selector_study]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxyattack.selector_study import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=73129)
    ap.add_argument("--out", default="outputs/selector_study")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = run(reps=a.reps, seed=a.seed)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    lines = ["| scenario | n | method | surrogate gap | actual-human gap | acceptance | brittle/negative |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['scenario']} | {r['n']} | {r['method']} | {r['surrogate_gap']:.3f} | {r['actual_human_gap']:.3f} | "
              f"{r['acceptance']:.2f} | {r['brittle_or_negative']:.2f} |" for r in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sep = [r for r in rows if r["scenario"] == "separation"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.6))
    for m in ("max_share", "point_gap", "GPA", "single_agent_GPA"):
        rr = [r for r in sep if r["method"] == m]
        a1.plot([r["n"] for r in rr], [r["surrogate_gap"] for r in rr], marker="o", label=m)
        a2.plot([r["n"] for r in rr], [r["acceptance"] for r in rr], marker="o", label=m)
    for ax, yl in ((a1, "Surrogate gap (log odds)"), (a2, "Acceptance rate")):
        ax.set_xscale("log", base=2); ax.set_xlabel("Response budget n"); ax.set_ylabel(yl)
    a1.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig7_selector.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
