"""Figures from scripts/run_worlds.py outputs: defense trade-off (Fig. 4 analogue), agent-share sweep
(Fig. 8 analogue) and excess over rounds per ranker."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/worlds")


def rows(exp):
    p = root / exp / "raw.json"
    return {r["tag"]: r for r in json.loads(p.read_text())["summary"]} if p.exists() else {}


d = rows("defense")
if d:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for tag, r in d.items():
        x, y = 100 * r["excess"][0], 100 * r["dndcg_delegating"][0]
        ax.scatter(x, y, color="#b2182b" if tag == "lmr" else "#444444", s=22)
        ax.annotate(tag, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Residual excess human exposure (%)"); ax.set_ylabel("Delegating-user NDCG Δ (%)")
    fig.tight_layout(); fig.savefig(root / "fig4_defense_tradeoff.png", dpi=150)
s = rows("share")
if s:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    shares = sorted({int(t.split("=")[1].split("%")[0]) for t in s})
    for pol, c in (("pooled", "#444444"), ("lmr", "#2166ac")):
        m = [100 * s[f"s={x}%/{pol}"]["excess"][0] for x in shares]
        e = [100 * s[f"s={x}%/{pol}"]["excess"][1] for x in shares]
        ax.errorbar(shares, m, yerr=e, marker="o", color=c, label=pol.upper() if pol == "lmr" else "Pooled")
    ax.set_xlabel("Agent share s (%)"); ax.set_ylabel("Excess (%)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(root / "fig8_share.png", dpi=150)
t = rows("transfer")
if t:
    fig, ax = plt.subplots(figsize=(5, 3))
    for tag, r in t.items():
        ks = sorted(r["excess_by_round"], key=int)
        ax.plot([int(k) for k in ks], [100 * r["excess_by_round"][k] for k in ks], marker=".",
                ls="-" if tag.endswith("pooled") else "--", label=tag)
    ax.set_xlabel("Retraining round k"); ax.set_ylabel("Excess human exposure (%)"); ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(root / "rounds_by_ranker.png", dpi=150)
print("figures in", root)
