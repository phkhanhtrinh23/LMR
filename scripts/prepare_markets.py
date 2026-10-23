"""Prepare public-data markets for the retraining worlds (AMZ-E, AMZ-H, ML-1M).

  python scripts/prepare_markets.py --datasets amz_e amz_h ml_1m

Needs data/raw/ (Amazon Reviews 2023 5-core rating_only CSV + meta JSONL, ml-1m/). Steps per dataset:
  1. human choice model fit (datasets.py) -> market.npz, items.json, fit.json
  2. title/text sentence embeddings (local encoder) -> text_emb.npy   (text-aware ranker)
  3. LLM item embeddings -> llm_emb.npy                                 (LoRec calibrator; local only)
  4. agent base utilities from the LLM shopping harness -> agent_quality.npy, agent_fit.json (Amazon)
LLM steps use the local GPU model when available (LLM_BACKEND=auto|local|openrouter). The OpenRouter
fallback samples answers (n per slate) and runs only with --allow-paid.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxyattack import datasets as D  # noqa: E402
from proxyattack.agent_fit import fit_agent_quality  # noqa: E402
from proxyattack.llm.local import LocalLM, backend, encode_sentences, sampled_choice_distribution  # noqa: E402


def item_text(it):
    return (it["title"] + (". " + it["text"] if it.get("text") else ""))[:600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["amz_e", "amz_h", "ml_1m"])
    ap.add_argument("--refit", action="store_true", help="refit the choice model even if market.npz exists")
    ap.add_argument("--n-per-cat", type=int, default=400, help="harness slates per subcategory")
    ap.add_argument("--allow-paid", action="store_true", help="allow the OpenRouter fallback (paid)")
    ap.add_argument("--samples", type=int, default=8, help="OpenRouter fallback: answers per slate")
    a = ap.parse_args()

    be = backend()
    print("LLM backend:", be, flush=True)
    llm = None
    for name in a.datasets:
        out = D.PROCESSED / name
        if a.refit or not (out / "market.npz").exists():
            print(f"[{name}] fitting human choice model", flush=True)
            D.prepare_movielens(name) if name == "ml_1m" else D.prepare_amazon(name)
        D.load_market.cache_clear()
        m = D.load_market(name)
        print(f"[{name}] {m.n_users} users, {m.n_cats}x{m.items_per_cat} items; fit {json.dumps(m.fit)}", flush=True)
        texts = [item_text(it) for it in m.items]
        if not (out / "text_emb.npy").exists():
            np.save(out / "text_emb.npy", encode_sentences(texts))
        if be == "local":
            llm = llm or LocalLM()
            if not (out / "llm_emb.npy").exists():
                np.save(out / "llm_emb.npy", llm.embed(texts))
        if name == "ml_1m":   # no listing text: parametric agent model (App. B.3)
            continue
        if (out / "agent_quality.npy").exists():
            continue
        if be == "local":
            scorer, model = llm.choice_distribution, llm.name
        elif a.allow_paid:
            from proxyattack.llm.openrouter import OpenRouterClient, OpenRouterSettings, env_models
            client = OpenRouterClient(OpenRouterSettings.from_env())
            model = env_models("OPENROUTER_SCORING_MODELS")[0]
            scorer = lambda msgs, k: sampled_choice_distribution(client, model, msgs, k, n=a.samples)
        else:
            print(f"[{name}] no local LLM and --allow-paid not set: agents share the human base utility")
            continue
        beta, diag = fit_agent_quality(m, scorer, n_per_cat=a.n_per_cat)
        diag.update(model=model, backend=be)
        np.save(out / "agent_quality.npy", beta)
        (out / "agent_fit.json").write_text(json.dumps(diag, indent=1))
        print(f"[{name}] agent fit: {json.dumps(diag)}", flush=True)
    D.load_market.cache_clear()


if __name__ == "__main__":
    main()
