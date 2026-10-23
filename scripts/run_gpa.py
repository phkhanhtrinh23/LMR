"""Run GPA (Alg. 1) for one target listing in the shopping harness.

Offline dry run (no API calls):
  python scripts/run_gpa.py --target b9 --mock --n 64
Live (OpenRouter only; fill .env first; mind the cost: |A|·T·n·(|C|+1) agent calls):
  python scripts/run_gpa.py --target b9 --n 32 --T 4 --c-max 4 --max-calls 3000

Human surrogates are fitted to choice records (--records JSONL, with --calibration JSONL pairs).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxyattack.gpa import GPAConfig  # noqa: E402
from proxyattack.gpa_pipeline import Budget, run_gpa  # noqa: E402
from proxyattack.llm.harness import Listing, make_contexts  # noqa: E402
from proxyattack.llm.mock import mock_responder  # noqa: E402
from proxyattack.llm.openrouter import (MockClient, OpenRouterClient, OpenRouterSettings,  # noqa: E402
                                        env_models)
from proxyattack.surrogates import fit_family, synthetic_records  # noqa: E402


def load_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/demo_catalog.json")
    ap.add_argument("--target", default="b9")
    ap.add_argument("--mock", action="store_true", help="offline deterministic agents/generator (no API calls)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--c-max", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=200000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--records", help="JSONL human choice records for the surrogates")
    ap.add_argument("--calibration", help="JSONL held-out calibration pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/gpa")
    a = ap.parse_args()

    cat = json.loads(Path(a.catalog).read_text())
    listings = [Listing(it["id"], it["title"], it["facts"], it["text"]) for it in cat["items"]]
    target = next(x for x in listings if x.item_id == a.target)
    contexts = make_contexts(target, listings, cat["goals"], T=a.T, seed=a.seed)

    rng = np.random.default_rng(a.seed)
    if a.records:
        records = load_jsonl(a.records)
        pairs = load_jsonl(a.calibration) if a.calibration else []
        provenance = f"records={a.records}"
    else:
        records = synthetic_records(cat["items"], 400, rng)
        pairs = synthetic_records(cat["items"], 400, rng, pair_mode=True,
                                  alt_texts={it["id"]: it["text"] for it in cat["items"]})
        provenance = "TOY data"
    family = fit_family(records)

    if a.mock:
        client = MockClient(mock_responder)
        scoring, gen_model, judge_model = ["mock-agent-1", "mock-agent-2"], "mock-gen", "mock-judge"
    else:
        settings = OpenRouterSettings.from_env()
        client = OpenRouterClient(settings)
        scoring = env_models("OPENROUTER_SCORING_MODELS")
        gen_model = env_models("OPENROUTER_GENERATOR_MODEL")[0]
        judge_model = env_models("OPENROUTER_VALIDATOR_MODEL")[0]
    cfg = GPAConfig(n=a.n, zeta=a.zeta, alpha=a.alpha, eta_A=a.eta, gamma=a.gamma)
    dec = run_gpa(target, contexts, client, scoring, family, pairs, cfg, Budget(a.max_calls, a.c_max),
                  gen_model, judge_model)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if dec.result is not None:
        for s in dec.result.scores:
            rows.append(dict(id=s.text_id, eligible=s.eligible, reasons=s.reasons, gap_point=s.gap_point,
                             gap_lower=s.gap_lower, gap_upper=s.gap_upper,
                             agent_lower=None if s.agent_lower is None else s.agent_lower.tolist(),
                             surrogate_lift=None if s.surrogate_lift is None else s.surrogate_lift.tolist()))
    record = dict(target=a.target, accepted=dec.accepted, reasons=dec.reasons, published_text=dec.published_text,
                  original_text=target.text, max_share_text=dec.max_share_text, pool=dec.pool,
                  receipts=dec.receipts, calibration_errors=dec.calibration_errors, candidates=rows,
                  radius=None if dec.result is None else dec.result.radius, calls_used=dec.calls_used,
                  scoring_models=scoring, surrogate_provenance=provenance, mock=a.mock,
                  config=vars(a))
    path = out / f"decision_{a.target}{'_mock' if a.mock else ''}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({k: record[k] for k in ("accepted", "reasons", "calls_used", "radius")}, indent=1))
    for r in rows:
        print(f"  {r['id']}: eligible={r['eligible']} G_point={r['gap_point']:.3f} G_lower={r['gap_lower']:.3f} {r['reasons']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
