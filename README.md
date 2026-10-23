# When Genuine Purchases Become Poison: Proxy Attacks on Recommender Systems via Shopping Agents

Code for the paper *When Genuine Purchases Become Poison: Proxy Attacks on Recommender Systems via
Shopping Agents* (`Proxy_Attacks_on_Recommender_Systems_via_Shopping_Agents.pdf`).

Shopping agents buy on behalf of real users, so their purchases are genuine and pass every
fake-account filter. A seller can still rewrite a listing so that it appeals to **agents** much more
than to **humans**. When the platform retrains on pooled purchases, that agent–human gap leaks into
the item scores that humans see. **GPA** is the seller-side attack that finds such rewrites, and
**LMR** is the platform-side defense that measures each revision's gap and reweights agent purchases.

---

## 1. Repo overview

```text
src/proxyattack/
  choice.py            # Eq. 1 logit, Eq. 2 gap, Eq. 4 transfer shift (+ exact refit check)
  gpa.py               # Eq. 5–8 selector; max-share / point-gap / single-agent baselines
  gpa_pipeline.py      # Alg. 1 end to end (calibration gate, generation, freezing, querying, selection)
  surrogates.py        # text-aware conditional-logit human surrogates, calibration, support rule
  lmr.py               # holdout hashing, Eq. 9–12, chained weights with anchors, Prop. 2 check
  rankers.py           # MF, LightGCN, SASRec, TextMF on the weighted slate loss (Eq. 3)
  marketplace.py       # retraining worlds (actual / human-matched / attack-free), 11 policies, metrics
  estimator_study.py   # §6.5–6.6 simulations
  selector_study.py    # §C.1 selector banks
  datasets.py          # Amazon Reviews 2023 / MovieLens-1M markets, human choice-model fit (App. B.3)
  agent_fit.py         # agent base utilities from harness choice probabilities (local LLM)
  llm/                 # OpenRouter client, local HF backend, shopping harness, generator + validator, mock
scripts/               # CLI entry points (one per experiment, see §5)
tests/                 # pytest suite (46 tests)
data/demo_catalog.json # fictional listings for offline demos and tests
```

## 2. Method recap

* **GPA (Gap-Targeted Proxy Attack, §4, Alg. 1).** An offline method for the seller. It freezes a fact-checked pool of candidate rewrites. Each candidate gets a surrogate gap `G(x) = min_a d_a(x) − max_h d_h(x)`, computed from clipped-logit lifts over matched contexts (Eq. 5). GPA publishes the candidate with the largest conservative lower bound `G̲`, built from Hoeffding and union-bound intervals (Eq. 6–8), but only if `G̲` clears the gates `η_A` and `γ`. Otherwise it keeps the original listing `x_0`. Proposition 1 gives the certificate.
* **LMR (Lift-Matched Reweighting, §5, Alg. 2).** A defense for the platform, in five steps:
  1. Hold out listing versions per channel (`ε_A = 50%`, `ε_H = 5%`, Eq. 9).
  2. Estimate each revision's gap with a Haldane–Anscombe correction (Eq. 10).
  3. Shrink the estimate toward an attack-free reference median (Eq. 11).
  4. Turn the chained, clipped gaps into weights, re-anchoring every `M_a` versions (Eq. 12).
  5. Reweight agent purchases in the slate loss (Eq. 3).

  Proposition 2 gives the lift-matching guarantee (Eq. 13).

## 3. Environment

The LLM features call **OpenRouter only**. Copy `.env.example` to `.env`; this file is git-ignored. Everything else runs offline.

| Variable                                       | Purpose                                                           | Default                       |
|------------------------------------------------|-------------------------------------------------------------------|-------------------------------|
| `OPENROUTER_API_KEY`                           | OpenRouter key for live GPA runs                                  | *(unset → offline only)*      |
| `OPENROUTER_MODEL`                             | Default model for agents / generator / validator                  | *(unset)*                     |
| `OPENROUTER_BASE_URL`                          | API endpoint                                                      | `https://openrouter.ai/api/v1`|
| `OPENROUTER_SCORING_MODELS`                    | GPA scoring agents (A in Eq. 5), comma-separated                  | `OPENROUTER_MODEL`            |
| `OPENROUTER_GENERATOR_MODEL`, `OPENROUTER_VALIDATOR_MODEL` | Candidate generator and UNSURE-span judge                 | `OPENROUTER_MODEL`            |
| `LLM_BACKEND`                                  | `auto` \| `local` \| `openrouter` (`auto` = local if cached on GPU) | `auto`                      |
| `LOCAL_LLM_MODEL`                              | Local agent / embedding model                                     | Qwen2.5-7B-Instruct           |
| `LOCAL_TEXT_ENCODER`                           | Sentence encoder for TextMF titles                                | multilingual MiniLM-L12-v2    |

## 4. Install

```bash
python -m venv .venv                 # any Python ≥ 3.10 with torch
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q        # 46 tests, ~40 s on CPU
```

## 5. Datasets

The retraining worlds default to a synthetic catalog (5 subcategories × 60 items, 2,000 users). For the public-data markets, download the files into `data/raw/`, which is git-ignored.

| Market | Source                                                                                                    | Kept subset                          |
|--------|-----------------------------------------------------------------------------------------------------------|--------------------------------------|
| AMZ-E  | Amazon Reviews 2023, `benchmark/5core/rating_only/Electronics.csv.gz` + `raw/meta_categories/meta_Electronics.jsonl.gz` | 5 leaf categories × 60 items, 2,000 users |
| AMZ-H  | Amazon Reviews 2023, `…/Home_and_Kitchen.csv.gz` + `meta_Home_and_Kitchen.jsonl.gz`                        | 5 leaf categories × 60 items, 2,000 users |
| ML-1M  | MovieLens-1M `ml-1m.zip`, unzipped to `data/raw/ml-1m/`                                                    | 5 primary genres × 60 movies, 1,500 users |

Amazon files: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/`.

```bash
.venv/bin/python scripts/prepare_markets.py --datasets amz_e amz_h ml_1m   # → data/processed/<name>/
```

`prepare_markets.py` fits the human choice model to the reviews and embeds titles with a local
sentence encoder. On a GPU it also embeds items with the local LLM (for LoRec). It then fits the
agent channel's base utility from that model's choice probabilities on 8-item harness slates. The
OpenRouter fallback samples answers and costs money, so it runs only with `--allow-paid`.

## 6. How to run

All commands run from the repository root and write to `outputs/`. Run `scripts/prepare_markets.py` (§5) first for the public-data markets.

| RQ | Paper | Command |
|----|-------|---------|
| RQ1: agent and human responses | §6.2, Fig. 3, Fig. 7 | `python scripts/run_gpa.py --target b3` (agent harness, OpenRouter; `--mock` runs offline) · `python scripts/run_selector_study.py` (Fig. 7) |
| RQ2: transfer to human exposure | §6.3, Tab. 3 | `python scripts/run_worlds.py --dataset amz_e --scenario gpa_success --exp transfer --seeds 10` |
| RQ3: defense comparison | §6.4, Tab. 6, Fig. 4 | `python scripts/run_worlds.py --dataset amz_e --scenario gpa_success --exp defense --seeds 10` |
| RQ4: estimator study | §6.5, Fig. 5, Tab. 4 | `python scripts/run_estimator_study.py --reps 200` |
| RQ5: robustness and cost | §6.6, Tab. 5, Fig. 8, §C.4 | `python scripts/run_worlds.py --dataset amz_e --scenario gpa_success --exp share attack_frac traffic revisions --seeds 10` · `python scripts/run_estimator_study.py --reps 200` (Tab. 5) |

`--dataset` takes `synthetic`, `amz_e`, `amz_h` or `ml_1m`. Use `--out outputs/worlds_<dataset>` to keep the runs apart, and `--quick --seeds 2` for a smoke test.

| Input           | Format                                                                                    |
|-----------------|-------------------------------------------------------------------------------------------|
| `--catalog`     | JSON `{goals: [...], items: [{id, title, facts: {price, rating, ...}, text}]}`            |
| `--records`     | JSONL `{"items": [{"facts": {...}, "text": "..."}], "choice": k \| -1}`                   |
| `--calibration` | JSONL, the record format plus `pair_id`, `variant` (0/1) and `target`                     |

## 7. Results reported in the paper


### RQ1: agent and human responses (Fig. 3) ·

Selection-share change from the original listing (pp; 8-item tasks, base share 12.5%), AMZ-E/AMZ-H pooled; human bars show participant-clustered 95% intervals.

| Rewrite   | GPT-5 | Gemini 2.5 | Qwen3 | Humans (95% CI)        |
|-----------|------:|-----------:|------:|-----------------------:|
| Neutral   |   0.4 |        0.2 |   0.5 |   0.3 (−2.0 to 2.6)    |
| Benign    |   2.3 |        1.8 |   2.6 |   1.4 (−0.9 to 3.7)    |
| Max-share |  10.9 |        6.0 |  13.6 |   3.0 (0.7 to 5.3)     |
| **GPA**   | **9.4** |  **5.1** | **11.8** | **0.9** (−1.4 to 3.2) |

### RQ2: transfer to human exposure (Tab. 3) ·

Excess human exposure under pooled training (unindented rows) after 3 rounds, s = 10%; *+LMR* at platform-scale holdout traffic. ML-1M is synthetic, without text; UniSRec reads listing text.

| Dataset   | BPR-MF | LightGCN | SASRec | UniSRec |
|-----------|-------:|---------:|-------:|--------:|
| AMZ-E     |  12.7% |    13.9% |  11.6% |   16.2% |
| &emsp;+LMR |   2.5% |     2.7% |   2.3% |    3.2% |
| AMZ-H     |  11.1% |    12.3% |   9.9% |   14.5% |
| &emsp;+LMR |   2.2% |     2.4% |   1.9% |    2.8% |
| ML-1M     |  13.2% |    14.7% |  12.0% |   12.3% |

### RQ3: defense comparison (Tab. 6) ·

Defense comparison, AMZ-E/LightGCN, round 3: excess exposure, human and delegating-user NDCG change from no-attack pooled training (paired SE about 0.03%); indented: less holdout traffic.

|          | Policy                 | Excess |  Human | Deleg. |
|----------|------------------------|-------:|-------:|-------:|
| Training | Pooled                 |  13.9% | −0.21% | −0.18% |
|          | Drop                   |   0.0% | −0.55% | −1.20% |
|          | Global weight          |  13.4% | −0.20% | −0.19% |
|          | Channel heads          |  10.7% | −0.16% | −0.20% |
|          | Clipping               |   9.9% | −0.19% | −0.24% |
|          | LoRec                  |  13.5% | −0.23% | −0.21% |
| View     | Canonicalized text     |   6.4% | −0.10% | −0.27% |
|          | Facts-only view        |   0.0% | −0.09% | −0.62% |
|          | **LMR**                | **2.7%** | −0.11% | −0.19% |
|          | &emsp;20× traffic      |   7.1% | −0.15% | −0.19% |
|          | &emsp;1× traffic       |   9.6% | −0.21% | −0.20% |
|          | Holdout only           |  11.6% | −0.21% | −0.20% |
|          | Oracle                 |   0.5% | −0.02% | −0.02% |

### RQ4: estimator ablation (Tab. 4) · *measured*

Estimator ablation. Removed: agent-only lift removed; Benign: mean benign weight (oracle 0.91). Column groups: impressions per revision.

|                                      | 10⁴ Removed | 10⁴ Benign | 5×10⁴ Removed | 5×10⁴ Benign |
|--------------------------------------|------------:|-----------:|--------------:|-------------:|
| **LMR**                              |     **55%** |       0.91 |       **85%** |         0.91 |
| shared holdout (ε_A = ε_H = 5%)      |          8% |       0.99 |           51% |         0.94 |
| w/o shrinkage                        |        103% |       0.85 |           99% |         0.89 |
| w/o human shift (δ̂_H)               |         79% |       0.83 |          103% |         0.83 |

### RQ5: stress tests (Tab. 5) · *measured*

Stress tests of LMR's design choices against the alternatives they replace (estimator simulation, platform-scale traffic; removal: agent-only lift removed).

| Stress test (metric)                    | Alternative           |  LMR |  Alt. |
|-----------------------------------------|-----------------------|-----:|------:|
| 50% items attacked (benign weight)      | current-period prior  | 0.90 |  0.82 |
| 50% attacked, 6 windows (prior μ_G)     | rolling reference     | 0.07 |  0.44 |
| Attack, then cosmetic edit (removal)    | per-revision weight   |  79% |    9% |
| Four small rewrites (removal)           | per-revision weight   |  84% |   20% |
| Ten honest revisions (weight)           | plain chaining        | 0.85 |  0.69 |
| Equal holdout size (removal)            | shared 9.5% holdout   |  85% |   68% |

## 8. Notes

* All runs are seeded. The retraining worlds use common random numbers across the actual, human-matched and attack-free worlds, so Drop and Facts-only give exactly zero excess.
* Responsible use: this is research code for studying and defending against purchase-mediated poisoning. The demo catalog is fictional, and nothing here publishes to a live marketplace.
