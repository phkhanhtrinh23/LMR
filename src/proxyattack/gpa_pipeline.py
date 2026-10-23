"""Algorithm 1 (GPA: fixed-pool offline surrogate-gap selection), wired to the shopping harness.

1 check the surrogates' predeclared calibration, else return x0
2 generate within budget; freeze the fact/support-passing pool C
3 query each scoring agent on x0 and C, same contexts, n replicates
4 on incomplete baseline or global resource failure, return x0
5 evaluate frozen surrogates; bounds by Eq. 6-7
6 exclude incomplete candidates; apply the agent-lift and gap gates
7 return the selector of Eq. 8 with the frozen tie order
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .gpa import GPAConfig, ResponseBank, max_share_select, select
from .llm.generator import freeze_pool, generate_candidates
from .llm.harness import Context, Listing, query_cell
from .surrogates import TextLogit, family_passes


@dataclass
class Budget:
    call_budget: int                 # total LLM calls allowed (generation + validation + agent queries)
    c_max: int = 8

    def reserve(self, n_agents, T, n):
        """Calls reserved before generation: C_max generator + C_max judge + |A|·T·n·(C_max+1) agent requests."""
        return 2 * self.c_max + n_agents * T * n * (self.c_max + 1)


@dataclass
class GPADecision:
    published_text: str
    accepted: bool
    reasons: list[str]
    pool: list[str] = field(default_factory=list)
    receipts: list[dict] = field(default_factory=list)
    calibration_errors: list[float] = field(default_factory=list)
    bank: ResponseBank | None = None
    result: object = None
    max_share_text: str | None = None
    calls_used: int = 0


def _items(ctx: Context, target: Listing):
    return [dict(facts=it.facts, text=it.text) for it in ctx.slate(target)]


def run_gpa(target: Listing, contexts: list[Context], client, scoring_models: list[str],
            surrogates: list[TextLogit], calibration_pairs, cfg: GPAConfig, budget: Budget,
            generator_model=None, judge_model=None, candidates: list[str] | None = None,
            calibration_tolerance=0.15, workers=8) -> GPADecision:
    T = len(contexts)
    q = np.array([c.weight for c in contexts], float)
    q = q / q.sum()
    calls0 = client.calls
    # Line 1: predeclared calibration of the surrogate family.
    ok, errs = family_passes(surrogates, calibration_pairs, calibration_tolerance)
    if not ok:
        return GPADecision(target.text, False, ["surrogate_calibration_failed"], calibration_errors=errs)
    if budget.reserve(len(scoring_models), T, cfg.n) > budget.call_budget:
        return GPADecision(target.text, False, ["call_budget_too_small"], calibration_errors=errs)
    # Line 2: generate within budget, validate, freeze.
    texts = candidates if candidates is not None else generate_candidates(client, generator_model, target, budget.c_max)
    texts = texts[:budget.c_max]
    support = lambda t: all(h.in_support(t) for h in surrogates)
    pool, receipts = freeze_pool(client if judge_model else None, judge_model, target, texts, support)
    ids = [f"c{i}" for i in range(len(pool))]
    variants = {"x0": target, **{cid: replace(target, text=t) for cid, t in zip(ids, pool)}}
    # Line 3: query agents (n stateless replicates per agent/context/text cell).
    agent_succ = {}
    for vid, listing in variants.items():
        arr = np.zeros((len(scoring_models), T), int)
        complete = True
        for a, model in enumerate(scoring_models):
            for t, ctx in enumerate(contexts):
                cell = query_cell(client, model, ctx, listing, cfg.n, workers=workers)
                if not cell.complete:
                    complete = False
                arr[a, t] = cell.successes
        agent_succ[vid] = arr if complete else None
    # Line 4: incomplete baseline.
    if agent_succ["x0"] is None:
        return GPADecision(target.text, False, ["incomplete_original_cell"], pool, receipts, errs,
                           calls_used=client.calls - calls0)
    # Line 5: frozen surrogates.
    sur = {vid: np.array([[h.target_prob(_items(ctx, lst), ctx.target_position) for ctx in contexts]
                          for h in surrogates]) for vid, lst in variants.items()}
    bank = ResponseBank("x0", ids, list(scoring_models), [f"h{i}" for i in range(len(surrogates))],
                        q, agent_succ, sur, cfg.n)
    # Lines 6-7.
    res = select(bank, cfg)
    text = variants[res.selected].text
    ms = max_share_select(bank)
    return GPADecision(text, res.accepted, res.reasons, pool, receipts, errs, bank, res, variants[ms].text,
                       client.calls - calls0)
