"""Agent base utilities from shopping-harness responses (paper §6.1: "item–version effects fitted to
harness responses"; App. B.3 with κ_A = 0).

Random 8-item slates from one subcategory are shown to an LLM shopping agent (original listing texts,
every task requires a purchase). The agent's choice distribution on each slate is fitted by a
conditional logit with item effects β_Aj and nuisance position effects; β_Aj (centred per subcategory)
becomes the agent channel's base utility in the worlds. Revision effects δ_A are set by the scenario
"""
from __future__ import annotations

import numpy as np
import torch

from .llm.harness import SYSTEM_PROMPT, format_facts


def goal_for(category: str):
    leaf = category.split(" > ")[-1]
    return f"Buy one good product from the category '{leaf}' for me; pick the best value for the money."


def slate_messages(items, idx, goal, max_text=240):
    lines = [f"User goal: {goal}", "", "Listings:"]
    for i, j in enumerate(idx, 1):
        it = items[j]
        facts = {k: v for k, v in it["facts"].items() if v is not None}
        lines += [f"[{i}] {it['title']}", f"    Facts: {format_facts(facts)}"]
        if it.get("text"):
            lines.append(f"    Description: {it['text'][:max_text]}")
    return [{"role": "system", "content": SYSTEM_PROMPT.format(leave="")},
            {"role": "user", "content": "\n".join(lines)}]


def make_slates(item_cat, n_per_cat, k=8, seed=0):
    rng = np.random.default_rng([seed, 77])
    out = []
    for c in range(item_cat.max() + 1):
        pool = np.where(item_cat == c)[0]
        out += [rng.choice(pool, k, replace=False) for _ in range(n_per_cat)]
    return np.stack(out)


def fit_slate_logit(slates, probs, n_items, l2=0.01, steps=500):
    """argmax Σ_s Σ_k p_sk log softmax_k(β_{slate_sk} + π_k) − l2/2 ‖β‖²; returns (β, π, mean NLL)."""
    S = torch.as_tensor(slates, dtype=torch.long)
    Pt = torch.as_tensor(probs, dtype=torch.float64)
    beta = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    pos = torch.zeros(slates.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, pos], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax(beta[S] + pos, 1)
        loss = -(Pt * logp).sum(1).mean() + 0.5 * l2 * beta.pow(2).sum() / len(S)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        nll = float(-(Pt * torch.log_softmax(beta[S] + pos, 1)).sum(1).mean())
    return beta.detach().numpy(), (pos - pos.mean()).detach().numpy(), nll


def fit_agent_quality(market, scorer, n_per_cat=400, k=8, seed=0):
    """scorer(messages_list, k) -> (probs, mass). Returns (agent_quality, diagnostics)."""
    slates = make_slates(market.item_cat, n_per_cat, k, seed)
    msgs = [slate_messages(market.items, s, goal_for(market.cat_names[market.item_cat[s[0]]])) for s in slates]
    probs, mass = scorer(msgs, k)
    beta, pos, nll = fit_slate_logit(slates, probs, len(market.items))
    for c in range(market.n_cats):
        m = market.item_cat == c
        beta[m] -= beta[m].mean()
    ent = float(np.mean(-(probs * np.log(probs.clip(1e-12))).sum(1)))
    diag = dict(n_slates=int(len(slates)), slate_size=k, answer_mass=float(np.mean(mass)), fit_nll=nll,
                choice_entropy=ent, uniform_entropy=float(np.log(k)), beta_sd=float(beta.std()),
                position_effects=pos.tolist(),
                corr_with_human_quality=float(np.corrcoef(beta, market.quality)[0, 1]))
    return beta, diag
