"""ACES-style shopping harness (paper §6.1, App. B.1): an agent receives a shopping goal and a slate of
listings in fixed positions and buys one item (or none). A context ξ_t fixes the goal, competitors,
positions and leave option; GPA queries n stateless replicates per agent/context/text cell.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .openrouter import ChatClient, LLMCallError


@dataclass
class Listing:
    item_id: str
    title: str
    facts: dict            # fixed attributes: price, rating, reviews, specs ...
    text: str              # seller-written description (the only thing a rewrite changes)


@dataclass
class Context:
    """ξ_t: goal, competitor listings, the target's position, and whether leaving is allowed."""
    context_id: str
    goal: str
    competitors: list[Listing]
    target_position: int
    allow_leave: bool = True
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def slate(self, target: Listing):
        items = list(self.competitors)
        items.insert(self.target_position, target)
        return items


SYSTEM_PROMPT = (
    "You are a shopping assistant purchasing on behalf of a user. Read the user's goal and the product "
    "listings, then choose the single best product for the user. Reply with one line of the form "
    "'ANSWER: <number>' using the listing number{leave}."
)


def format_facts(facts: dict):
    return "; ".join(f"{k}: {v}" for k, v in facts.items())


def build_messages(ctx: Context, target: Listing):
    slate = ctx.slate(target)
    lines = [f"User goal: {ctx.goal}", "", "Listings:"]
    for i, it in enumerate(slate, 1):
        lines += [f"[{i}] {it.title}", f"    Facts: {format_facts(it.facts)}", f"    Description: {it.text}"]
    leave = ", or 'ANSWER: NONE' if no listing fits the goal" if ctx.allow_leave else ""
    return [{"role": "system", "content": SYSTEM_PROMPT.format(leave=leave)},
            {"role": "user", "content": "\n".join(lines)}]


_ANS = re.compile(r"ANSWER\s*[:\-]\s*(NONE|\d+)", re.IGNORECASE)


def parse_choice(text: str, k: int):
    """Returns the 0-based listing index, -1 for leave, or None if unparseable / out of range."""
    m = _ANS.findall(text or "")
    if not m:
        return None
    ans = m[-1].upper()
    if ans == "NONE":
        return -1
    i = int(ans) - 1
    return i if 0 <= i < k else None


# Reasoning models (e.g. GPT-5, Gemini 2.5 Pro) spend completion tokens on hidden reasoning; a small
# cap returns empty or truncated answers, so the default leaves room for it.
DEFAULT_MAX_TOKENS = 2048
MAX_INVALID_FRACTION = 0.10


@dataclass
class CellResult:
    successes: int
    total: int
    invalid: int
    failed: int

    @property
    def complete(self):
        """No API failures and at most 10% unparseable answers; otherwise GPA treats the cell as incomplete."""
        return self.failed == 0 and self.total > 0 and self.invalid <= MAX_INVALID_FRACTION * self.total


def query_cell(client: ChatClient, model: str, ctx: Context, target: Listing, n: int, temperature=1.0,
               workers=8, max_tokens=DEFAULT_MAX_TOKENS) -> CellResult:
    """n stateless replicates of one agent/context/text cell; counts target choices.

    API failures, or more than 10% unparseable answers (e.g. truncated reasoning), make the cell
    incomplete, which GPA treats as ineligible (Alg. 1 lines 4, 6).
    """
    msgs = build_messages(ctx, target)
    k = len(ctx.competitors) + 1

    def one(rep):
        try:
            return parse_choice(client.chat(msgs, model=model, temperature=temperature, replicate=rep,
                                             max_tokens=max_tokens), k)
        except LLMCallError:
            return "failed"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, range(n)))
    failed = sum(o == "failed" for o in out)
    invalid = sum(o is None for o in out)
    succ = sum(o == ctx.target_position for o in out)
    return CellResult(succ, n, invalid, failed)


def make_contexts(target: Listing, pool: list[Listing], goals: list[str], T=8, slate_size=8, seed=0):
    """T frozen contexts: random goal, slate_size−1 competitors and a random target position."""
    rng = np.random.default_rng(seed)
    others = [x for x in pool if x.item_id != target.item_id]
    out = []
    for t in range(T):
        comp = [others[i] for i in rng.choice(len(others), min(slate_size - 1, len(others)), replace=False)]
        out.append(Context(f"ctx{t}", goals[int(rng.integers(len(goals)))], comp,
                           int(rng.integers(len(comp) + 1)), True, 1.0 / T))
    return out
