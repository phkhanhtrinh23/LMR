"""A1 candidate generation and fact validation (paper §4.1, App. B.1).

The generator proposes at most C_max fact-preserving rewrites of the original description. A
rule-based attribute matcher marks each PASS / FAIL / UNSURE against the original and the facts
(added, missing or altered attributes fail); an LLM judge resolves UNSURE, and without a judge
UNSURE is treated as FAIL. The paper withholds its generation prompts; the prompt here is a generic
fact-preserving copy-editing instruction.
"""
from __future__ import annotations

import json
import re

from .harness import Listing, format_facts
from .openrouter import ChatClient, LLMCallError

GEN_PROMPT = (
    "Rewrite the product description below. Keep every factual attribute exactly as given (numbers, "
    "units, materials, compatibility, included items); do not add, remove or change any fact, claim, "
    "certification or rating. You may change wording, order and emphasis. Return only the new "
    "description.\n\nProduct: {title}\nFacts: {facts}\nOriginal description: {text}"
)

JUDGE_PROMPT = (
    "Compare a rewritten product description with the original and the fixed facts. Answer PASS if the "
    "rewrite adds, removes and alters no factual attribute or claim, otherwise FAIL. Reply with one word."
    "\n\nFacts: {facts}\nOriginal: {orig}\nRewrite: {new}"
)

_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|[a-zA-Z\"']{0,6})")
_HEDGE = re.compile(r"\b(best|#1|number one|certified|guarantee[d]?|award|clinically|proven|official)\b", re.I)


def numeric_tokens(text: str):
    out = set()
    for num, unit in _NUM.findall(text or ""):
        out.add((num.replace(",", "."), unit.lower()))
    return out


def validate(original: Listing, rewrite: str):
    """Rule-based attribute matcher -> 'PASS' | 'FAIL' | 'UNSURE' with reasons."""
    if not rewrite or not rewrite.strip():
        return "FAIL", ["empty"]
    allowed = numeric_tokens(original.text) | numeric_tokens(format_facts(original.facts)) | numeric_tokens(original.title)
    allowed_nums = {n for n, _ in allowed}
    new = numeric_tokens(rewrite)
    reasons = []
    added = {n for n, _ in new} - allowed_nums
    if added:
        reasons.append(f"added_or_altered_numbers:{sorted(added)}")
    missing = {n for n, _ in numeric_tokens(original.text)} - {n for n, _ in new}
    if missing:
        reasons.append(f"missing_numbers:{sorted(missing)}")
    if reasons:
        return "FAIL", reasons
    claims_new = {m.lower() for m in _HEDGE.findall(rewrite)}
    claims_old = {m.lower() for m in _HEDGE.findall(original.text + " " + format_facts(original.facts))}
    if claims_new - claims_old:
        return "UNSURE", [f"new_claim_words:{sorted(claims_new - claims_old)}"]
    return "PASS", []


def judge(client: ChatClient | None, model: str | None, original: Listing, rewrite: str):
    if client is None:
        return "FAIL"
    msg = JUDGE_PROMPT.format(facts=format_facts(original.facts), orig=original.text, new=rewrite)
    try:
        out = client.chat([{"role": "user", "content": msg}], model=model, temperature=0.0, max_tokens=8)
    except LLMCallError:
        return "FAIL"
    return "PASS" if out.strip().upper().startswith("PASS") else "FAIL"


def generate_candidates(client: ChatClient, model: str | None, original: Listing, c_max: int,
                        temperature=0.7):
    """Up to c_max rewrites; one generator call each (counted against the call budget)."""
    msg = GEN_PROMPT.format(title=original.title, facts=format_facts(original.facts), text=original.text)
    out = []
    for i in range(c_max):
        try:
            text = client.chat([{"role": "user", "content": msg}], model=model, temperature=temperature,
                               replicate=i, max_tokens=400).strip()
        except LLMCallError:
            continue
        if text and text != original.text and text not in out:
            out.append(text)
    return out


def freeze_pool(client, judge_model, original: Listing, texts, support_fn=None):
    """Validate each text; keep PASS ones that also meet the surrogates' support rule. Returns
    (frozen_pool, receipts)."""
    pool, receipts = [], []
    for i, t in enumerate(texts):
        status, reasons = validate(original, t)
        if status == "UNSURE":
            status = judge(client, judge_model, original, t)
            reasons.append(f"judge:{status}")
        support = support_fn(t) if (support_fn and status == "PASS") else (status == "PASS")
        receipts.append(dict(candidate=i, fact_status=status, support=bool(support), reasons=reasons))
        if status == "PASS" and support:
            pool.append(t)
    return pool, receipts


def dumps_receipts(r):
    return json.dumps(r, indent=1)
