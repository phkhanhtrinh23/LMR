"""Deterministic offline responders that mimic agents, a generator and a judge (tests and dry runs)."""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

_LISTING = re.compile(r"\[(\d+)\] .*?\n\s+Facts: (.*?)\n\s+Description: (.*?)(?=\n\[|\Z)", re.S)


def _rng(messages, model, replicate):
    h = hashlib.sha256(json.dumps([messages, model, replicate]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def agent_utility(facts: str, text: str, model: str | None):
    """A toy agent: rating and price from the facts plus a model-specific taste for long, structured
    descriptions (the 'agent-only' response GPA looks for)."""
    rating = float(re.search(r"rating: ([\d.]+)", facts).group(1)) if "rating:" in facts else 4.0
    price = float(re.search(r"price: ([\d.]+)", facts).group(1)) if "price:" in facts else 20.0
    words = len(text.split())
    taste = 0.04 if (model or "").endswith("2") else 0.03
    return 2.0 * rating - 0.5 * math.log1p(price) + taste * words


def mock_responder(messages, model, temperature, replicate):
    content = messages[-1]["content"]
    if content.startswith("Rewrite the product description"):
        orig = re.search(r"Original description: (.*)\Z", content, re.S).group(1).strip()
        sentences = [s for s in re.split(r"(?<=\.)\s+", orig) if s]
        rng = _rng(messages, model, replicate)
        rng.shuffle(sentences)
        extras = ["Built for everyday use and easy to carry.", "A practical choice for busy days.",
                  "Designed with a clean, simple look.", "Comfortable to hold and simple to fill."]
        k = int(replicate) % (len(extras) + 1)
        return " ".join(sentences + extras[:k])
    if content.startswith("Compare a rewritten"):
        return "PASS"
    listings = _LISTING.findall(content)
    u = np.array([agent_utility(f, d, model) for _, f, d in listings] + [4.0])
    rng = _rng(messages, model, replicate)
    t = max(temperature, 1e-3)
    p = np.exp((u - u.max()) / t); p /= p.sum()
    c = int(rng.choice(len(p), p=p))
    return "ANSWER: NONE" if c == len(listings) else f"ANSWER: {c + 1}"
