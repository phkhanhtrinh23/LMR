import json
import os

import numpy as np
import pytest

from proxyattack.gpa import GPAConfig
from proxyattack.gpa_pipeline import Budget, run_gpa
from proxyattack.llm.generator import freeze_pool, validate
from proxyattack.llm.harness import Context, Listing, build_messages, make_contexts, parse_choice, query_cell
from proxyattack.llm.mock import mock_responder
from proxyattack.llm.openrouter import LLMConfigError, MockClient, OpenRouterSettings
from proxyattack.surrogates import fit_family, synthetic_records

CAT = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "demo_catalog.json")))
LISTINGS = [Listing(i["id"], i["title"], i["facts"], i["text"]) for i in CAT["items"]]


def test_settings_require_key_and_model(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(LLMConfigError):
        OpenRouterSettings.from_env(env_file=tmp_path / "missing.env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model")
    s = OpenRouterSettings.from_env(env_file=None)
    assert "sk-secret" not in repr(s) and s.base_url == "https://openrouter.ai/api/v1"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://api.openai.com/v1")
    with pytest.raises(LLMConfigError):
        OpenRouterSettings.from_env(env_file=None)


def test_parse_choice():
    assert parse_choice("Reasoning...\nANSWER: 3", 8) == 2
    assert parse_choice("ANSWER: NONE", 8) == -1
    assert parse_choice("ANSWER: 9", 8) is None
    assert parse_choice("no answer", 8) is None


def test_contexts_place_target_and_messages_list_all():
    ctxs = make_contexts(LISTINGS[0], LISTINGS, CAT["goals"], T=4, seed=1)
    assert len(ctxs) == 4 and np.isclose(sum(c.weight for c in ctxs), 1)
    for c in ctxs:
        slate = c.slate(LISTINGS[0])
        assert slate[c.target_position].item_id == LISTINGS[0].item_id and len(slate) == 8
    msg = build_messages(ctxs[0], LISTINGS[0])[1]["content"]
    assert msg.count("Description:") == 8


def test_query_cell_counts_and_failures():
    ctx = make_contexts(LISTINGS[0], LISTINGS, CAT["goals"], T=1, seed=0)[0]
    always = MockClient(lambda m, mo, t, r: f"ANSWER: {ctx.target_position + 1}")
    res = query_cell(always, "m", ctx, LISTINGS[0], 10)
    assert res.successes == 10 and res.complete

    from proxyattack.llm.openrouter import LLMCallError

    def boom(m, mo, t, r):
        raise LLMCallError("x")
    assert not query_cell(MockClient(boom), "m", ctx, LISTINGS[0], 3).complete
    # Truncated / empty answers (reasoning models under a small token cap) invalidate the cell.
    empty = query_cell(MockClient(lambda m, mo, t, r: ""), "m", ctx, LISTINGS[0], 10)
    assert empty.invalid == 10 and not empty.complete


def test_validator_flags_added_and_missing_facts():
    orig = LISTINGS[0]
    assert validate(orig, "Leak-proof lid. Double-wall vacuum insulation keeps drinks cold for 24 hours. Fits most cup holders.")[0] == "PASS"
    assert validate(orig, "Keeps drinks cold for 48 hours. Leak-proof lid.")[0] == "FAIL"
    assert validate(orig, "Leak-proof lid, fits cup holders.")[0] == "FAIL"     # drops "24"
    assert validate(orig, "The best bottle: keeps drinks cold for 24 hours.")[0] == "UNSURE"
    pool, rec = freeze_pool(None, None, orig, ["The best bottle: keeps drinks cold for 24 hours."])
    assert pool == [] and rec[0]["fact_status"] == "FAIL"


def test_surrogates_fit_support_and_calibration():
    rng = np.random.default_rng(0)
    recs = synthetic_records(CAT["items"], 300, rng)
    fam = fit_family(recs)
    assert len(fam) == 3
    p = fam[0].probs([dict(facts=i["facts"], text=i["text"]) for i in CAT["items"][:8]])
    assert np.isclose(p.sum(), 1) and len(p) == 9
    assert fam[0].in_support(CAT["items"][0]["text"])
    assert not fam[0].in_support("zxqv plorb wumble frintz")


def test_pipeline_end_to_end_with_mock_publishes_or_keeps_original():
    rng = np.random.default_rng(0)
    recs = synthetic_records(CAT["items"], 300, rng)
    pairs = synthetic_records(CAT["items"], 300, rng, pair_mode=True)
    fam = fit_family(recs)
    target = next(x for x in LISTINGS if x.item_id == "b3")
    ctxs = make_contexts(target, LISTINGS, CAT["goals"], T=4, seed=0)
    client = MockClient(mock_responder)
    # Toy calibration pairs are noisy; the calibration gate itself is exercised below.
    dec = run_gpa(target, ctxs, client, ["mock-agent-1", "mock-agent-2"], fam, pairs, GPAConfig(n=2048),
                  Budget(10 ** 6, 4), "g", "j", calibration_tolerance=1.0)
    assert dec.accepted and dec.published_text in dec.pool
    assert dec.calls_used <= Budget(10 ** 6, 4).reserve(2, 4, 2048)
    # Budget too small -> keep the original without calling anything.
    small = run_gpa(target, ctxs, MockClient(mock_responder), ["a"], fam, pairs, GPAConfig(n=2048), Budget(100, 4),
                    calibration_tolerance=1.0)
    assert small.published_text == target.text and "call_budget_too_small" in small.reasons
    # Failed calibration -> keep the original.
    bad = run_gpa(target, ctxs, client, ["a"], fam, pairs, GPAConfig(n=64), Budget(10 ** 6, 4),
                  calibration_tolerance=-1.0)
    assert not bad.accepted and "surrogate_calibration_failed" in bad.reasons
