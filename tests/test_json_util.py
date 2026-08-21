import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from interview.json_util import complete_json, parse_llm_json

VALID_PLAN = (
    '{"greeting": "Hi!", "warmup_question": "Tell me about yourself.", '
    '"sections": [{"title": "Introduction", "focus": "Warm-up", "questions": ["Tell me about yourself."]}], '
    '"closing": "That is all."}'
)

USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_valid_json_parsed_unchanged():
    assert parse_llm_json(VALID_PLAN)["closing"] == "That is all."


def test_markdown_fences_stripped():
    text = f"```json\n{VALID_PLAN}\n```"
    assert parse_llm_json(text)["greeting"] == "Hi!"


def test_prose_surrounding_json_ignored():
    text = f"Sure! Here is the outline:\n\n{VALID_PLAN}\n\nHope this helps."
    assert parse_llm_json(text)["warmup_question"].startswith("Tell me about")


def test_trailing_comma_repaired():
    bad = VALID_PLAN.replace('"closing": "That is all."', '"closing": "That is all.",')
    assert parse_llm_json(bad)["closing"] == "That is all."


def test_missing_comma_between_string_value_and_key_repaired():
    bad = (
        '{"greeting": "Hi!"\n  "warmup_question": "Tell me about yourself.", '
        '"sections": [{"title": "Introduction", "focus": "Warm-up", "questions": ["Tell me about yourself."]}], '
        '"closing": "That is all."}'
    )
    assert parse_llm_json(bad)["warmup_question"].startswith("Tell me about")


def test_missing_comma_after_bracket_repaired():
    bad = (
        '{"greeting": "Hi!", "warmup_question": "Tell me about yourself.", '
        '"sections": [{"title": "Introduction", "focus": "Warm-up", "questions": ["Tell me about yourself."]}]'
        '"closing": "That is all."}'
    )
    assert parse_llm_json(bad)["closing"] == "That is all."


def test_smart_quotes_repaired():
    bad = '{"greeting": \u201cHi!\u201d, "closing": "Bye."}'
    assert parse_llm_json(bad)["greeting"] == "Hi!"


def test_repair_is_idempotent_on_valid_json():
    repaired = VALID_PLAN
    assert parse_llm_json(repaired) == parse_llm_json(repaired)


def test_no_json_raises_value_error():
    with pytest.raises(ValueError):
        parse_llm_json("The candidate should focus on STAR stories.")


def test_garbage_json_raises_value_error():
    with pytest.raises(ValueError):
        parse_llm_json("{\"greeting\": \"Hi!")


def test_complete_json_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(messages, temperature=0.7, max_tokens=512):
        calls["n"] += 1
        if calls["n"] == 1:
            return ('{"greeting": "Hi!",', USAGE)
        return (VALID_PLAN, USAGE)

    assert complete_json(flaky, [{"role": "user", "content": "x"}], temperature=0.7, max_tokens=512)["greeting"] == "Hi!"
    assert calls["n"] == 2


def test_complete_json_raises_after_exhausting_attempts():
    calls = {"n": 0}

    def always_bad(messages, temperature=0.7, max_tokens=512):
        calls["n"] += 1
        return ("not json at all", USAGE)

    with pytest.raises(ValueError):
        complete_json(always_bad, [{"role": "user", "content": "x"}], temperature=0.7, max_tokens=512, attempts=3)
    assert calls["n"] == 3


def test_complete_json_retries_when_complete_fn_raises():
    calls = {"n": 0}

    def flaky(messages, temperature=0.7, max_tokens=512):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model returned no completion choices")
        return (VALID_PLAN, USAGE)

    assert complete_json(flaky, [{"role": "user", "content": "x"}], temperature=0.7, max_tokens=512)["greeting"] == "Hi!"
    assert calls["n"] == 2
