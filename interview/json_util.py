"""Shared helpers for parsing and retrying LLM JSON output."""

import json
import re


def _strip_fences(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.strip().startswith("```"):
            lines.append(line)
    return "\n".join(lines).strip()


def _snippet(text: str, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _repair(text: str) -> str:
    """Targeted fixes for common LLM JSON slips. Idempotent on valid JSON."""
    fixed = text
    # Smart/curly quotes -> straight quotes.
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    # Drop stray // comment lines.
    fixed = re.sub(r"^\s*//.*$", "", fixed, flags=re.MULTILINE)
    # Trailing commas before } or ].
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    # Missing comma after a closing } or ] when the next key follows.
    fixed = re.sub(r'([}\]])("[^"\n]+"\s*:)', r"\1,\2", fixed)
    # Missing comma between a string value and the next key.
    fixed = re.sub(
        r'"([^"\n]{1,300})"\s*("(?=[^"\n]+"\s*:))',
        r'"\1",\2',
        fixed,
    )
    return fixed


def parse_llm_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response.

    Tries strict parsing first, then applies targeted repairs for common
    LLM output slips. Raises ValueError with a snippet of the offending
    text when the response cannot be salvaged.
    """
    cleaned = _strip_fences(text.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON: {_snippet(cleaned)}")

    window = cleaned[start:end + 1]
    try:
        return json.loads(window)
    except (json.JSONDecodeError, ValueError):
        pass

    repaired = _repair(window)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Model returned unparseable JSON: {_snippet(window)} ({exc})"
        ) from exc


def complete_json(complete_fn, messages, temperature, max_tokens, attempts=3) -> dict:
    """Call complete_fn until it yields parseable JSON (up to attempts times).

    Models are non-deterministic, so a fresh generation often fixes a syntax
    slip or a transient failure. Raises the last error when all attempts fail.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            text, _ = complete_fn(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                print(f"[JSON] complete failed on attempt {attempt}; retrying...")
                print(f"[JSON] error: {_snippet(str(exc))}")
                continue
            break
        try:
            return parse_llm_json(text)
        except ValueError as exc:
            last_error = exc
            if attempt < attempts:
                print(f"[JSON] parse failed on attempt {attempt}; retrying...")
                print(f"[JSON] raw: {_snippet(text)}")
    raise last_error
