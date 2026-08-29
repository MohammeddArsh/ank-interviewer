"""Post-interview evaluation (score + written feedback)."""

from brain.llm import complete
from interview import prompts
from interview.json_util import complete_json


def evaluate(job_description: str, resume_text: str, transcript: list) -> dict:
    """Score the interview. Returns {score, strengths, improvements, verdict}."""
    transcript_text = "\n".join(
        f"{'Interviewer' if t['role'] == 'interviewer' else 'Candidate'}: {t['text']}"
        for t in transcript
    ) or "(no answers given)"

    prompt = prompts.evaluation_prompt(job_description, resume_text, transcript_text)
    data = None
    for _attempt in range(2):
        messages = [
            {"role": "system", "content": "You are a precise assistant. You only respond with valid JSON."},
            {"role": "user", "content": prompt},
        ]
        data = complete_json(complete, messages, temperature=0.5, max_tokens=1000)
        score = _parse_score(data)
        # Free models sometimes return a harsh score with no improvement items — an
        # internally inconsistent, unusable evaluation. Retry once with a nudge.
        if score >= 50 or (data or {}).get("improvements"):
            break
        prompt += (
            "\n\nNote from reviewer: your previous answer scored the candidate harshly "
            "(below 50) but listed no improvements. Re-evaluate with the calibrated rubric: "
            "anything below 50 must name at least three substantial, transcript-specific "
            "improvements."
        )
        print("[EVAL] harsh score with no improvements; retrying with calibration nudge")

    return {
        "score": _parse_score(data),
        "strengths": [str(x) for x in (data.get("strengths") or [])][:4],
        "improvements": [str(x) for x in (data.get("improvements") or [])][:4],
        "verdict": str(data.get("verdict") or "").strip(),
    }


def _parse_score(data: dict) -> int:
    if data is None:
        return 0
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))
