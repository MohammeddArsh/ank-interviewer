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

    messages = [
        {"role": "system", "content": "You are a precise assistant. You only respond with valid JSON."},
        {"role": "user", "content": prompts.evaluation_prompt(job_description, resume_text, transcript_text)},
    ]
    data = complete_json(complete, messages, temperature=0.5, max_tokens=1000)

    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    return {
        "score": score,
        "strengths": [str(x) for x in (data.get("strengths") or [])][:4],
        "improvements": [str(x) for x in (data.get("improvements") or [])][:4],
        "verdict": str(data.get("verdict") or "").strip(),
    }
