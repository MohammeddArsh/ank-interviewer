"""Generate the structured interview outline (sections + questions)."""

from brain.llm import complete
from interview import prompts
from interview.json_util import complete_json


def generate_plan(job_description: str, resume_text: str, max_questions: int) -> dict:
    """Return {greeting, warmup_question, sections: [{title, focus, questions}], closing}."""
    messages = [
        {"role": "system", "content": "You are a precise assistant. You only respond with valid JSON."},
        {"role": "user", "content": prompts.plan_prompt(job_description, resume_text, max_questions)},
    ]
    data = complete_json(complete, messages, temperature=0.7, max_tokens=1800)

    sections = data.get("sections") or []
    warmup = (data.get("warmup_question") or "").strip()
    if warmup:
        if sections:
            sections[0]["questions"] = [warmup] + [
                q for q in sections[0].get("questions", []) if q != warmup
            ]
        else:
            sections = [{"title": "Introduction", "focus": "Warm-up", "questions": [warmup]}]

    # Guardrails: force exactly max_questions total, drop empty sections.
    flat = [q for s in sections for q in s.get("questions", [])][:max_questions]
    counts = [len(s.get("questions", [])) for s in sections]
    idx = 0
    for i, s in enumerate(sections):
        n = min(counts[i], max(0, len(flat) - idx))
        s["questions"] = flat[idx:idx + n]
        idx += n
    sections = [s for s in sections if s.get("questions")]

    return {
        "greeting": (data.get("greeting") or "").strip(),
        "warmup_question": warmup,
        "sections": sections,
        "closing": (data.get("closing") or "").strip(),
    }
