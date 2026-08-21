import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MODE", "openrouter")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-dummy")

import pytest
from fastapi.testclient import TestClient

import app as app_module

ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

FIXED_PLAN = {
    "greeting": "Welcome! Here's how we'll run today.",
    "warmup_question": "Tell me about yourself.",
    "sections": [
        {"title": "Introduction", "focus": "Warm-up", "questions": ["Tell me about yourself."]},
        {"title": "Technical", "focus": "Technical skills", "questions": ["Explain your stack.", "Describe a hard bug."]},
        {"title": "Behavioral", "focus": "Behavior", "questions": ["Tell me about a conflict."]},
    ],
    "closing": "That's everything.",
}

EVAL_JSON = (
    '{"score": 82, "strengths": ["Clear answers", "Good examples"], '
    '"improvements": ["Be more concise"], "verdict": "Solid performance."}'
)


def fake_engine_complete(messages, temperature=0.7, max_tokens=512):
    content = messages[-1]["content"]
    if "probes deeper" in content:
        return ("FOLLOW_UP", ZERO_USAGE)
    if "NEXT planned" in content:
        return ("BRIDGE", ZERO_USAGE)
    if "previews the next section" in content:
        return ("TRANSITION", ZERO_USAGE)
    if "has no questions" in content:
        return ("CLOSING_NO", ZERO_USAGE)
    if "has questions" in content:
        return ("CLOSING_YES", ZERO_USAGE)
    if "any questions" in content:
        return ("CLOSING_OFFER", ZERO_USAGE)
    return ("UNKNOWN", ZERO_USAGE)


@pytest.fixture(autouse=True)
def mock_ai(monkeypatch):
    monkeypatch.setattr("interview.engine.generate_plan", lambda jd, resume, max_q: dict(FIXED_PLAN))
    monkeypatch.setattr("interview.engine.complete", fake_engine_complete)
    monkeypatch.setattr("interview.evaluator.complete", lambda *a, **k: (EVAL_JSON, ZERO_USAGE))
    monkeypatch.setattr("interview.routes.transcribe", lambda path: "My answer")
    monkeypatch.setattr("interview.routes.speak_to_chunks", lambda text: [])
    yield


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def _start(client):
    r = client.post("/interview/start", data={
        "job_description": "Senior Python backend engineer",
        "resume_text": "5 years Python, FastAPI, AWS",
        "interviewer": '{"name": "Alex", "role": "Recruiter"}',
    })
    return r


def test_start_builds_interview(client):
    r = _start(client)
    assert r.status_code == 200
    data = r.json()
    assert data["utterance"].startswith("Welcome!")
    assert data["state"]["phase"] == "answering_main"
    assert data["state"]["total_questions"] == 4
    assert data["state"]["current_question"] == "Tell me about yourself."
    assert data["state"]["interviewer"]["name"] == "Alex"


def test_start_requires_inputs(client):
    r = client.post("/interview/start", data={"job_description": "x", "resume_text": ""})
    assert r.status_code == 400


def test_prepare_then_begin(client):
    r = client.post("/interview/prepare", data={
        "job_description": "Senior Python backend engineer",
        "resume_text": "5 years Python, FastAPI, AWS",
        "interviewer": '{"name": "Alex", "role": "Recruiter"}',
    })
    assert r.status_code == 200
    plan = r.json()["plan"]
    assert plan["sections"] == ["Introduction", "Technical", "Behavioral"]
    assert plan["total_questions"] == 4
    assert plan["interviewer"]["name"] == "Alex"

    r = client.get("/interview/state")
    assert r.json()["session"]["phase"] == "ready"

    r = client.post("/interview/begin")
    assert r.status_code == 200
    d = r.json()
    assert d["utterance"].startswith("Welcome!")
    assert d["state"]["phase"] == "answering_main"
    assert d["state"]["current_question"] == "Tell me about yourself."


def test_begin_requires_prepare(client):
    client.post("/interview/reset")
    r = client.post("/interview/begin")
    assert r.status_code == 400


def test_full_flow_followup_bridge_transition_close(client):
    _start(client)
    wav = io.BytesIO(b"RIFFfake-wav")
    files = {"audio": ("a.webm", wav, "audio/webm")}

    # Q1 (Introduction): answer warm-up -> follow-up
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "FOLLOW_UP"
    assert d["transcript"] == "My answer"
    assert d["state"]["is_followup"] is True

    # follow-up answered -> transition into Technical section
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "TRANSITION"
    assert d["state"]["current_section"] == "Technical"
    assert d["state"]["is_followup"] is False

    # Q2 (Technical): answer -> follow-up
    r = client.post("/interview/answer", files=files)
    assert r.json()["utterance"] == "FOLLOW_UP"

    # follow-up answered -> bridge to next question (still Technical)
    r = client.post("/interview/answer", files=files)
    assert r.json()["utterance"] == "BRIDGE"
    assert r.json()["state"]["current_section"] == "Technical"

    # Q3 (Technical): answer -> follow-up
    r = client.post("/interview/answer", files=files)
    assert r.json()["utterance"] == "FOLLOW_UP"

    # follow-up answered -> transition into Behavioral section
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "TRANSITION"
    assert d["state"]["current_section"] == "Behavioral"

    # Q4 (Behavioral): answer -> follow-up
    r = client.post("/interview/answer", files=files)
    assert r.json()["utterance"] == "FOLLOW_UP"

    # follow-up answered -> closing offer
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "CLOSING_OFFER"
    assert d["state"]["phase"] == "closing"

    # candidate's answer is substantive -> treated as having a question -> wrap-up + done
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "CLOSING_YES"
    assert d["done"] is True
    assert d["evaluation"]["score"] == 82
    assert d["evaluation"]["verdict"] == "Solid performance."
    assert "evaluation_segments" in d


def test_candidate_with_questions_closes_with_ack(client):
    client.post("/interview/reset")
    _start(client)
    wav = io.BytesIO(b"RIFFfake-wav")
    files = {"audio": ("a.webm", wav, "audio/webm")}
    for _ in range(8):  # reach closing phase
        client.post("/interview/answer", files=files)
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "CLOSING_YES"  # "My answer" is treated as a question
    assert d["done"] is True
    assert d["evaluation"]["score"] == 82


def test_candidate_no_questions_closes(client, monkeypatch):
    client.post("/interview/reset")
    _start(client)
    monkeypatch.setattr("interview.routes.transcribe", lambda p: "No, I don't have any questions")
    wav = io.BytesIO(b"RIFFfake-wav")
    files = {"audio": ("a.webm", wav, "audio/webm")}
    for _ in range(8):  # reach closing phase
        client.post("/interview/answer", files=files)
    r = client.post("/interview/answer", files=files)
    d = r.json()
    assert d["utterance"] == "CLOSING_NO"
    assert d["done"] is True


def test_skip_moves_forward(client):
    client.post("/interview/reset")
    _start(client)
    r = client.post("/interview/skip")
    d = r.json()
    assert d["utterance"] == "FOLLOW_UP"  # warm-up skipped -> follow-up of warm-up
    assert d["state"]["phase"] == "answering_followup"


def test_end_early_returns_evaluation(client):
    client.post("/interview/reset")
    _start(client)
    r = client.post("/interview/end")
    d = r.json()
    assert d["done"] is True
    assert d["pending"] is True
    for _ in range(100):
        j = client.get("/interview/results").json()
        if j.get("ready"):
            assert j["evaluation"]["score"] == 82
            assert j["evaluation_segments"] == []
            return
        time.sleep(0.05)
    raise AssertionError("evaluation never became ready")


def test_state_before_start(client):
    client.post("/interview/reset")
    r = client.get("/interview/state")
    assert r.status_code == 200
    assert r.json()["session"] is None


def test_upload_txt(client):
    r = client.post("/interview/upload", files={"file": ("resume.txt", b"python fastapi aws", "text/plain")})
    assert r.status_code == 200
    assert r.json()["text"] == "python fastapi aws"


def test_upload_unsupported(client):
    r = client.post("/interview/upload", files={"file": ("resume.png", b"\x89PNG", "image/png")})
    assert r.status_code == 400


def test_answer_without_session(client):
    client.post("/interview/reset")
    r = client.post("/interview/answer", files={"audio": ("a.webm", b"x", "audio/webm")})
    assert r.status_code == 400


def test_plan_generates_json(monkeypatch):
    from interview import plan
    monkeypatch.setattr("interview.plan.complete", lambda *a, **k: (
        '```json\n{"greeting":"Hi","warmup_question":"W?","sections":[{"title":"Introduction","focus":"w","questions":["W?"]},{"title":"Tech","focus":"t","questions":["Q1"]}],"closing":"Bye"}\n```',
        ZERO_USAGE,
    ))
    result = plan.generate_plan("jd", "resume", 6)
    assert result["greeting"] == "Hi"
    assert result["warmup_question"] == "W?"
    assert result["sections"][1]["questions"] == ["Q1"]


def test_evaluator_parses_json(monkeypatch):
    from interview import evaluator
    monkeypatch.setattr("interview.evaluator.complete", lambda *a, **k: (EVAL_JSON, ZERO_USAGE))
    result = evaluator.evaluate("jd", "resume", [])
    assert result["score"] == 82
    assert "Clear answers" in result["strengths"]


def _session_with_plan():
    from interview import engine
    sess = engine.InterviewSession("jd", "resume")
    sess.plan = dict(FIXED_PLAN)
    sess.sections = [dict(s) for s in FIXED_PLAN["sections"]]
    sess.total_questions = 6
    sess.current_question = "Explain your stack."
    return sess


def test_fallback_bridge_is_natural_when_llm_fails(monkeypatch):
    from interview import engine

    def boom(messages, temperature=0.7, max_tokens=512):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("interview.engine.complete", boom)
    monkeypatch.setattr("interview.engine.time.sleep", lambda *a, **k: None)

    sess = _session_with_plan()
    out = sess._generate(engine._BRIDGE_INS)

    assert "Let's move on." not in out
    assert "current question was" not in out.lower()
    assert out.startswith("Thanks for that.")
    assert "Tell me about yourself." in out


def test_fallback_followup_does_not_leak_internal_text(monkeypatch):
    from interview import engine

    def boom(messages, temperature=0.7, max_tokens=512):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("interview.engine.complete", boom)
    monkeypatch.setattr("interview.engine.time.sleep", lambda *a, **k: None)

    sess = _session_with_plan()
    sess.phase = "answering_followup"
    out = sess._generate(engine._FOLLOW_UP_INS)

    assert "Let's move on." not in out
    assert "current question was" not in out.lower()
    assert "Explain your stack." not in out
    assert out.startswith("Thanks for sharing that.")


def test_transition_prompt_is_coherent(monkeypatch):
    from interview import engine

    captured = {}

    def fake(messages, temperature=0.7, max_tokens=512):
        captured["content"] = messages[-1]["content"]
        if "previews the next section" in captured["content"]:
            return ("Let's move to Technical. Explain your stack.", ZERO_USAGE)
        return ("FOLLOW_UP", ZERO_USAGE)

    monkeypatch.setattr("interview.engine.complete", fake)

    sess = engine.InterviewSession("jd", "resume")
    sess.plan = dict(FIXED_PLAN)
    sess.sections = [dict(s) for s in FIXED_PLAN["sections"]]
    sess.total_questions = 4
    sess.current_question = "Tell me about yourself."
    sess.phase = "answering_main"

    sess.handle_answer("I did X")        # warm-up answered -> follow-up
    sess.handle_answer("more detail")    # follow-up answered -> advance -> transition

    content = captured["content"]
    assert "finished section 'Introduction'" in content
    assert "next section 'Technical'" in content
    assert "question 1 of 2" in content
    assert "Currently in section" not in content


def test_generate_retries_meta_reasoning_output(monkeypatch):
    from interview import engine

    calls = {"n": 0}

    def fake(messages, temperature=0.6, max_tokens=512):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("The user wants me to act as Priya, a hiring manager. I need to ask a follow-up question.", ZERO_USAGE)
        return ("Could you tell me more about that?", ZERO_USAGE)

    monkeypatch.setattr("interview.engine.complete", fake)
    monkeypatch.setattr("interview.engine.time.sleep", lambda *a, **k: None)

    sess = _session_with_plan()
    sess.phase = "answering_followup"
    out = sess._generate(engine._FOLLOW_UP_INS)

    assert out == "Could you tell me more about that?"
    assert calls["n"] == 2


def test_generate_uses_fallback_when_always_rambling(monkeypatch):
    from interview import engine

    def fake(messages, temperature=0.6, max_tokens=512):
        return ("The user wants me to act as Priya. I need to transition to the next section.", ZERO_USAGE)

    monkeypatch.setattr("interview.engine.complete", fake)
    monkeypatch.setattr("interview.engine.time.sleep", lambda *a, **k: None)

    sess = _session_with_plan()
    out = sess._generate(engine._BRIDGE_INS)

    assert out.startswith("Thanks for that.")
    assert "current question was" not in out.lower()
