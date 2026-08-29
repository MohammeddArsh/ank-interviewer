"""Regression guard for the free-tier crash loop.

The interview/chat routes must run on FastAPI's threadpool (sync `def`), never
on the asyncio event loop: while one slow request (LLM / STT / gTTS) is
mid-flight, the transcription WebSocket and `/health` have to stay responsive.
If someone converts a route back to `async def` and calls blocking code, these
tests hang / fail.
"""

import inspect
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MODE", "openrouter")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-dummy")

import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from audio import stream_stt

SR = 16000

FIXED_PLAN = {
    "greeting": "Welcome! Here's how we'll run today.",
    "warmup_question": "Tell me about yourself.",
    "sections": [{"title": "Introduction", "focus": "Warm-up", "questions": ["Tell me about yourself."]}],
    "closing": "That's everything.",
}
EVAL_JSON = (
    '{"score": 80, "strengths": ["Clear"], "improvements": ["Concise"], "verdict": "Good."}'
)
FORM = {
    "job_description": "Senior Python backend engineer",
    "resume_text": "5 years Python, FastAPI",
    "interviewer": '{"name": "Alex", "role": "Recruiter"}',
}


def _pcm(ms: int, value: int = 1000) -> np.ndarray:
    return np.full(int(SR * ms / 1000), value, dtype=np.int16)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


class ScriptedVad:
    verdict = 1.0

    def __call__(self, x):
        n = max(len(x) // stream_stt.VAD_WINDOW, 1)
        return np.full(n, self.verdict)


def _slow_plan(jd, resume, max_q):
    """Plan generation that mimicks a slow OpenRouter round-trip."""
    time.sleep(0.6)
    return dict(FIXED_PLAN)


def _make_clients():
    ws_client = TestClient(app_module.app)
    route_client = TestClient(app_module.app)
    health_client = TestClient(app_module.app)
    ws_client.__enter__()
    route_client.__enter__()
    health_client.__enter__()
    return ws_client, route_client, health_client


def _close_clients(*clients):
    for c in clients:
        try:
            c.__exit__(None, None, None)
        except Exception:
            pass


def test_blocking_routes_are_sync_not_coro():
    from interview.routes import (
        answer,
        answer_text,
        begin,
        end,
        prepare,
        reset,
        results,
        skip,
        start,
        state,
    )

    assert not inspect.iscoroutinefunction(start)
    assert not inspect.iscoroutinefunction(prepare)
    assert not inspect.iscoroutinefunction(begin)
    assert not inspect.iscoroutinefunction(answer)
    assert not inspect.iscoroutinefunction(answer_text)
    assert not inspect.iscoroutinefunction(skip)
    assert not inspect.iscoroutinefunction(end)
    assert not inspect.iscoroutinefunction(results)
    assert not inspect.iscoroutinefunction(state)
    assert not inspect.iscoroutinefunction(reset)
    assert not inspect.iscoroutinefunction(app_module.chat)


def test_stream_ws_and_health_survive_slow_sync_route(monkeypatch):
    """While /interview/prepare blocks for ~0.6s, the streaming WS and /health
    must keep answering — otherwise the interview WebSocket dies (1011) and
    Render restarts the instance."""
    vad = ScriptedVad()
    monkeypatch.setattr(stream_stt, "_vad", vad)
    monkeypatch.setattr(stream_stt, "_transcribe_pcm", lambda pcm: f"text-{len(pcm)}")
    monkeypatch.setattr("interview.engine.generate_plan", _slow_plan)
    monkeypatch.setattr("interview.evaluator.complete", lambda *a, **k: (EVAL_JSON, {}))
    monkeypatch.setattr("interview.routes.transcribe", lambda path: "My answer")
    monkeypatch.setattr("interview.routes.speak_to_chunks", lambda text: [])

    ws_client, route_client, health_client = _make_clients()
    try:
        with ws_client.websocket_connect("/ws/transcribe") as ws:
            result = {}

            def post_prepare():
                result["resp"] = route_client.post("/interview/prepare", data=FORM)

            t = threading.Thread(target=post_prepare)
            t.start()
            time.sleep(0.1)  # let the sync route get mid-flight

            # The streaming socket must still work while the route is busy.
            vad.verdict = 1.0
            ws.send_bytes(_pcm(400).tobytes())
            assert ws.receive_json() == {"type": "start"}

            # /health must answer well within the route's sleep window.
            t0 = time.monotonic()
            for _ in range(4):
                r = health_client.get("/health")
                assert r.status_code == 200
            assert time.monotonic() - t0 < 0.45

            vad.verdict = 0.0
            ws.send_bytes(_silence(800).tobytes())
            assert ws.receive_json()["type"] == "final"
            ws.send_text(json.dumps({"type": "stop"}))
            assert ws.receive_json() == {"type": "done"}

            t.join(timeout=10)
            assert not t.is_alive()
            assert result["resp"].status_code == 200
    finally:
        _close_clients(ws_client, route_client, health_client)


def test_health_stays_responsive_while_slow_route_runs(monkeypatch):
    monkeypatch.setattr("interview.engine.generate_plan", _slow_plan)
    monkeypatch.setattr("interview.evaluator.complete", lambda *a, **k: (EVAL_JSON, {}))
    monkeypatch.setattr("interview.routes.speak_to_chunks", lambda text: [])

    route_client = TestClient(app_module.app)
    health_client = TestClient(app_module.app)
    route_client.__enter__()
    health_client.__enter__()
    try:
        result = {}

        def post_prepare():
            result["resp"] = route_client.post("/interview/prepare", data=FORM)

        t = threading.Thread(target=post_prepare)
        t.start()
        time.sleep(0.1)

        t0 = time.monotonic()
        for _ in range(4):
            r = health_client.get("/health")
            assert r.status_code == 200
            assert r.json() == {"status": "ok"}
        assert time.monotonic() - t0 < 0.45

        t.join(timeout=10)
        assert not t.is_alive()
        assert result["resp"].status_code == 200
    finally:
        _close_clients(route_client, health_client)
