import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MODE", "openrouter")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-dummy")

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_index_has_interviewer_ui(client):
    r = client.get("/")
    body = r.text
    assert "AI Mock Interviewer" in body
    for persona in ("Recruiter", "Technical Lead", "HR Manager", "Executive"):
        assert persona in body
    assert "av-mouth-wide" in body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analytics(client):
    r = client.get("/analytics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_metrics_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "# HELP" in body
    assert "ank_chat_requests_total" in body


def test_chat_requires_audio(client):
    r = client.post("/chat")
    assert r.status_code == 422


def test_reset(client):
    r = client.post("/reset")
    assert r.status_code == 200
    assert r.json() == {"status": "reset"}
