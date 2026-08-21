import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MODE", "openrouter")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-dummy")

import httpx
import pytest
from openai import NotFoundError

import brain.llm_openrouter as mod


def _free_model(model_id, context_length=100000):
    return {
        "id": model_id,
        "context_length": context_length,
        "pricing": {"prompt": "0", "completion": "0", "request": "0", "image": "0", "web_search": "0"},
    }


def _paid_model(model_id):
    return {
        "id": model_id,
        "context_length": 100000,
        "pricing": {"prompt": "1", "completion": "1"},
    }


def _fake_chat_response(model, text="Hello"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


@pytest.fixture(autouse=True)
def _reset_cache():
    mod._cache["models"] = None
    mod._cache["fetched_at"] = 0.0
    mod._last_model = None
    yield
    mod._cache["models"] = None
    mod._cache["fetched_at"] = 0.0
    mod._last_model = None


def _fake_get_response(models):
    resp = SimpleNamespace()
    resp.json = lambda: {"data": models}
    resp.raise_for_status = lambda: None
    return resp


class TestDiscovery:
    def test_filters_free_and_orders_by_priority(self, monkeypatch):
        monkeypatch.setattr(
            mod.client,
            "get",
            lambda *a, **k: _fake_get_response([
                _paid_model("meta-llama/llama-3.3-70b-instruct:free"),
                _free_model("nvidia/nemotron-3-ultra-550b-a55b:free", 1000000),
                _free_model("some/random-small:free", 8000),
                _free_model("qwen/qwen3-next-80b-a3b-instruct:free", 262144),
            ]),
        )
        models = mod._discover_free_models()
        assert models[0] == "qwen/qwen3-next-80b-a3b-instruct:free"
        assert models[1] == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert "some/random-small:free" in models
        assert "openrouter/free" in models
        assert "meta-llama/llama-3.3-70b-instruct:free" not in models

    def test_returns_none_when_no_free_models(self, monkeypatch):
        monkeypatch.setattr(
            mod.client,
            "get",
            lambda *a, **k: _fake_get_response([_paid_model("x/y:free")]),
        )
        assert mod._discover_free_models() is None

    def test_returns_none_on_network_error(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(mod.client, "get", boom)
        assert mod._discover_free_models() is None


class TestModelList:
    def test_uses_bootstrap_when_discovery_fails(self, monkeypatch):
        monkeypatch.setattr(mod, "_discover_free_models", lambda: None)
        models = mod._model_list()
        assert mod.FREE_ROUTER in models
        assert models == list(__import__("config").BOOTSTRAP_OPENROUTER_MODELS)

    def test_keeps_previous_list_when_refresh_fails(self, monkeypatch):
        monkeypatch.setattr(mod, "_discover_free_models", lambda: None)
        mod._cache["models"] = ["a/model:free", "openrouter/free"]
        mod._cache["fetched_at"] = 0.0  # stale → triggers refresh attempt
        assert mod._model_list() == ["a/model:free", "openrouter/free"]

    def test_pin_env_list_takes_precedence(self, monkeypatch):
        mod.OPENROUTER_MODELS = ["pinned/model:free"]
        try:
            assert mod._model_list() == ["pinned/model:free"]
        finally:
            mod.OPENROUTER_MODELS = __import__("config").OPENROUTER_MODELS


class TestCreate:
    def test_404_advances_to_next_model(self, monkeypatch):
        monkeypatch.setattr(mod, "_model_list", lambda: ["dead/model:free", "good/model:free"])
        calls = []

        def fake_create(*args, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
                raise NotFoundError(
                    "This model is unavailable for free.",
                    response=httpx.Response(404, request=req),
                    body={"error": {"message": "unavailable", "code": 404}},
                )
            return _fake_chat_response("good/model:free")

        monkeypatch.setattr(mod.client.chat.completions, "create", fake_create)
        resp = mod._create([{"role": "user", "content": "hi"}], 0.7, 100)
        assert calls == ["dead/model:free", "good/model:free"]
        assert resp.model == "good/model:free"
        assert mod._last_model == "good/model:free"

    def test_success_sets_last_model(self, monkeypatch):
        monkeypatch.setattr(mod, "_model_list", lambda: ["meta-llama/llama-3.3-70b-instruct:free"])
        monkeypatch.setattr(
            mod.client.chat.completions,
            "create",
            lambda *a, **k: _fake_chat_response("meta-llama/llama-3.3-70b-instruct:free"),
        )
        text, usage = mod.complete([{"role": "user", "content": "hi"}])
        assert text == "Hello"
        assert usage["model"] == "meta-llama/llama-3.3-70b-instruct:free"
        assert mod.get_last_model() == "meta-llama/llama-3.3-70b-instruct:free"

    def test_empty_choices_raises(self, monkeypatch):
        monkeypatch.setattr(mod, "_model_list", lambda: ["meta-llama/llama-3.3-70b-instruct:free"])
        monkeypatch.setattr(
            mod.client.chat.completions,
            "create",
            lambda *a, **k: SimpleNamespace(model="meta-llama/llama-3.3-70b-instruct:free", choices=[]),
        )
        with pytest.raises(RuntimeError, match="no completion choices"):
            mod.complete([{"role": "user", "content": "hi"}])
