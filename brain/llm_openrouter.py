import random
import threading
import time

import openai
from openai import OpenAI

from config import (
    BOOTSTRAP_OPENROUTER_MODELS,
    FREE_MODEL_REFRESH_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_MODELS,
)

# The OpenAI SDK's default timeout is 600 s and it adds its own retry layer on
# top of ours below. Bound both: the app route already runs in a worker thread,
# so a hung upstream should fail fast and fall back instead of freezing the box.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY or "missing-api-key",
    timeout=30.0,
    max_retries=0,
)

# OpenRouter's free router — always available, picks a working free model per
# request. Serves as the ultimate fallback when specific free slugs rotate out.
FREE_ROUTER = "openrouter/free"

# Curated quality priority for discovered free models (best conversational
# instruct models first). Ordered prefix match, then by context length.
_PRIORITY_PREFIXES = [
    "qwen/qwen3-next",
    "nvidia/nemotron-3",
    "google/gemma-4-31b",
    "google/gemma-3-27b",
    "meta-llama/llama-3.3-70b",
    "meta-llama/llama-3.1-405b",
    "moonshotai/kimi-k2",
    "z-ai/glm-4",
    "openai/gpt-oss",
    "deepseek/deepseek-v3",
    "deepseek/deepseek-r1",
    "mistralai/mistral-small",
    "mistralai/mistral-7b",
]

MAX_RETRIES = 2
_MAX_DISCOVERED = 3
# Hard cap on distinct completion attempts per request (models tried x retries).
# Free-tier models fail together when the free lane is down; there is no point
# grinding through the whole list. With each attempt bounded by the client
# timeout above, worst-case request latency stays ~2-3 min instead of ~10+.
_MAX_TOTAL_ATTEMPTS = 6

_cache = {"models": None, "fetched_at": 0.0}
_lock = threading.Lock()
_last_model = None


def _discover_free_models():
    """Fetch currently-free OpenRouter models, ordered by quality priority.

    Returns a list of model IDs (primary first) or None if discovery fails.
    """
    try:
        resp = client.get("/models", timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    free = []
    for m in data.get("data", []):
        pricing = m.get("pricing") or {}
        if (pricing.get("prompt") or "0") == "0" and (pricing.get("completion") or "0") == "0":
            mid = m.get("id", "")
            if mid.endswith(":free"):
                free.append(m)

    if not free:
        return None

    def rank(m):
        mid = m["id"].lower()
        for i, prefix in enumerate(_PRIORITY_PREFIXES):
            if prefix in mid:
                return (i, -(m.get("context_length") or 0))
        return (len(_PRIORITY_PREFIXES), -(m.get("context_length") or 0))

    free.sort(key=rank)
    models = [m["id"] for m in free[:_MAX_DISCOVERED]]
    if FREE_ROUTER not in models:
        models.append(FREE_ROUTER)
    return models


def _model_list() -> list:
    """Resolve the model list: env pin > cached discovery > bootstrap defaults."""
    if OPENROUTER_MODELS:
        return OPENROUTER_MODELS

    now = time.time()
    with _lock:
        if _cache["models"] is not None and now - _cache["fetched_at"] < FREE_MODEL_REFRESH_SECONDS:
            return _cache["models"]

    discovered = _discover_free_models()
    with _lock:
        if discovered:
            _cache["models"] = discovered
            _cache["fetched_at"] = time.time()
        elif _cache["models"] is None:
            # First resolution failed — fall back to the bootstrap defaults.
            _cache["models"] = list(BOOTSTRAP_OPENROUTER_MODELS)
            _cache["fetched_at"] = time.time()
        # else: keep serving a previously-discovered list even if a refresh failed.
    return _cache["models"]


def _create(messages, temperature, max_tokens):
    """Chat completion with client-side model fallback + exponential-backoff retry.

    Free-tier models rotate and OpenRouter's own `models` fallback does NOT
    advance past "unavailable for free" 404s, so we try each model ourselves.
    """
    models = _model_list()
    last_exc = None
    attempts = 0
    finished = False
    for model in models:
        if finished:
            break
        for attempt in range(MAX_RETRIES):
            if attempts >= _MAX_TOTAL_ATTEMPTS:
                print(f"[LLM] reached {_MAX_TOTAL_ATTEMPTS}-attempt budget; giving up")
                finished = True
                break
            attempts += 1
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"models": models[:3]} if len(models) > 1 else None,
                )
                if not getattr(resp, "choices", None):
                    # OpenRouter answers HTTP 200 with an in-body error object
                    # when every endpoint for the fallback list fails. Treat
                    # it like any other failure and advance to the next model.
                    extra = getattr(resp, "model_extra", None) or {}
                    print(f"[LLM] {model} answered 200 without choices: {extra.get('error') or 'unknown reason'}")
                    last_exc = RuntimeError(f"{model}: no completion choices")
                    break
                global _last_model
                _last_model = resp.model
                print(f"[LLM] answered via {resp.model}")
                return resp
            except openai.NotFoundError as e:
                # Model unavailable/retired — advance to the next model at once.
                print(f"[LLM] {model} unavailable (404) — trying next model")
                last_exc = e
                break
            except Exception as e:
                last_exc = e
                if attempt == MAX_RETRIES - 1:
                    print(f"[LLM] {model} failed after retries: {e}")
                    break
                time.sleep(min((2 ** attempt) + random.uniform(0, 1), 2.0))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("all OpenRouter models failed")


def complete(messages: list, temperature: float = 0.7, max_tokens: int = 512) -> tuple:
    """Returns (reply_text, token_usage_dict) using the OpenAI chat format."""
    response = _create(messages, temperature=temperature, max_tokens=max_tokens)
    if not response.choices:
        raise RuntimeError("model returned no completion choices")
    text = (response.choices[0].message.content or "").strip()

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    meta = getattr(response, "usage", None)
    if meta:
        usage = {
            "prompt_tokens": meta.prompt_tokens or 0,
            "completion_tokens": meta.completion_tokens or 0,
            "total_tokens": meta.total_tokens or 0,
        }
    usage["model"] = getattr(response, "model", None)
    return text, usage


def get_last_model():
    """The model that answered the most recent successful call (diagnostics)."""
    return _last_model


def get_reply(messages: list) -> tuple:
    """Returns (reply_text, token_usage_dict) — generic chat helper."""
    return complete(messages, temperature=0.7, max_tokens=512)
