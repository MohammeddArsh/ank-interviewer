import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

# Set MODE as an environment variable in Railway/Render dashboard.
# "openrouter" = OpenRouter LLM + local faster-whisper (free, default) ·
# "local" = Ollama + faster-whisper (fully offline).
MODE = os.getenv("MODE", "openrouter")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Free-tier OpenRouter models rotate constantly (retired to paid, replaced, etc.).
# By default the available free models are auto-discovered at runtime; set
# OPENROUTER_MODELS to a JSON array to PIN a specific list and skip discovery.
OPENROUTER_MODELS = json.loads(os.getenv("OPENROUTER_MODELS", "[]"))

# Used only if discovery fails (no key / network down). Keep recent, still-known
# free slugs + the openrouter/free router as the ultimate always-available fallback.
BOOTSTRAP_OPENROUTER_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]
FREE_MODEL_REFRESH_SECONDS = int(os.getenv("FREE_MODEL_REFRESH_SECONDS", str(6 * 3600)))

# Writable scratch space for TTS/uploaded audio.
# In containers (Docker/Azure) $HOME may not be writable, so default to tempdir.
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# Audio settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
CHANNELS = 1

# Interview settings
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "6"))
FOLLOW_UPS_PER_QUESTION = int(os.getenv("FOLLOW_UPS_PER_QUESTION", "1"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))

# Interview duration presets chosen in the setup wizard. Each maps to the
# total number of planned questions and probing follow-ups per answer.
DURATION_PRESETS = {
    "quick": {"questions": 4, "follow_ups": 0},     # ~8 minutes
    "standard": {"questions": 6, "follow_ups": 1},  # ~18 minutes
    "deep": {"questions": 9, "follow_ups": 1},      # ~30 minutes
}
DEFAULT_DURATION = os.getenv("DEFAULT_DURATION", "standard")


def resolve_duration(key: str) -> tuple[int, int]:
    """Map a preset key to (max_questions, follow_ups_per_question).

    Unknown/missing keys fall back to the env-configured defaults so
    existing deployments behave exactly as before.
    """
    preset = DURATION_PRESETS.get((key or "").strip().lower())
    if preset:
        return preset["questions"], preset["follow_ups"]
    return MAX_QUESTIONS, FOLLOW_UPS_PER_QUESTION

# LLM settings (local mode only)
LLM_MODEL_LOCAL = os.getenv("LLM_MODEL_LOCAL", "llama3.2")

# Speech-to-text settings. STT_BACKEND picks the engine shared by the
# streaming WebSocket path and the legacy WAV fallback:
#   "moonshine" — Moonshine ONNX (default; fast on small CPU instances)
#   "whisper"   — local faster-whisper via WHISPER_MODEL
STT_BACKEND = os.getenv("STT_BACKEND", "moonshine").strip().lower()
MOONSHINE_MODEL = os.getenv("MOONSHINE_MODEL", "base").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SYSTEM_PROMPT = """You are Ank, a friendly and concise voice assistant built by Mohammed Arsh.
Keep answers short and conversational — you are speaking out loud, not writing an essay.
Avoid bullet points or markdown formatting in your responses."""
