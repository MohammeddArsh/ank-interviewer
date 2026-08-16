import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

# Set MODE as an environment variable in Railway/Render dashboard
# Defaults to "openai" if not set
MODE = os.getenv("MODE", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Writable scratch space for TTS/uploaded audio.
# In containers (Docker/Azure) $HOME may not be writable, so default to tempdir.
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# Audio settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
CHANNELS = 1

# LLM settings
LLM_MODEL_OPENAI = "gpt-4o-mini"
LLM_MODEL_LOCAL  = "llama3.2"

# Whisper settings (local mode only)
WHISPER_MODEL = "base"

SYSTEM_PROMPT = """You are Ank, a friendly and concise voice assistant built by Mohammed Arsh.
Keep answers short and conversational — you are speaking out loud, not writing an essay.
Avoid bullet points or markdown formatting in your responses."""
