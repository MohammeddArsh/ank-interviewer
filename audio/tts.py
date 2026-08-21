import os
import re
import subprocess
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor

from config import TEMP_DIR

FFMPEG_DIR = r"C:\Users\mohda\ffmpeg-master-latest-win64-gpl\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_DIR
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

from gtts import gTTS  # noqa: E402 - must set PATH before importing gtts deps


def speak_to_file(text):
    """Generate TTS audio and return the file path (for web use)."""
    tmp_path = os.path.join(TEMP_DIR, "tts_" + uuid.uuid4().hex + ".mp3")
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tmp_path)
    return tmp_path


def _split_sentences(text: str) -> list:
    """Split into subtitle-sized chunks on sentence boundaries, merging tiny fragments."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    merged = []
    for p in parts:
        if merged and len(merged[-1]) + len(p) < 26:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def speak_to_chunks(text: str) -> list:
    """Split text into subtitle-synced TTS chunks.

    Returns [{"text": str, "audio_url": str}, ...]. Chunks are generated in
    parallel so multi-sentence replies don't feel slow.
    """
    chunks = _split_sentences(text)
    if not chunks:
        return []

    paths = []
    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as pool:
        for chunk in chunks:
            tmp_path = os.path.join(TEMP_DIR, "tts_" + uuid.uuid4().hex + ".mp3")
            paths.append((chunk, pool.submit(_render_chunk, chunk, tmp_path)))

    return [
        {"text": chunk, "audio_url": "/audio/" + os.path.basename(fut.result())}
        for chunk, fut in paths
    ]


def _render_chunk(text: str, tmp_path: str) -> str:
    gTTS(text=text, lang="en", slow=False).save(tmp_path)
    return tmp_path


def speak(text):
    """Generate TTS and play immediately (for terminal use)."""
    print("Assistant: " + text)
    tmp_path = speak_to_file(text)
    try:
        subprocess.run(
            [os.path.join(FFMPEG_DIR, "ffplay.exe"), "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            check=True
        )
    except Exception as e:
        print("TTS playback failed: " + str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
