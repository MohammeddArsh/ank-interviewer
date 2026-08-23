"""Shared speech-to-text dispatcher.

Both the streaming WebSocket path (audio/stream_stt.py) and the legacy WAV
upload path route through here. STT_BACKEND selects the engine:

  "moonshine" — vendored Moonshine ONNX models (default; several times
                faster than real-time on small CPU instances)
  "whisper"   — local faster-whisper (audio/stt_local.py)

Engines load lazily so a deployment only ever pays memory for the one it
uses, and a single lock serialises inference on CPU.
"""

from __future__ import annotations

import threading

import numpy as np

from config import MOONSHINE_MODEL, MOONSHINE_PRECISION, STT_BACKEND

_lock = threading.Lock()
_moonshine = None
_tokenizer = None

# Moonshine asserts >0.1 s of audio per call; shorter segments are silence
# fragments not worth decoding.
_MIN_PCM_SAMPLES = 1600


def _get_moonshine():
    global _moonshine, _tokenizer
    if _moonshine is None:
        with _lock:
            if _moonshine is None:
                from audio._vendor.moonshine_onnx import MoonshineOnnxModel
                from audio._vendor.moonshine_onnx.transcribe import load_tokenizer

                print(f"Loading local Moonshine '{MOONSHINE_MODEL}' "
                      f"ONNX model ({MOONSHINE_PRECISION})...")
                _moonshine = MoonshineOnnxModel(
                    model_name=MOONSHINE_MODEL, model_precision=MOONSHINE_PRECISION
                )
                _tokenizer = load_tokenizer()
                print("Moonshine ready.")
    return _moonshine, _tokenizer


def transcribe_pcm(pcm_int16: np.ndarray) -> str:
    """Transcribe an Int16 LE mono buffer at 16 kHz with the configured engine."""
    if pcm_int16 is None or pcm_int16.size < _MIN_PCM_SAMPLES or not np.any(pcm_int16):
        return ""
    if STT_BACKEND == "moonshine":
        return _transcribe_moonshine(pcm_int16)
    if STT_BACKEND != "whisper":
        print(f"[STT] Unknown STT_BACKEND '{STT_BACKEND}' — falling back to whisper")
    return _transcribe_whisper_pcm(pcm_int16)


def _transcribe_moonshine(pcm_int16: np.ndarray) -> str:
    model, tokenizer = _get_moonshine()
    audio = pcm_int16.astype(np.float32) / 32768.0
    # SegmentTracker caps segments at 15 s; Moonshine accepts up to 64 s.
    with _lock:
        tokens = model.generate(audio[None, :])
    return tokenizer.decode_batch(tokens)[0].strip()


def _transcribe_whisper_pcm(pcm_int16: np.ndarray) -> str:
    from audio.stt_local import model as whisper_model

    audio = pcm_int16.astype(np.float32) / 32768.0
    with _lock:
        segments, _info = whisper_model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return "".join(s.text for s in segments).strip()


def transcribe_file(path: str) -> str:
    """Transcribe a RIFF/WAV file recorded by the client."""
    import wave

    try:
        with wave.open(path, "rb") as w:
            frames = w.readframes(w.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
    except Exception as e:
        raise ValueError(f"Could not decode {path} as WAV: {e}") from e
    return transcribe_pcm(pcm)


def get_engine_name() -> str:
    return "moonshine" if STT_BACKEND == "moonshine" else "whisper"
