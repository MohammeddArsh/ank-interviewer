"""Tests for the STT engine dispatcher (no real model downloads)."""

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from audio import stt_engine


@pytest.fixture(autouse=True)
def _fresh_engine(monkeypatch):
    """Never touch real weights; reset the lazy singletons between tests."""
    monkeypatch.setattr(stt_engine, "_moonshine", None)
    monkeypatch.setattr(stt_engine, "_tokenizer", None)
    yield


def _speech_pcm(ms=1000):
    return np.full(int(16000 * ms / 1000), 800, dtype=np.int16)


class TestGuards:
    def test_none_returns_empty(self):
        assert stt_engine.transcribe_pcm(None) == ""

    def test_silence_returns_empty(self):
        assert stt_engine.transcribe_pcm(np.zeros(16000, dtype=np.int16)) == ""

    def test_too_short_returns_empty(self):
        # Moonshine asserts >0.1 s; fragments must not reach the engine.
        assert stt_engine.transcribe_pcm(np.full(1599, 900, dtype=np.int16)) == ""


class TestDispatch:
    def test_moonshine_backend(self, monkeypatch):
        calls = {}

        class FakeModel:
            def generate(self, audio):
                calls["shape"] = audio.shape
                calls["dtype"] = audio.dtype
                return [[5, 6, 7]]

        class FakeTok:
            def decode_batch(self, batches):
                return [" hello world "]

        monkeypatch.setattr(
            stt_engine, "_get_moonshine", lambda: (FakeModel(), FakeTok())
        )
        out = stt_engine.transcribe_pcm(_speech_pcm())
        assert out == "hello world"
        assert calls["shape"] == (1, 16000)
        assert calls["dtype"] == np.float32

    def test_whisper_backend(self, monkeypatch):
        captured = {}
        pcm = _speech_pcm()

        def fake_whisper(x):
            captured["pcm"] = x
            return "hi there"

        monkeypatch.setattr(stt_engine, "STT_BACKEND", "whisper")
        monkeypatch.setattr(stt_engine, "_transcribe_whisper_pcm", fake_whisper)
        assert stt_engine.transcribe_pcm(pcm) == "hi there"
        assert np.array_equal(captured["pcm"], pcm)  # int16 handed to engine

    def test_unknown_backend_falls_back_to_whisper(self, monkeypatch):
        monkeypatch.setattr(stt_engine, "STT_BACKEND", "bogus")
        monkeypatch.setattr(
            stt_engine, "_transcribe_whisper_pcm", lambda x: "fallback"
        )
        assert stt_engine.transcribe_pcm(_speech_pcm()) == "fallback"

    def test_get_engine_name_tracks_backend(self, monkeypatch):
        monkeypatch.setattr(stt_engine, "STT_BACKEND", "moonshine")
        assert stt_engine.get_engine_name() == "moonshine"
        monkeypatch.setattr(stt_engine, "STT_BACKEND", "whisper")
        assert stt_engine.get_engine_name() == "whisper"


class TestFileTranscription:
    def test_reads_wav_and_dispatches(self, tmp_path, monkeypatch):
        path = str(tmp_path / "answer.wav")
        pcm = _speech_pcm()
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())

        seen = {}
        def fake(x):
            seen["pcm"] = x
            return "from file"

        monkeypatch.setattr(stt_engine, "transcribe_pcm", fake)
        assert stt_engine.transcribe_file(path) == "from file"
        assert np.array_equal(seen["pcm"], pcm)

    def test_non_wav_raises_value_error(self, tmp_path):
        bad = tmp_path / "garbage.wav"
        bad.write_bytes(b"not a wav at all")
        with pytest.raises(ValueError, match="Could not decode"):
            stt_engine.transcribe_file(str(bad))
