"""Tests for the streaming transcription module (segmenter, gate, WS)."""

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from audio import stream_stt
from audio.stream_stt import SegmentTracker, VadGate

SR = 16000


def _pcm(ms: int, value: int = 1000) -> np.ndarray:
    """Int16 buffer of `ms` milliseconds at 16 kHz."""
    return np.full(int(SR * ms / 1000), value, dtype=np.int16)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


class TestSegmentTracker:
    def test_idle_until_speech(self):
        t = SegmentTracker()
        assert t.feed(_silence(100), is_speech=False) == ("idle", None)
        assert not t.open

    def test_open_on_speech(self):
        t = SegmentTracker()
        event, seg = t.feed(_pcm(100), is_speech=True)
        assert event == "open" and seg is None and t.open

    def test_extend_while_speaking(self):
        t = SegmentTracker()
        t.feed(_pcm(100), is_speech=True)
        event, seg = t.feed(_pcm(100), is_speech=True)
        assert event == "extend" and seg is None and t.samples == 2 * int(SR * 0.1)

    def test_close_after_silence(self):
        t = SegmentTracker()
        t.feed(_pcm(200), is_speech=True)
        event, seg = t.feed(_silence(800), is_speech=False)  # > SEGMENT_SILENCE_S
        assert event == "close"
        assert seg is not None and len(seg) == int(SR * (0.2 + 0.8))
        assert not t.open and t.samples == 0

    def test_extend_through_short_pause(self):
        t = SegmentTracker()
        t.feed(_pcm(200), is_speech=True)
        event, _ = t.feed(_silence(300), is_speech=False)  # < silence threshold
        assert event == "extend"

    def test_close_on_max_length(self):
        t = SegmentTracker()
        event, seg = t.feed(_pcm(16000), is_speech=True)  # >= MAX_SEGMENT_S
        assert event == "close" and seg is not None

    def test_flush_returns_open_segment(self):
        t = SegmentTracker()
        t.feed(_pcm(150), is_speech=True)
        seg = t.flush()
        assert seg is not None and len(seg) == int(SR * 0.15)
        assert t.flush() is None  # nothing open anymore

    def test_snapshot_copies_buffer(self):
        t = SegmentTracker()
        t.feed(_pcm(100), is_speech=True)
        snap = t.snapshot()
        assert len(snap) == int(SR * 0.1)
        assert t.samples == len(snap)  # snapshot must not consume the buffer


class TestVadGate:
    def test_rms_fallback_flags_loud_audio(self, monkeypatch):
        monkeypatch.setattr(stream_stt, "_vad", None)
        gate = VadGate()
        assert gate.decide(_pcm(100, value=8000)) is True
        assert gate.decide(_silence(100)) is False

    def test_silero_path_handles_uneven_sizes(self, monkeypatch):
        class FakeVad:
            def __call__(self, x):
                assert len(x) % stream_stt.VAD_WINDOW == 0
                return np.ones(len(x) // stream_stt.VAD_WINDOW)

        monkeypatch.setattr(stream_stt, "_vad", FakeVad())
        gate = VadGate()
        assert gate.decide(_pcm(53)) is True   # 848 samples → 512-window + 336 tail
        assert gate.decide(_pcm(10)) is False  # 336+160=496 < 512 → tail-only, no window


class ScriptedVad:
    """Fake Silero VAD whose per-window verdict each test controls."""

    verdict = 1.0  # 1.0 = every window is speech, 0.0 = silence

    def __call__(self, x):
        n = max(len(x) // stream_stt.VAD_WINDOW, 1)
        return np.full(n, self.verdict)


@pytest.fixture()
def client(monkeypatch):
    """Minimal app with the WS route; whisper and VAD are stubbed."""
    monkeypatch.setattr(stream_stt, "_transcribe_pcm", lambda pcm: f"text-{len(pcm)}")
    monkeypatch.setattr(stream_stt, "_vad", ScriptedVad())
    app = FastAPI()
    app.include_router(stream_stt.router)
    return TestClient(app)


class TestWsTranscribe:
    def test_round_trip_closes_on_silence(self, client):
        vad = stream_stt._vad
        with client.websocket_connect("/ws/transcribe") as ws:
            vad.verdict = 1.0
            ws.send_bytes(_pcm(600).tobytes())
            assert ws.receive_json() == {"type": "start"}
            vad.verdict = 0.0
            ws.send_bytes(_silence(800).tobytes())  # >= SEGMENT_SILENCE_S → close
            msg = ws.receive_json()
            assert msg["type"] == "final"
            assert msg["text"].startswith("text-")

    def test_stop_flushes_open_segment(self, client):
        stream_stt._vad.verdict = 1.0
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_bytes(_pcm(400).tobytes())
            assert ws.receive_json() == {"type": "start"}
            ws.send_text(json.dumps({"type": "stop"}))
            msg = ws.receive_json()
            assert msg["type"] == "final"

    def test_silence_never_opens(self, client):
        stream_stt._vad.verdict = 0.0
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_bytes(_silence(300).tobytes())
            ws.send_text(json.dumps({"type": "stop"}))
            with pytest.raises(WebSocketDisconnect):  # no events queued; server closes
                ws.receive_json()

    def test_empty_text_final_is_suppressed(self, client, monkeypatch):
        monkeypatch.setattr(stream_stt, "_transcribe_pcm", lambda pcm: "")
        stream_stt._vad.verdict = 1.0
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_bytes(_pcm(400).tobytes())
            assert ws.receive_json() == {"type": "start"}
            ws.send_text(json.dumps({"type": "stop"}))
            with pytest.raises(WebSocketDisconnect):  # blank transcription → no final
                ws.receive_json()
