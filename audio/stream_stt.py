"""Real-time streaming transcription over WebSocket.

The browser streams raw 16 kHz mono PCM (Int16 little-endian) frames while
the candidate speaks. Silero VAD (bundled with faster-whisper) segments
speech; the currently-open segment is re-transcribed periodically and pushed
back as live partials, and every closed segment is emitted as a final.

Server -> client JSON events:
  {"type": "start"}                     a speech segment opened
  {"type": "partial", "text": "..."}    in-progress text of the open segment
  {"type": "final", "text": "..."}      completed segment text

Client -> server:
  binary frames   raw Int16 LE PCM @ 16 kHz mono
  {"type": "stop"} flush the open segment and close cleanly
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import SAMPLE_RATE

router = APIRouter()

SEGMENT_SILENCE_S = 0.7      # silence that closes an open segment
MAX_SEGMENT_S = 15.0         # hard cap so worst-case latency stays bounded
PARTIAL_INTERVAL_S = 0.8     # minimum spacing between partial transcriptions
PARTIAL_NEW_AUDIO_S = 0.5    # skip partials until this much new speech audio
SPEECH_START_PROB = 0.5
STOP_GRACE_S = 2.0           # allow the last final to finish before closing
VAD_WINDOW = 512             # silero expects multiples of 512 samples @16k

# One worker => transcriptions never interleave on CPU.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

try:  # bundled with faster-whisper (onnxruntime); degrade to RMS if missing
    from faster_whisper.vad import get_vad_model
    _vad = get_vad_model()
except Exception:
    _vad = None


def _transcribe_pcm(pcm_int16: np.ndarray) -> str:
    """Transcribe an Int16 PCM buffer with the configured STT engine."""
    from audio.stt_engine import transcribe_pcm

    return transcribe_pcm(pcm_int16)


class VadGate:
    """Per-chunk speech decisions. Silero when available, RMS floor fallback."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, threshold: float = SPEECH_START_PROB):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._tail = np.zeros(0, dtype=np.float32)
        self.noise_floor = 0.01

    def decide(self, pcm_int16: np.ndarray) -> bool:
        x = pcm_int16.astype(np.float32) / 32768.0
        if _vad is None:
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            speech = rms > max(self.noise_floor * 3.0, 0.012)
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * min(rms, self.noise_floor * 4.0)
            return speech
        buf = np.concatenate([self._tail, x])
        usable = (buf.size // VAD_WINDOW) * VAD_WINDOW
        self._tail = buf[usable:]
        if usable == 0:
            return False
        probs = np.asarray(_vad(buf[:usable]))
        return bool(probs.size and float(np.max(probs)) >= self.threshold)


class SegmentTracker:
    """Turns per-chunk VAD decisions into open/extend/close segment events."""

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 silence_s: float = SEGMENT_SILENCE_S, max_s: float = MAX_SEGMENT_S):
        self.sample_rate = sample_rate
        self.silence_samples = int(silence_s * sample_rate)
        self.max_samples = int(max_s * sample_rate)
        self.open = False
        self._chunks: list[np.ndarray] = []
        self.samples = 0
        self._silent_run = 0

    def feed(self, pcm: np.ndarray, is_speech: bool) -> tuple[str, np.ndarray | None]:
        """Returns ('idle'|'open'|'extend'|'close', full-segment PCM on close)."""
        n = pcm.shape[0]
        if not self.open:
            if not is_speech:
                return "idle", None
            self.open = True
            self._chunks = [pcm]
            self.samples = n
            self._silent_run = 0
            if n >= self.max_samples:
                return "close", self._take()
            return "open", None

        self._chunks.append(pcm)
        self.samples += n
        self._silent_run = 0 if is_speech else self._silent_run + n
        if self.samples >= self.max_samples or self._silent_run >= self.silence_samples:
            return "close", self._take()
        return "extend", None

    def flush(self) -> np.ndarray | None:
        """Close an open segment (if any), e.g. on stop/disconnect."""
        return self._take() if self.open else None

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)

    def _take(self) -> np.ndarray:
        seg = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)
        self.open = False
        self._chunks = []
        self.samples = 0
        self._silent_run = 0
        return seg


@router.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    started = time.monotonic()

    tracker = SegmentTracker()
    gate = VadGate()
    finals: asyncio.Queue = asyncio.Queue()
    partial_state = {"at": started, "audio_seen": 0, "in_flight": False}

    async def emit_final(segment: np.ndarray):
        try:
            text = await loop.run_in_executor(_pool, _transcribe_pcm, segment)
            if text:
                await ws.send_json({"type": "final", "text": text})
        except Exception:
            pass  # socket closed mid-transcription — nothing to deliver

    async def final_worker():
        while True:
            item = await finals.get()
            if item is None:
                return
            await emit_final(item)

    async def maybe_partial():
        now = time.monotonic()
        if partial_state["in_flight"] or not tracker.open:
            return
        if (now - partial_state["at"] < PARTIAL_INTERVAL_S
                or tracker.samples - partial_state["audio_seen"] < int(PARTIAL_NEW_AUDIO_S * tracker.sample_rate)):
            return
        tail = tracker.snapshot()
        if not tail.size:
            return
        partial_state.update(in_flight=True, at=now, audio_seen=tracker.samples)
        try:
            text = await loop.run_in_executor(_pool, _transcribe_pcm, tail)
            if text:
                await ws.send_json({"type": "partial", "text": text})
        except Exception:
            pass
        finally:
            partial_state["in_flight"] = False

    worker = asyncio.create_task(final_worker())
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                pcm = np.frombuffer(data, dtype=np.int16)
                if not pcm.size:
                    continue
                event, segment = tracker.feed(pcm, gate.decide(pcm))
                if event == "open":
                    partial_state.update(at=time.monotonic(), audio_seen=0)
                    await ws.send_json({"type": "start"})
                elif event == "close" and segment is not None and segment.size:
                    await finals.put(segment)
                elif event == "extend":
                    await maybe_partial()
            elif (raw := message.get("text")):
                try:
                    control = json.loads(raw)
                except ValueError:
                    continue
                if control.get("type") == "stop":
                    tail = tracker.flush()
                    if tail is not None and tail.size:
                        await finals.put(tail)
                    break
    except WebSocketDisconnect:
        pass
    finally:
        tail = tracker.flush()
        if tail is not None and tail.size:
            await finals.put(tail)
        await finals.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=STOP_GRACE_S)
        except (asyncio.TimeoutError, Exception):
            worker.cancel()
        try:
            await ws.close()
        except Exception:
            pass
