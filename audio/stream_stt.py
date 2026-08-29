"""Real-time streaming transcription over WebSocket.

The browser streams raw 16 kHz mono PCM (Int16 little-endian) frames while
the candidate speaks. Silero VAD (bundled ONNX, loaded directly through
onnxruntime) segments speech; every closed segment is transcribed once and
pushed back as a final caption, followed by a "done" frame once nothing is
pending.

There are deliberately no live mid-word partials: they re-transcribe the
growing segment every few hundred milliseconds, which on small CPU instances
(Render free tier ~0.1 vCPU) piles decodes onto the single worker, starves
the asyncio event loop, and trips uvicorn's WebSocket keepalive watchdog
(connection killed with 1011 mid-answer). Per-phrase finals keep captions
useful at a fraction of the compute.

Server -> client JSON events:
  {"type": "start"}                     a speech segment opened
  {"type": "final", "text": "..."}      completed segment text
  {"type": "done"}                      all finals delivered; safe to submit

Client -> server:
  binary frames   raw Int16 LE PCM @ 16 kHz mono
  {"type": "stop"} flush the open segment and close cleanly
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import SAMPLE_RATE

router = APIRouter()

SEGMENT_SILENCE_S = 0.7      # silence that closes an open segment
MAX_SEGMENT_S = 15.0         # hard cap so worst-case latency stays bounded
SPEECH_START_PROB = 0.5
# Slow CPUs can take several seconds to decode the trailing segment after a
# stop; give the final worker enough room before tearing the socket down.
STOP_GRACE_S = 12.0
VAD_WINDOW = 512             # silero expects multiples of 512 samples @16k

# One worker => transcriptions never interleave on CPU.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

_UNSET = object()        # sentinel: VAD not loaded yet
_vad = _UNSET
_vad_lock = threading.Lock()


def _get_vad():
    """Lazily load the bundled Silero ONNX model once; degrade to RMS if missing.

    Loaded on first use (not import) so boot memory stays flat on small
    instances that never open a streaming socket.
    """
    global _vad
    if _vad is not _UNSET:
        return _vad or None
    with _vad_lock:
        if _vad is not _UNSET:
            return _vad or None
        try:  # bundled Silero ONNX via onnxruntime; degrade to RMS if missing
            from audio.silero_vad import get_vad
            _vad = get_vad()
        except Exception:
            _vad = None
    return _vad


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
        vad = _get_vad()
        if vad is None:
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            speech = rms > max(self.noise_floor * 3.0, 0.012)
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * min(rms, self.noise_floor * 4.0)
            return speech
        buf = np.concatenate([self._tail, x])
        usable = (buf.size // VAD_WINDOW) * VAD_WINDOW
        self._tail = buf[usable:]
        if usable == 0:
            return False
        probs = np.asarray(vad(buf[:usable]))
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

    tracker = SegmentTracker()
    gate = VadGate()
    finals: asyncio.Queue = asyncio.Queue()

    async def _safe_send(payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            pass  # socket already gone — nothing to deliver

    async def emit_final(segment: np.ndarray):
        seconds = segment.shape[0] / SAMPLE_RATE
        t0 = time.monotonic()
        try:
            text = await loop.run_in_executor(_pool, _transcribe_pcm, segment)
        except Exception as e:
            print(f"[STT] decode failed ({seconds:.1f}s segment): {e!r}")
            return
        print(f"[STT] {seconds:.1f}s audio -> {time.monotonic() - t0:.1f}s decode")
        if text:
            await _safe_send({"type": "final", "text": text})

    async def final_worker():
        while True:
            item = await finals.get()
            if item is None:
                break
            await emit_final(item)
        await _safe_send({"type": "done"})  # client may now submit its transcript

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
                    await ws.send_json({"type": "start"})
                elif event == "close" and segment is not None and segment.size:
                    await finals.put(segment)
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
