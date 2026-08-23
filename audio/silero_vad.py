"""Minimal Silero VAD (v6) loader.

Loads the bundled ONNX model directly through onnxruntime so the streaming
path never needs to import faster-whisper (which drags in ctranslate2 and
costs ~100 MB of RSS on small instances). Call protocol mirrors
faster-whisper's SileroVADModel; model file is MIT-licensed from
https://github.com/snakers4/silero-vad.
"""

from __future__ import annotations

import os

import numpy as np

_ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "silero_vad_v6.onnx")

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        _session = onnxruntime.InferenceSession(
            _ASSET_PATH,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
    return _session


def get_vad():
    """Return a callable: (audio float32, len % 512 == 0) -> probs array."""
    _get_session()
    return _run


def _run(audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64):
    assert audio.ndim == 1, "Input should be a 1D array"
    assert audio.shape[0] % num_samples == 0, "Input size should be a multiple of num_samples"

    session = _get_session()
    h = np.zeros((1, 1, 128), dtype="float32")
    c = np.zeros((1, 1, 128), dtype="float32")

    batched_audio = audio.reshape(-1, num_samples)
    context = batched_audio[..., -context_size_samples:]
    context[-1] = 0
    context = np.roll(context, 1, 0)
    batched_audio = np.concatenate([context, batched_audio], 1)

    output, _h, _c = session.run(
        None,
        {"input": batched_audio.astype(np.float32), "h": h, "c": c},
    )
    return output
