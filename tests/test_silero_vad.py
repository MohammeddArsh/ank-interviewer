"""Tests for the bundled Silero VAD loader (real ONNX, no downloads)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio.silero_vad import get_vad


def test_returns_callable():
    vad = get_vad()
    assert callable(vad)


def test_silence_scores_low():
    vad = get_vad()
    probs = vad(np.zeros(512 * 4, dtype=np.float32))
    assert probs.shape[0] == 4
    assert float(np.max(probs)) < 0.5


def test_accepts_arbitrary_multiples_of_512():
    vad = get_vad()
    for n in (512, 512 * 3, 512 * 10):
        probs = vad(np.random.RandomState(0).randn(n).astype(np.float32) * 0.1)
        assert probs.shape[0] == n // 512
