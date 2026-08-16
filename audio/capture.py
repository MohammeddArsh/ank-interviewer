import os
import uuid

import sounddevice as sd
from scipy.io.wavfile import write

from config import CHANNELS, RECORD_SECONDS, SAMPLE_RATE, TEMP_DIR


def record_audio() -> str:
    print(f"Listening for {RECORD_SECONDS} seconds... speak now!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )
    sd.wait()
    print("Got it.")

    tmp_path = os.path.join(TEMP_DIR, "rec_" + uuid.uuid4().hex + ".wav")
    write(tmp_path, SAMPLE_RATE, audio)
    return tmp_path


def cleanup(filepath: str):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass
