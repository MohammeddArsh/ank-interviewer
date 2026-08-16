import os
import subprocess
import uuid
import warnings

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
