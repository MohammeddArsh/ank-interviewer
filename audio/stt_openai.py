from openai import OpenAI

from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe(audio_filepath: str) -> str:
    with open(audio_filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    text = result.text.strip()
    print(f"You said: \"{text}\"")
    return text
