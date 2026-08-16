"""
Conversational AI Assistant with Speech Interface
--------------------------------------------------
Run this file to start the assistant:  python main.py

Say "goodbye", "exit", or "quit" to stop.
"""

import os
import sys

# Make sure sibling folders are importable
sys.path.insert(0, os.path.dirname(__file__))

from audio.capture import cleanup, record_audio
from audio.stt import transcribe
from audio.tts import speak
from brain.llm import get_reply
from brain.memory import ConversationMemory

EXIT_PHRASES = {"goodbye", "exit", "quit", "bye", "stop"}


def main():
    print("\n🤖  AI Voice Assistant is ready!")
    print("💡  Speak after the prompt. Say 'goodbye' to quit.\n")
    print("─" * 50)

    memory = ConversationMemory()
    speak("Hello! I'm ready to chat. What's on your mind?")

    while True:
        try:
            # Step 1: Record audio
            audio_path = record_audio()

            # Step 2: Transcribe to text
            user_text = transcribe(audio_path)
            cleanup(audio_path)

            # Skip empty transcriptions (silence, background noise)
            if not user_text:
                print("⚠️   Didn't catch that, try again.")
                continue

            # Step 3: Check for exit command
            if any(phrase in user_text.lower() for phrase in EXIT_PHRASES):
                speak("Goodbye! Have a great day.")
                print("\n👋  Session ended.")
                break

            # Step 4: Add to memory and get LLM reply
            memory.add_user_message(user_text)
            memory.trim_if_needed(max_turns=20)

            reply = get_reply(memory.get_messages())

            # Step 5: Save reply to memory and speak it
            memory.add_assistant_message(reply)
            speak(reply)

            print("─" * 50)

        except KeyboardInterrupt:
            print("\n\n👋  Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌  Error: {e}")
            speak("Sorry, something went wrong. Let's try again.")
            continue


if __name__ == "__main__":
    main()
