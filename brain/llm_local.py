import ollama

from config import LLM_MODEL_LOCAL


def get_reply(messages: list) -> tuple:
    """Returns (reply_text, token_usage_dict) — local mode has no token counts."""
    response = ollama.chat(
        model=LLM_MODEL_LOCAL,
        messages=messages,
        options={"temperature": 0.7, "num_predict": 300}
    )
    reply = response["message"]["content"].strip()
    # Ollama doesn't expose token counts the same way, return zeros
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    return reply, usage
