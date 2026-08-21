import ollama

from config import LLM_MODEL_LOCAL


def complete(messages: list, temperature: float = 0.7, max_tokens: int = 512) -> tuple:
    """Returns (reply_text, token_usage_dict) — local mode has no token counts."""
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    if system:
        chat_messages.insert(0, {"role": "system", "content": system})

    response = ollama.chat(
        model=LLM_MODEL_LOCAL,
        messages=chat_messages,
        options={"temperature": temperature, "num_predict": max_tokens}
    )
    reply = response["message"]["content"].strip()
    # Ollama doesn't expose token counts the same way, return zeros
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    return reply, usage


def get_reply(messages: list) -> tuple:
    """Returns (reply_text, token_usage_dict)."""
    return complete(messages)
