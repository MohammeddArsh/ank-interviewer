from openai import OpenAI

from config import LLM_MODEL_OPENAI, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_reply(messages: list) -> tuple:
    """Returns (reply_text, token_usage_dict)"""
    response = client.chat.completions.create(
        model=LLM_MODEL_OPENAI,
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    reply = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    return reply, usage
