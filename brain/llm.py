from config import MODE

if MODE == "local":
    from brain.llm_local import complete, get_reply

    def get_last_model():
        return None

else:
    from brain.llm_openrouter import complete, get_last_model, get_reply

__all__ = ["get_reply", "complete", "get_last_model"]
