import os
from langchain_ollama import ChatOllama


def get_chat_llm(model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0,
    )