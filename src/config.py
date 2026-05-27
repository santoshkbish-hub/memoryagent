from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    db_path: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_chat_model: str
    deepseek_memory_model: str
    embedding_provider: str
    embedding_dimensions: int
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str


def load_settings(env_file: str | None = None) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    db_path = os.getenv("MEMORY_DB_PATH", "./memory_agent.sqlite3")
    return Settings(
        db_path=str(Path(db_path).expanduser()),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        deepseek_memory_model=os.getenv("DEEPSEEK_MEMORY_MODEL", os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "384")),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or None,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
