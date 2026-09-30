"""Application settings, read from environment variables / `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["openai", "ollama", "anthropic"] = "openai"
    llm_model: str = "gpt-5.4-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    anthropic_api_key: str | None = None
    llm_max_output_tokens: int = 1500

    # Embeddings / reranking
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    # Chunks whose rerank relevance (0-1) falls below this are dropped as off-topic.
    rerank_min_score: float = 0.01

    # Retrieval / chunking defaults (overridable per collection)
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 5
    candidates: int = 20

    # App
    data_dir: Path = Path("./data")
    max_upload_mb: int = 25

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
