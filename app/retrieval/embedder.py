"""Embedding providers. Collections remember which one built their index."""

import threading
from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings (default: BAAI/bge-small-en-v1.5). No API key, runs on CPU."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=str(get_settings().models_dir))
        self._lock = threading.Lock()  # ONNX sessions are shared; keep batches serialized.

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return [v.tolist() for v in self._model.passage_embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the model's query instruction prefix where it has one.
        with self._lock:
            return next(iter(self._model.query_embed(text))).tolist()


class OpenAIEmbedder:
    def __init__(self, model_name: str):
        from openai import OpenAI

        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self._model = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 256):
            response = self._client.embeddings.create(model=self._model, input=texts[start : start + 256])
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=4)
def get_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "fastembed":
        return FastEmbedEmbedder(model_name)
    if provider == "openai":
        return OpenAIEmbedder(model_name)
    raise ValueError(f"Unknown embedding provider: {provider}")
