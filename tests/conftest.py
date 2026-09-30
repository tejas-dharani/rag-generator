import hashlib
import math

import pytest

from app.config import Settings
from app.retrieval.bm25 import tokenize
from app.service import RAGService


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, instant, and good enough to rank by word overlap."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def settings(tmp_path) -> Settings:
    # _env_file=None: tests never read the developer's real .env (or API keys).
    return Settings(_env_file=None, data_dir=tmp_path, rerank_enabled=False, openai_api_key=None)


@pytest.fixture
def service(settings) -> RAGService:
    return RAGService(settings, embedder_factory=lambda provider, model: FakeEmbedder())
