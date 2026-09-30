"""Hybrid retrieval: vector + BM25 candidates, merged with Reciprocal Rank Fusion, then reranked."""

import math
import threading
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.retrieval.vector_store import StoredChunk


@dataclass
class RetrievedChunk:
    id: str
    text: str
    filename: str
    page: int | None
    heading: str | None
    score: float  # rerank relevance in [0, 1], or the fused score when reranking is off

    @classmethod
    def from_stored(cls, chunk: StoredChunk, score: float) -> "RetrievedChunk":
        m = chunk.metadata
        return cls(chunk.id, chunk.text, m.get("filename", "unknown"), m.get("page"), m.get("heading"), score)


def reciprocal_rank_fusion(ranked_lists: list[list[StoredChunk]], k: int = 60) -> list[tuple[StoredChunk, float]]:
    """Merge rankings without having to compare their (incompatible) raw scores."""
    scores: dict[str, float] = {}
    chunks: dict[str, StoredChunk] = {}
    for ranking in ranked_lists:
        for rank, chunk in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            chunks.setdefault(chunk.id, chunk)
    order = sorted(scores, key=scores.get, reverse=True)
    return [(chunks[i], scores[i]) for i in order]


class Reranker:
    """Local cross-encoder that scores (query, passage) pairs jointly — more precise than embeddings."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name, cache_dir=str(get_settings().models_dir))
        self._lock = threading.Lock()

    def score(self, query: str, texts: list[str]) -> list[float]:
        with self._lock:
            logits = list(self._model.rerank(query, texts))
        return [1.0 / (1.0 + math.exp(-x)) for x in logits]


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)


def hybrid_search(
    query: str,
    vector_hits: list[StoredChunk],
    keyword_hits: list[StoredChunk],
    top_k: int,
    reranker: Reranker | None = None,
    min_score: float = 0.0,
    min_keep: int = 3,
) -> list[RetrievedChunk]:
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
    if not fused:
        return []
    if reranker is None:
        return [RetrievedChunk.from_stored(c, s) for c, s in fused[:top_k]]

    candidates = [c for c, _ in fused]
    scores = reranker.score(query, [c.text for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    kept = [pair for i, pair in enumerate(ranked[:top_k]) if i < min_keep or pair[1] >= min_score]
    return [RetrievedChunk.from_stored(c, s) for c, s in kept]
