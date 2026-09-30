"""In-memory BM25 keyword index per collection, rebuilt lazily after changes."""

import re
import threading

from rank_bm25 import BM25Okapi

from app.retrieval.vector_store import StoredChunk

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i in is it its of on or that the this to was "
    "were what when where which who why will with do does did can you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[StoredChunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks]) if chunks else None

    def search(self, query: str, n_results: int) -> list[StoredChunk]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [
            StoredChunk(self.chunks[i].id, self.chunks[i].text, self.chunks[i].metadata, float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


class BM25Cache:
    """Holds one index per collection; `invalidate` after documents are added or removed."""

    def __init__(self):
        self._indexes: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def get(self, collection_id: str, loader) -> BM25Index:
        with self._lock:
            index = self._indexes.get(collection_id)
            if index is None:
                index = BM25Index(loader())
                self._indexes[collection_id] = index
            return index

    def invalidate(self, collection_id: str) -> None:
        with self._lock:
            self._indexes.pop(collection_id, None)
