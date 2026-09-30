"""ChromaDB wrapper: one Chroma collection per RAG collection."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict
    score: float = 0.0


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )

    @staticmethod
    def _name(collection_id: str) -> str:
        return f"rag_{collection_id}"

    def _collection(self, collection_id: str):
        # Embeddings are always supplied by us, so no Chroma embedding function is attached.
        return self._client.get_or_create_collection(
            self._name(collection_id), embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add(self, collection_id: str, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        clean = [{k: v for k, v in m.items() if v is not None} for m in metadatas]
        collection = self._collection(collection_id)
        for start in range(0, len(ids), 1000):
            end = start + 1000
            collection.add(
                ids=ids[start:end], documents=texts[start:end], embeddings=embeddings[start:end], metadatas=clean[start:end]
            )

    def query(self, collection_id: str, embedding: list[float], n_results: int) -> list[StoredChunk]:
        collection = self._collection(collection_id)
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            StoredChunk(id=i, text=t, metadata=m or {}, score=1.0 - d)
            for i, t, m, d in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def all_chunks(self, collection_id: str) -> list[StoredChunk]:
        result = self._collection(collection_id).get(include=["documents", "metadatas"])
        return [StoredChunk(id=i, text=t, metadata=m or {}) for i, t, m in zip(result["ids"], result["documents"], result["metadatas"])]

    def delete_document(self, collection_id: str, doc_id: str) -> None:
        self._collection(collection_id).delete(where={"doc_id": doc_id})

    def delete_collection(self, collection_id: str) -> None:
        try:
            self._client.delete_collection(self._name(collection_id))
        except Exception:  # Already gone (e.g. collection never received documents).
            pass
