"""RAG service: collections, ingestion and question answering, independent of the web layer."""

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

from app.config import Settings
from app.generation.llm import LLMProvider, get_llm
from app.generation.prompts import NOT_FOUND, build_condense_messages, build_messages, cited_numbers
from app.ingestion.chunker import chunk_sections
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_file
from app.retrieval.bm25 import BM25Cache
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.hybrid import RetrievedChunk, get_reranker, hybrid_search
from app.retrieval.vector_store import VectorStore
from app.storage.metadata import Collection, CollectionSettings, Document, MetadataStore

logger = logging.getLogger(__name__)


class NotFoundError(LookupError):
    pass


class RAGService:
    def __init__(self, settings: Settings, embedder_factory=get_embedder, llm_factory=get_llm):
        self.settings = settings
        self.metadata = MetadataStore(settings.db_path)
        self.vectors = VectorStore(settings.chroma_dir)
        self.bm25 = BM25Cache()
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory

    @property
    def llm(self) -> LLMProvider:
        return self._llm_factory()

    # ---- collections ----

    def create_collection(self, name: str, description: str = "", **overrides) -> Collection:
        s = self.settings
        settings = CollectionSettings(
            chunk_size=overrides.get("chunk_size") or s.chunk_size,
            chunk_overlap=overrides.get("chunk_overlap") if overrides.get("chunk_overlap") is not None else s.chunk_overlap,
            top_k=overrides.get("top_k") or s.top_k,
            embedding_provider=s.embedding_provider,
            embedding_model=s.embedding_model,
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self.metadata.create_collection(name.strip(), description.strip(), settings)

    def get_collection(self, collection_id: str) -> Collection:
        collection = self.metadata.get_collection(collection_id)
        if collection is None:
            raise NotFoundError(f"Collection '{collection_id}' not found")
        return collection

    def list_collections(self) -> list[Collection]:
        return self.metadata.list_collections()

    def delete_collection(self, collection_id: str) -> None:
        self.get_collection(collection_id)
        self.vectors.delete_collection(collection_id)
        self.bm25.invalidate(collection_id)
        self.metadata.delete_collection(collection_id)
        for path in self._upload_dir(collection_id).glob("*"):
            path.unlink(missing_ok=True)

    def embedder_for(self, collection: Collection) -> Embedder:
        return self._embedder_factory(collection.settings.embedding_provider, collection.settings.embedding_model)

    # ---- documents ----

    def register_upload(self, collection_id: str, filename: str, data: bytes) -> tuple[Document, bool]:
        """Validate and save an upload. Returns (document, is_new); duplicates are not re-ingested."""
        self.get_collection(collection_id)
        filename = Path(filename).name or "upload"
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {filename}")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"{filename} is larger than {self.settings.max_upload_mb} MB")
        if not data:
            raise ValueError(f"{filename} is empty")

        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.metadata.find_document_by_hash(collection_id, sha256)
        if existing is not None:
            return existing, False

        doc = self.metadata.add_document(collection_id, filename, sha256, len(data))
        path = self._upload_path(collection_id, doc.id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return doc, True

    def ingest_document(self, doc_id: str) -> None:
        """Parse, chunk, embed and index one registered document. Safe to run in a background task."""
        doc = self.metadata.get_document(doc_id)
        if doc is None:
            return
        try:
            collection = self.get_collection(doc.collection_id)
            data = self._upload_path(doc.collection_id, doc.id, doc.filename).read_bytes()
            sections = parse_file(doc.filename, data)
            chunks = chunk_sections(sections, collection.settings.chunk_size, collection.settings.chunk_overlap)
            embeddings = self.embedder_for(collection).embed_documents([c.text for c in chunks])
            self.vectors.add(
                collection.id,
                ids=[f"{doc.id}:{c.index}" for c in chunks],
                texts=[c.text for c in chunks],
                embeddings=embeddings,
                metadatas=[
                    {"doc_id": doc.id, "filename": doc.filename, "page": c.page, "heading": c.heading, "chunk_index": c.index}
                    for c in chunks
                ],
            )
            self.bm25.invalidate(collection.id)
            self.metadata.update_document(doc.id, "ready", chunk_count=len(chunks))
            logger.info("Indexed %s: %d chunks", doc.filename, len(chunks))
        except Exception as exc:  # Surface any failure to the user as the document's status.
            logger.exception("Failed to ingest %s", doc.filename)
            self.metadata.update_document(doc.id, "failed", error=str(exc) or type(exc).__name__)

    def list_documents(self, collection_id: str) -> list[Document]:
        self.get_collection(collection_id)
        return self.metadata.list_documents(collection_id)

    def delete_document(self, collection_id: str, doc_id: str) -> None:
        doc = self.metadata.get_document(doc_id)
        if doc is None or doc.collection_id != collection_id:
            raise NotFoundError(f"Document '{doc_id}' not found")
        self.vectors.delete_document(collection_id, doc_id)
        self.bm25.invalidate(collection_id)
        self.metadata.delete_document(doc_id)
        self._upload_path(collection_id, doc.id, doc.filename).unlink(missing_ok=True)

    # ---- retrieval ----

    def retrieve(self, collection_id: str, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        collection = self.get_collection(collection_id)
        n = self.settings.candidates
        vector_hits = self.vectors.query(collection.id, self.embedder_for(collection).embed_query(query), n)
        bm25 = self.bm25.get(collection.id, lambda: self.vectors.all_chunks(collection.id))
        keyword_hits = bm25.search(query, n)
        reranker = get_reranker(self.settings.reranker_model) if self.settings.rerank_enabled else None
        return hybrid_search(
            query,
            vector_hits,
            keyword_hits,
            top_k=top_k or collection.settings.top_k,
            reranker=reranker,
            min_score=self.settings.rerank_min_score,
        )

    # ---- question answering ----

    def answer_stream(self, collection_id: str, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Yield events: `sources` (retrieved chunks), `token` (answer text) and `done` (final answer + citations)."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        self.get_collection(collection_id)

        search_query = question
        if history:
            # Follow-ups like "what about part-time staff?" need the conversation to be searchable.
            search_query = self.llm.generate(build_condense_messages(question, history)) or question

        chunks = self.retrieve(collection_id, search_query)
        sources = [
            {"number": n, "filename": c.filename, "page": c.page, "heading": c.heading, "text": c.text, "score": round(c.score, 4)}
            for n, c in enumerate(chunks, start=1)
        ]
        yield {"type": "sources", "sources": sources, "search_query": search_query}

        if not chunks:
            # Nothing relevant was retrieved: refuse without spending an LLM call.
            yield {"type": "token", "text": NOT_FOUND}
            yield {"type": "done", "answer": NOT_FOUND, "cited": [], "grounded": False}
            return

        parts: list[str] = []
        for token in self.llm.stream(build_messages(question, chunks, history)):
            parts.append(token)
            yield {"type": "token", "text": token}
        answer = "".join(parts).strip()
        cited = cited_numbers(answer, len(chunks))
        yield {"type": "done", "answer": answer, "cited": cited, "grounded": bool(cited) and NOT_FOUND not in answer}

    def answer(self, collection_id: str, question: str, history: list[dict] | None = None) -> dict:
        result: dict = {}
        for event in self.answer_stream(collection_id, question, history):
            if event["type"] == "sources":
                result["sources"] = event["sources"]
                result["search_query"] = event["search_query"]
            elif event["type"] == "done":
                result.update(answer=event["answer"], cited=event["cited"], grounded=event["grounded"])
        return result

    def _upload_dir(self, collection_id: str) -> Path:
        return self.settings.uploads_dir / collection_id

    def _upload_path(self, collection_id: str, doc_id: str, filename: str) -> Path:
        return self._upload_dir(collection_id) / f"{doc_id}{Path(filename).suffix.lower()}"
