"""SQLite store for collections and documents (the chunks themselves live in Chroma)."""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CollectionSettings:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_provider: str
    embedding_model: str


@dataclass
class Collection:
    id: str
    name: str
    description: str
    created_at: str
    settings: CollectionSettings
    document_count: int = 0
    chunk_count: int = 0


@dataclass
class Document:
    id: str
    collection_id: str
    filename: str
    sha256: str
    size_bytes: int
    status: str  # processing | ready | failed
    created_at: str
    error: str | None = None
    chunk_count: int = 0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    settings TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (collection_id, sha256)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MetadataStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._tx() as cur:
            cur.execute("PRAGMA foreign_keys = ON")
            cur.executescript(_SCHEMA)
            # Documents left mid-ingestion by a crash/restart can never finish.
            cur.execute(
                "UPDATE documents SET status='failed', error='Interrupted by server restart' WHERE status='processing'"
            )

    @contextmanager
    def _tx(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ---- collections ----

    def create_collection(self, name: str, description: str, settings: CollectionSettings) -> Collection:
        collection = Collection(uuid.uuid4().hex[:12], name, description, _now(), settings)
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO collections (id, name, description, created_at, settings) VALUES (?, ?, ?, ?, ?)",
                (collection.id, name, description, collection.created_at, json.dumps(asdict(settings))),
            )
        return collection

    def list_collections(self) -> list[Collection]:
        with self._tx() as cur:
            rows = cur.execute(
                """SELECT c.*, COUNT(d.id) AS document_count, COALESCE(SUM(d.chunk_count), 0) AS chunk_count
                   FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
                   GROUP BY c.id ORDER BY c.created_at DESC"""
            ).fetchall()
        return [self._row_to_collection(r) for r in rows]

    def get_collection(self, collection_id: str) -> Collection | None:
        with self._tx() as cur:
            row = cur.execute(
                """SELECT c.*, COUNT(d.id) AS document_count, COALESCE(SUM(d.chunk_count), 0) AS chunk_count
                   FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
                   WHERE c.id = ? GROUP BY c.id""",
                (collection_id,),
            ).fetchone()
        return self._row_to_collection(row) if row else None

    def delete_collection(self, collection_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            cur.execute("DELETE FROM collections WHERE id = ?", (collection_id,))

    @staticmethod
    def _row_to_collection(row: sqlite3.Row) -> Collection:
        return Collection(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            settings=CollectionSettings(**json.loads(row["settings"])),
            document_count=row["document_count"],
            chunk_count=row["chunk_count"],
        )

    # ---- documents ----

    def find_document_by_hash(self, collection_id: str, sha256: str) -> Document | None:
        with self._tx() as cur:
            row = cur.execute(
                "SELECT * FROM documents WHERE collection_id = ? AND sha256 = ?", (collection_id, sha256)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def add_document(self, collection_id: str, filename: str, sha256: str, size_bytes: int) -> Document:
        doc = Document(uuid.uuid4().hex[:12], collection_id, filename, sha256, size_bytes, "processing", _now())
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO documents (id, collection_id, filename, sha256, size_bytes, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc.id, collection_id, filename, sha256, size_bytes, doc.status, doc.created_at),
            )
        return doc

    def update_document(self, doc_id: str, status: str, chunk_count: int = 0, error: str | None = None) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE documents SET status = ?, chunk_count = ?, error = ? WHERE id = ?",
                (status, chunk_count, error, doc_id),
            )

    def get_document(self, doc_id: str) -> Document | None:
        with self._tx() as cur:
            row = cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, collection_id: str) -> list[Document]:
        with self._tx() as cur:
            rows = cur.execute(
                "SELECT * FROM documents WHERE collection_id = ? ORDER BY created_at, filename", (collection_id,)
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            collection_id=row["collection_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            status=row["status"],
            created_at=row["created_at"],
            error=row["error"],
            chunk_count=row["chunk_count"],
        )
