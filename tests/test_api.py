import json

import pytest
from fastapi.testclient import TestClient

from app.generation.llm import LLMError
from app.main import create_app
from app.service import RAGService
from tests.conftest import FakeEmbedder
from tests.helpers import make_pdf


class FakeLLM:
    def stream(self, messages):
        yield "The warranty lasts "
        yield "2 years [1]."

    def generate(self, messages):
        return "standalone question"


class FailingLLM:
    def stream(self, messages):
        raise LLMError("LLM request failed: invalid key")
        yield  # pragma: no cover

    generate = stream


def _client(settings, llm) -> TestClient:
    service = RAGService(settings, embedder_factory=lambda p, m: FakeEmbedder(), llm_factory=lambda: llm)
    return TestClient(create_app(service, warm_up=False))


@pytest.fixture
def client(settings):
    with _client(settings, FakeLLM()) as c:
        yield c


def _create_with_doc(client) -> str:
    cid = client.post("/api/collections", json={"name": "Product Manual"}).json()["id"]
    files = [("files", ("manual.pdf", make_pdf(["Warranty: the warranty lasts 2 years."]), "application/pdf"))]
    response = client.post(f"/api/collections/{cid}/documents", files=files)
    assert response.status_code == 202
    return cid


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_full_flow_create_upload_query(client):
    cid = _create_with_doc(client)

    docs = client.get(f"/api/collections/{cid}/documents").json()
    # TestClient runs background tasks before returning, so ingestion is already done.
    assert docs[0]["status"] == "ready" and docs[0]["chunk_count"] == 1

    result = client.post(f"/api/collections/{cid}/query", json={"question": "How long is the warranty?"}).json()
    assert result["answer"] == "The warranty lasts 2 years [1]."
    assert result["cited"] == [1]
    assert result["sources"][0]["filename"] == "manual.pdf" and result["sources"][0]["page"] == 1

    collection = client.get(f"/api/collections/{cid}").json()
    assert collection["document_count"] == 1 and collection["chunk_count"] == 1


def test_streaming_query_uses_sse(client):
    cid = _create_with_doc(client)
    with client.stream("POST", f"/api/collections/{cid}/query", json={"question": "warranty?", "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[-1]["cited"] == [1]


def test_upload_rejects_bad_files_but_keeps_good_ones(client):
    cid = client.post("/api/collections", json={"name": "Mixed"}).json()["id"]
    files = [("files", ("notes.txt", b"hello world", "text/plain")), ("files", ("malware.exe", b"MZ", "application/octet-stream"))]
    results = client.post(f"/api/collections/{cid}/documents", files=files).json()["documents"]
    assert results[0]["status"] == "processing"
    assert results[1]["status"] == "rejected" and "Unsupported" in results[1]["error"]


def test_not_found_and_validation_errors(client):
    assert client.get("/api/collections/missing").status_code == 404
    assert client.post("/api/collections/missing/query", json={"question": "hi"}).status_code == 404
    assert client.post("/api/collections", json={"name": ""}).status_code == 422
    assert client.post("/api/collections", json={"name": "x", "chunk_size": 200, "chunk_overlap": 200}).status_code == 422


def test_delete_document_and_collection(client):
    cid = _create_with_doc(client)
    doc_id = client.get(f"/api/collections/{cid}/documents").json()[0]["id"]
    assert client.delete(f"/api/collections/{cid}/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/collections/{cid}/documents").json() == []
    assert client.delete(f"/api/collections/{cid}").status_code == 204
    assert client.get("/api/collections").json() == []


def test_llm_failure_is_reported(settings):
    with _client(settings, FailingLLM()) as client:
        cid = _create_with_doc(client)
        response = client.post(f"/api/collections/{cid}/query", json={"question": "warranty?"})
        assert response.status_code == 502 and "invalid key" in response.json()["detail"]

        with client.stream("POST", f"/api/collections/{cid}/query", json={"question": "warranty?", "stream": True}) as r:
            events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
        assert events[-1]["type"] == "error"
