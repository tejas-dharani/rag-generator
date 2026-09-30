import pytest

from app.service import NotFoundError
from tests.helpers import make_pdf


def _ingest(service, collection_id, filename, data):
    doc, is_new = service.register_upload(collection_id, filename, data)
    if is_new:
        service.ingest_document(doc.id)
    return service.metadata.get_document(doc.id), is_new


def test_ingest_indexes_chunks_with_metadata(service):
    collection = service.create_collection("Manuals")
    doc, _ = _ingest(service, collection.id, "guide.pdf", make_pdf(["Install the pump.", "Clean the filter monthly."]))

    assert doc.status == "ready" and doc.chunk_count == 2
    chunks = service.vectors.all_chunks(collection.id)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["filename"] == "guide.pdf" for c in chunks)
    assert service.get_collection(collection.id).chunk_count == 2


def test_duplicate_upload_is_skipped(service):
    collection = service.create_collection("Docs")
    first, new1 = _ingest(service, collection.id, "a.txt", b"same content")
    second, new2 = _ingest(service, collection.id, "copy.txt", b"same content")
    assert new1 and not new2 and first.id == second.id


def test_failed_parse_is_reported_on_document(service):
    collection = service.create_collection("Docs")
    doc, _ = _ingest(service, collection.id, "scan.pdf", make_pdf([""]))
    assert doc.status == "failed" and "No extractable text" in doc.error


def test_upload_validation(service, settings):
    collection = service.create_collection("Docs")
    with pytest.raises(ValueError, match="Unsupported"):
        service.register_upload(collection.id, "virus.exe", b"x")
    with pytest.raises(ValueError, match="empty"):
        service.register_upload(collection.id, "empty.txt", b"")
    with pytest.raises(ValueError, match="larger"):
        service.register_upload(collection.id, "big.txt", b"x" * (settings.max_upload_mb * 1024 * 1024 + 1))


def test_collections_are_isolated_and_deletable(service):
    hr = service.create_collection("HR")
    tech = service.create_collection("Tech")
    _ingest(service, hr.id, "leave.txt", b"Employees get 20 days of annual leave.")
    _ingest(service, tech.id, "api.txt", b"The API rate limit is 100 requests per minute.")

    assert all("leave" in c.text for c in service.vectors.all_chunks(hr.id))
    assert all("API" in c.text for c in service.vectors.all_chunks(tech.id))

    service.delete_collection(hr.id)
    with pytest.raises(NotFoundError):
        service.get_collection(hr.id)
    assert [c.name for c in service.list_collections()] == ["Tech"]


def test_delete_document_removes_chunks(service):
    collection = service.create_collection("Docs")
    doc, _ = _ingest(service, collection.id, "a.txt", b"Alpha content.")
    _ingest(service, collection.id, "b.txt", b"Beta content.")
    service.delete_document(collection.id, doc.id)
    assert [c.metadata["filename"] for c in service.vectors.all_chunks(collection.id)] == ["b.txt"]


def test_per_collection_settings(service):
    collection = service.create_collection("Tiny chunks", chunk_size=100, chunk_overlap=10, top_k=3)
    assert (collection.settings.chunk_size, collection.settings.top_k) == (100, 3)
    with pytest.raises(ValueError):
        service.create_collection("Bad", chunk_size=100, chunk_overlap=100)
