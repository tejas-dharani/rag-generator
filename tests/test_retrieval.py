from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from app.retrieval.vector_store import StoredChunk


def _chunk(i: str, text: str = "") -> StoredChunk:
    return StoredChunk(id=i, text=text or i, metadata={"filename": f"{i}.txt", "page": 1})


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What is the error code E-404 in v2.1?") == ["error", "code", "e-404", "v2.1"]


def test_bm25_finds_exact_terms():
    index = BM25Index([_chunk("a", "Invoice INV-2291 was paid"), _chunk("b", "General payment terms"), _chunk("c", "Shipping info")])
    hits = index.search("status of INV-2291", 5)
    assert hits[0].id == "a"
    assert "c" not in [h.id for h in hits]  # zero-score chunks are dropped


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]])
    assert [c.id for c, _ in fused] == ["b", "a", "c"]


class FakeReranker:
    def score(self, query, texts):
        return [0.9 if "refund" in t else 0.001 for t in texts]


def test_reranker_reorders_and_filters_irrelevant():
    vector = [_chunk("x", "shipping times"), _chunk("y", "refund within 30 days")]
    results = hybrid_search("refund?", vector, [], top_k=5, reranker=FakeReranker(), min_score=0.01, min_keep=0)
    assert [r.id for r in results] == ["y"]
    assert results[0].filename == "y.txt" and results[0].score == 0.9


def test_min_keep_always_passes_best_chunks_to_llm():
    vector = [_chunk("x", "shipping times"), _chunk("z", "opening hours"), _chunk("y", "refund within 30 days")]
    results = hybrid_search("refund?", vector, [], top_k=5, reranker=FakeReranker(), min_score=0.01, min_keep=2)
    assert [r.id for r in results][0] == "y" and len(results) == 2


def test_service_retrieve_end_to_end(service):
    collection = service.create_collection("Policies")
    for name, text in [
        ("leave.txt", "Annual leave: employees receive 20 days of paid leave per year."),
        ("travel.txt", "Travel: book economy class for flights under 6 hours."),
        ("security.txt", "Security: rotate your password every 90 days."),
    ]:
        doc, _ = service.register_upload(collection.id, name, text.encode())
        service.ingest_document(doc.id)

    results = service.retrieve(collection.id, "How many days of paid leave do employees get?", top_k=2)
    assert results[0].filename == "leave.txt"
    assert len(results) == 2
