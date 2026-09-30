import pytest

from app.config import Settings
from app.generation.llm import LLMError, create_llm
from app.generation.prompts import NOT_FOUND, build_messages, cited_numbers
from app.retrieval.hybrid import RetrievedChunk
from app.service import RAGService
from tests.conftest import FakeEmbedder


class FakeLLM:
    def __init__(self, reply="Employees get 20 days [1]."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.reply.split(" ")  # tokens without spaces is fine for these checks

    def generate(self, messages):
        self.calls.append(messages)
        return "How many leave days do part-time employees get?"


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def rag(settings, llm):
    service = RAGService(settings, embedder_factory=lambda p, m: FakeEmbedder(), llm_factory=lambda: llm)
    collection = service.create_collection("HR")
    doc, _ = service.register_upload(collection.id, "leave.txt", b"Employees get 20 days of annual leave.")
    service.ingest_document(doc.id)
    return service, collection


def test_cited_numbers_ignores_out_of_range_and_duplicates():
    assert cited_numbers("A [2]. B [1][2]. C [9].", source_count=3) == [2, 1]


def test_prompt_numbers_sources_and_includes_rules():
    chunk = RetrievedChunk("x", "Leave is 20 days.", "hr.pdf", 4, None, 0.9)
    messages = build_messages("How much leave?", [chunk])
    assert "ONLY" in messages[0]["content"] and NOT_FOUND in messages[0]["content"]
    assert '<source id="1" file="hr.pdf" page="4">' in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Question: How much leave?")


def test_answer_returns_sources_and_citations(rag, llm):
    service, collection = rag
    result = service.answer(collection.id, "How many days of annual leave?")
    assert result["sources"][0]["filename"] == "leave.txt"
    assert result["cited"] == [1] and result["grounded"] is True
    assert len(llm.calls) == 1


def test_answer_stream_event_order(rag):
    service, collection = rag
    types = [e["type"] for e in service.answer_stream(collection.id, "annual leave days?")]
    assert types[0] == "sources" and types[-1] == "done" and "token" in types


def test_no_relevant_chunks_refuses_without_calling_llm(settings, llm):
    service = RAGService(settings, embedder_factory=lambda p, m: FakeEmbedder(), llm_factory=lambda: llm)
    empty = service.create_collection("Empty")
    result = service.answer(empty.id, "Anything?")
    assert result["answer"] == NOT_FOUND and result["grounded"] is False
    assert llm.calls == []


def test_follow_up_is_condensed_before_retrieval(rag, llm):
    service, collection = rag
    history = [{"role": "user", "content": "How much leave do employees get?"}, {"role": "assistant", "content": "20 days [1]."}]
    result = service.answer(collection.id, "and part-time staff?", history)
    assert result["search_query"] == "How many leave days do part-time employees get?"
    assert len(llm.calls) == 2


def test_missing_api_key_gives_helpful_error():
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        create_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key=None))
