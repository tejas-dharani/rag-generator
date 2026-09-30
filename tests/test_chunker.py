import pytest

from app.ingestion.chunker import chunk_sections, estimate_tokens
from app.ingestion.parsers import Section


def _long_text(paragraphs: int, sentences: int = 6) -> str:
    return "\n\n".join(
        " ".join(f"Paragraph {p} sentence {s} talks about topic number {p * 10 + s}." for s in range(sentences))
        for p in range(paragraphs)
    )


def test_chunks_respect_size_limit():
    chunks = chunk_sections([Section(_long_text(40), page=3, heading="Results")], chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 200 + 5
        assert chunk.page == 3 and chunk.heading == "Results"
        assert chunk.text.startswith("Results\n")


def test_chunks_overlap():
    chunks = chunk_sections([Section(_long_text(20, sentences=2))], chunk_size=120, overlap=40)
    # With overlap, the last paragraph of one chunk reappears at the start of the next.
    last_para = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.split("\n\n")[0] == last_para


def test_giant_sentence_is_split():
    chunks = chunk_sections([Section("word " * 3000)], chunk_size=300, overlap=0)
    assert len(chunks) > 5
    assert all(estimate_tokens(c.text) <= 305 for c in chunks)


def test_chunks_do_not_cross_sections_and_indexes_are_sequential():
    chunks = chunk_sections([Section("First page.", page=1), Section("Second page.", page=2)])
    assert [(c.page, c.index) for c in chunks] == [(1, 0), (2, 1)]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_sections([Section("x")], chunk_size=100, overlap=100)
