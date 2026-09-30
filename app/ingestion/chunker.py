"""Structure-aware chunking: split on paragraphs, then sentences, then words."""

import re
from dataclasses import dataclass

from app.ingestion.parsers import Section


@dataclass
class Chunk:
    text: str
    page: int | None
    heading: str | None
    index: int


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for WordPiece/BPE tokenizers (no tokenizer download needed).

    English averages ~1.3 tokens per word and ~4 characters per token; taking the larger
    of the two keeps chunks under the embedder's 512-token limit for numbers, code and URLs.
    """
    return max(round(len(text.split()) * 1.3), len(text) // 4)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split_units(text: str, max_tokens: int) -> list[str]:
    """Break text into pieces no larger than max_tokens, preferring natural boundaries."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            if estimate_tokens(sentence) <= max_tokens:
                units.append(sentence)
                continue
            words = sentence.split()
            step = max(1, int(max_tokens / 1.3))
            units.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return units


def chunk_sections(sections: list[Section], chunk_size: int = 400, overlap: int = 60) -> list[Chunk]:
    """Pack units greedily into chunks of ~chunk_size tokens, repeating ~overlap tokens.

    Chunks never span two sections, so every chunk has one page and one heading.
    The heading is prefixed to the chunk text so retrieval can match on it.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for section in sections:
        prefix = f"{section.heading}\n" if section.heading else ""
        budget = max(50, chunk_size - estimate_tokens(prefix))
        current: list[str] = []
        current_tokens = 0

        def emit() -> None:
            body = "\n\n".join(current).strip()
            if body:
                chunks.append(Chunk(prefix + body, section.page, section.heading, len(chunks)))

        for unit in _split_units(section.text, budget):
            unit_tokens = estimate_tokens(unit)
            if current and current_tokens + unit_tokens > budget:
                emit()
                # Carry trailing units into the next chunk as overlap.
                carried: list[str] = []
                carried_tokens = 0
                for previous in reversed(current):
                    t = estimate_tokens(previous)
                    if carried_tokens + t > overlap or carried_tokens + t + unit_tokens > budget:
                        break
                    carried.insert(0, previous)
                    carried_tokens += t
                current, current_tokens = carried, carried_tokens
            current.append(unit)
            current_tokens += unit_tokens
        emit()
    return chunks
