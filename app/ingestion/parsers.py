"""Turn uploaded files into text sections with source metadata (page / heading)."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm"}


class UnsupportedFileError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


@dataclass
class Section:
    """A contiguous piece of a document that shares one page / heading."""

    text: str
    page: int | None = None
    heading: str | None = None


def parse_file(filename: str, data: bytes) -> list[Section]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        sections = _parse_pdf(data)
    elif ext == ".docx":
        sections = _parse_docx(data)
    elif ext in {".md", ".markdown"}:
        sections = _parse_markdown(_decode(data))
    elif ext in {".html", ".htm"}:
        sections = _parse_html(_decode(data))
    else:
        sections = [Section(text=_decode(data))]

    sections = [Section(_clean(s.text), s.page, s.heading) for s in sections]
    sections = [s for s in sections if s.text]
    if not sections:
        hint = " It may be a scanned PDF (OCR is not supported)." if ext == ".pdf" else ""
        raise EmptyDocumentError(f"No extractable text found in '{filename}'.{hint}")
    return sections


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(data: bytes) -> list[Section]:
    reader = PdfReader(io.BytesIO(data))
    return [
        Section(text=page.extract_text() or "", page=number)
        for number, page in enumerate(reader.pages, start=1)
    ]


def _parse_docx(data: bytes) -> list[Section]:
    doc = DocxDocument(io.BytesIO(data))
    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(Section(text="\n\n".join(buffer), heading=heading))
            buffer.clear()

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
        if style.startswith("heading") or style == "title":
            flush()
            heading = text
        else:
            buffer.append(text)
    flush()

    for table in doc.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        if rows:
            sections.append(Section(text="\n".join(rows), heading="Table"))
    return sections


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _parse_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []
    in_code = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _MD_HEADING.match(line)
        if match:
            if buffer:
                sections.append(Section(text="\n".join(buffer), heading=heading))
                buffer = []
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append(Section(text="\n".join(buffer), heading=heading))
    return sections


_HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_HTML_BLOCKS = _HTML_HEADINGS | {"p", "li", "pre", "blockquote", "td", "th", "dt", "dd", "figcaption"}


def _parse_html(html: str) -> list[Section]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg"]):
        tag.decompose()
    root = soup.body or soup

    sections: list[Section] = []
    heading: str | None = soup.title.get_text(strip=True) if soup.title else None
    buffer: list[str] = []

    for element in root.find_all(_HTML_BLOCKS):
        # Skip blocks nested inside another block we already collect (e.g. <p> inside <li>).
        if element.find_parent(_HTML_BLOCKS - _HTML_HEADINGS):
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in _HTML_HEADINGS:
            if buffer:
                sections.append(Section(text="\n\n".join(buffer), heading=heading))
                buffer = []
            heading = text
        else:
            buffer.append(text)
    if buffer:
        sections.append(Section(text="\n\n".join(buffer), heading=heading))

    if not sections:  # Pages without semantic blocks: fall back to all visible text.
        sections.append(Section(text=root.get_text("\n", strip=True), heading=heading))
    return sections
