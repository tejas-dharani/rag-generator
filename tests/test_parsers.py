import pytest

from app.ingestion.parsers import EmptyDocumentError, UnsupportedFileError, parse_file
from tests.helpers import make_docx, make_pdf


def test_pdf_keeps_page_numbers():
    data = make_pdf(["Alpha page one text", "Beta page two text"])
    sections = parse_file("report.pdf", data)
    assert [s.page for s in sections] == [1, 2]
    assert "Alpha" in sections[0].text
    assert "Beta" in sections[1].text


def test_pdf_without_text_raises():
    with pytest.raises(EmptyDocumentError, match="scanned"):
        parse_file("scan.pdf", make_pdf([""]))


def test_docx_tracks_headings():
    data = make_docx([("heading", "Leave Policy"), ("para", "Staff get 20 days."), ("heading", "Travel"), ("para", "Book economy.")])
    sections = parse_file("handbook.docx", data)
    assert [(s.heading, s.text) for s in sections] == [
        ("Leave Policy", "Staff get 20 days."),
        ("Travel", "Book economy."),
    ]


def test_markdown_sections_and_code_blocks():
    text = "# Intro\nHello there.\n\n## Setup\nRun it.\n```\n# not a heading\n```\n"
    sections = parse_file("readme.md", text.encode())
    assert [s.heading for s in sections] == ["Intro", "Setup"]
    assert "# not a heading" in sections[1].text


def test_html_strips_scripts_and_uses_headings():
    html = """<html><head><title>Guide</title><script>var x = 1;</script></head>
    <body><nav>Menu</nav><h1>Returns</h1><p>Return within <b>30 days</b>.</p>
    <ul><li>Keep the receipt</li></ul>
    <table><tr><th>Code</th><th>Meaning</th></tr><tr><td>E1</td><td>Low pressure</td></tr></table></body></html>"""
    sections = parse_file("page.html", html.encode())
    assert sections[0].heading == "Returns"
    assert "Return within 30 days." in sections[0].text
    assert "Keep the receipt" in sections[0].text
    assert "E1 | Low pressure" in sections[0].text
    assert "var x" not in sections[0].text and "Menu" not in sections[0].text


def test_txt_handles_non_utf8():
    sections = parse_file("notes.txt", "café menu".encode("latin-1"))
    assert "caf" in sections[0].text


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        parse_file("image.png", b"\x89PNG")
