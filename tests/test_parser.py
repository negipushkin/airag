"""DocumentParser tests on TXT/MD (no binary fixtures needed)."""

import pytest

from app.ingestion.parser import DocumentParser, _sanitize_filename
from app.models import ElementType


@pytest.fixture
def parser():
    return DocumentParser()


def test_markdown_headings_and_paragraphs(parser):
    md = "# Title\n\nIntro paragraph.\n\n## Sub\n\n- item one\n- item two\n"
    doc = parser.parse(md.encode(), "notes.md")
    types = [e.type for e in doc.elements]
    assert types[0] == ElementType.HEADING
    assert ElementType.PARAGRAPH in types
    assert ElementType.LIST in types
    # section assignment
    para = next(e for e in doc.elements if e.type == ElementType.PARAGRAPH)
    assert para.section == "Title"


def test_all_caps_line_detected_as_heading(parser):
    txt = "SECTION ONE\n\nBody text here.\n"
    doc = parser.parse(txt.encode(), "policy.txt")
    assert doc.elements[0].type == ElementType.HEADING
    assert doc.elements[1].section == "SECTION ONE"


def test_code_fence_isolated(parser):
    md = "# Doc\n\n```\nprint('hi')\n```\n\nAfter.\n"
    doc = parser.parse(md.encode(), "x.md")
    code = [e for e in doc.elements if e.type == ElementType.CODE]
    assert len(code) == 1
    assert "print" in code[0].text


def test_unsupported_extension_raises(parser):
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse(b"data", "image.png")


def test_filename_sanitised():
    assert _sanitize_filename("../../etc/passwd") == "passwd"
    assert "<" not in _sanitize_filename("a<b>.txt")
