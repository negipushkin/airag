"""SemanticChunker unit tests (TDD 10.1: heading flush, paragraph overflow,
overlap carry-over, empty doc, table element isolation)."""

from app.ingestion.chunker import SemanticChunker, count_tokens
from app.models import DocumentElement, ElementType, ParsedDocument


def make_doc(elements: list[DocumentElement]) -> ParsedDocument:
    return ParsedDocument(filename="test.txt", elements=elements)


def test_empty_doc_produces_no_chunks():
    chunker = SemanticChunker(max_tokens=512, overlap_tokens=64)
    assert chunker.chunk(make_doc([])) == []


def test_heading_flushes_current_chunk():
    chunker = SemanticChunker(max_tokens=512, overlap_tokens=64)
    doc = make_doc([
        DocumentElement(type=ElementType.HEADING, level=1, text="Section A"),
        DocumentElement(type=ElementType.PARAGRAPH, text="Alpha content."),
        DocumentElement(type=ElementType.HEADING, level=1, text="Section B"),
        DocumentElement(type=ElementType.PARAGRAPH, text="Beta content."),
    ])
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert "Section A" in chunks[0].text and "Alpha" in chunks[0].text
    assert "Section B" in chunks[1].text and "Beta" in chunks[1].text
    assert chunks[0].section == "Section A"
    assert chunks[1].section == "Section B"


def test_paragraph_overflow_splits_with_overlap_carry():
    chunker = SemanticChunker(max_tokens=50, overlap_tokens=10)
    para = "word " * 30  # ~30 tokens each
    doc = make_doc([
        DocumentElement(type=ElementType.PARAGRAPH, text=para.strip()),
        DocumentElement(type=ElementType.PARAGRAPH, text=para.strip()),
    ])
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    # Second chunk starts with overlap carried from the first
    assert "overlap" in chunks[1].element_types
    assert all(c.token_count <= 50 + 10 for c in chunks)


def test_oversized_single_element_is_token_split():
    chunker = SemanticChunker(max_tokens=40, overlap_tokens=8)
    huge = "alpha beta gamma delta " * 40  # far beyond 40 tokens
    doc = make_doc([DocumentElement(type=ElementType.PARAGRAPH, text=huge.strip())])
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    # Reconstruction: every chunk holds a non-empty window
    assert all(c.text.strip() for c in chunks)


def test_table_element_kept_intact_when_it_fits():
    chunker = SemanticChunker(max_tokens=512, overlap_tokens=64)
    table = "a | b | c\n1 | 2 | 3"
    doc = make_doc([
        DocumentElement(type=ElementType.HEADING, level=1, text="Data"),
        DocumentElement(type=ElementType.TABLE, text=table),
    ])
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert table in chunks[0].text
    assert "table" in chunks[0].element_types


def test_chunk_indices_are_sequential():
    chunker = SemanticChunker(max_tokens=30, overlap_tokens=5)
    doc = make_doc([
        DocumentElement(type=ElementType.PARAGRAPH, text="word " * 20)
        for _ in range(5)
    ])
    chunks = chunker.chunk(doc)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0
