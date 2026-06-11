"""Stage 1 - Semantic document parsing.

Structure-aware extraction: preserves logical document units (headings,
paragraphs, tables, lists) instead of raw character streams.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from app.models import DocumentElement, ElementType, ParsedDocument

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 \-:&/,.()]{3,80}$")
_LIST_LINE = re.compile(r"^\s*([-*+•]|\d+[.)])\s+")


class DocumentParser:
    """Dispatches to a format-specific parser by file extension."""

    SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".markdown"}

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            elements = self._parse_pdf(data)
        elif ext == ".docx":
            elements = self._parse_docx(data)
        elif ext in (".txt", ".md", ".markdown"):
            elements = self._parse_text(data.decode("utf-8", errors="replace"))
        else:
            raise ValueError(f"Unsupported file type: {ext or filename!r}")

        self._assign_sections(elements)
        return ParsedDocument(filename=_sanitize_filename(filename), elements=elements)

    # ------------------------------------------------------------- PDF

    def _parse_pdf(self, data: bytes) -> list[DocumentElement]:
        import fitz  # PyMuPDF

        elements: list[DocumentElement] = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            # Median font size across the doc -> heading threshold
            sizes: list[float] = []
            pages_blocks = []
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                pages_blocks.append(blocks)
                for b in blocks:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if span["text"].strip():
                                sizes.append(span["size"])
            body_size = _median(sizes) if sizes else 11.0

            for page_no, blocks in enumerate(pages_blocks, start=1):
                for b in blocks:
                    lines = b.get("lines", [])
                    if not lines:
                        continue
                    text_parts: list[str] = []
                    max_size = 0.0
                    bold = False
                    for line in lines:
                        spans = line.get("spans", [])
                        line_text = "".join(s["text"] for s in spans)
                        if line_text.strip():
                            text_parts.append(line_text.strip())
                        for s in spans:
                            max_size = max(max_size, s["size"])
                            if s.get("flags", 0) & 2**4:  # bold flag
                                bold = True
                    text = " ".join(text_parts).strip()
                    if not text:
                        continue

                    is_heading = (
                        len(text) <= 120
                        and "\n" not in text
                        and (max_size >= body_size * 1.15 or (bold and max_size >= body_size))
                        and not text.endswith((".", ",", ";", ":"))
                    )
                    if is_heading:
                        level = 1 if max_size >= body_size * 1.5 else 2
                        elements.append(DocumentElement(
                            type=ElementType.HEADING, level=level, text=text,
                            page=page_no, metadata={"font_size": round(max_size, 1)},
                        ))
                    elif _LIST_LINE.match(text):
                        elements.append(DocumentElement(
                            type=ElementType.LIST, text=text, page=page_no))
                    else:
                        elements.append(DocumentElement(
                            type=ElementType.PARAGRAPH, text=text, page=page_no))

        # Tables via pdfplumber (appended where found, by page)
        try:
            elements = self._merge_pdf_tables(data, elements)
        except Exception:
            pass  # tables are best-effort; text extraction already succeeded
        return elements

    def _merge_pdf_tables(
        self, data: bytes, elements: list[DocumentElement]
    ) -> list[DocumentElement]:
        import pdfplumber

        tables_by_page: dict[int, list[str]] = {}
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    rows = [
                        " | ".join((cell or "").strip() for cell in row)
                        for row in table if any(cell for cell in row)
                    ]
                    if rows:
                        tables_by_page.setdefault(page_no, []).append("\n".join(rows))

        if not tables_by_page:
            return elements

        merged: list[DocumentElement] = []
        emitted: set[int] = set()
        for el in elements:
            merged.append(el)
            if el.page in tables_by_page and el.page not in emitted:
                # Emit page tables after the first element of that page's content
                continue
        # Simpler: append each page's tables after the last element of that page
        result: list[DocumentElement] = []
        for i, el in enumerate(elements):
            result.append(el)
            is_last_of_page = i + 1 == len(elements) or elements[i + 1].page != el.page
            if is_last_of_page and el.page in tables_by_page and el.page not in emitted:
                emitted.add(el.page)
                for t in tables_by_page[el.page]:
                    result.append(DocumentElement(
                        type=ElementType.TABLE, text=t, page=el.page))
        return result

    # ------------------------------------------------------------- DOCX

    def _parse_docx(self, data: bytes) -> list[DocumentElement]:
        import docx

        document = docx.Document(io.BytesIO(data))
        elements: list[DocumentElement] = []

        for block in _iter_docx_blocks(document):
            if block["kind"] == "paragraph":
                para = block["item"]
                text = para.text.strip()
                if not text:
                    continue
                style = (para.style.name or "") if para.style else ""
                m = re.match(r"Heading (\d+)", style)
                if m:
                    elements.append(DocumentElement(
                        type=ElementType.HEADING, level=int(m.group(1)),
                        text=text, metadata={"style": style}))
                elif style == "Title":
                    elements.append(DocumentElement(
                        type=ElementType.HEADING, level=1, text=text,
                        metadata={"style": style}))
                elif "List" in style or _LIST_LINE.match(text):
                    elements.append(DocumentElement(
                        type=ElementType.LIST, text=text, metadata={"style": style}))
                else:
                    elements.append(DocumentElement(
                        type=ElementType.PARAGRAPH, text=text, metadata={"style": style}))
            else:  # table
                table = block["item"]
                rows = []
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    elements.append(DocumentElement(
                        type=ElementType.TABLE, text="\n".join(rows)))
        return elements

    # ------------------------------------------------------------- TXT / MD

    def _parse_text(self, text: str) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        in_code = False
        code_lines: list[str] = []
        para_lines: list[str] = []

        def flush_para() -> None:
            if para_lines:
                block = "\n".join(para_lines).strip()
                if block:
                    etype = ElementType.LIST if _LIST_LINE.match(para_lines[0]) \
                        else ElementType.PARAGRAPH
                    elements.append(DocumentElement(type=etype, text=block))
                para_lines.clear()

        for line in text.splitlines():
            if line.strip().startswith("```"):
                if in_code:
                    elements.append(DocumentElement(
                        type=ElementType.CODE, text="\n".join(code_lines)))
                    code_lines.clear()
                    in_code = False
                else:
                    flush_para()
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue

            m = _MD_HEADING.match(line)
            if m:
                flush_para()
                elements.append(DocumentElement(
                    type=ElementType.HEADING, level=len(m.group(1)),
                    text=m.group(2).strip()))
            elif not line.strip():
                flush_para()
            elif _ALL_CAPS_HEADING.match(line.strip()) and len(line.strip().split()) <= 12:
                flush_para()
                elements.append(DocumentElement(
                    type=ElementType.HEADING, level=1, text=line.strip()))
            else:
                para_lines.append(line)

        if in_code and code_lines:
            elements.append(DocumentElement(
                type=ElementType.CODE, text="\n".join(code_lines)))
        flush_para()
        return elements

    # ------------------------------------------------------------- shared

    @staticmethod
    def _assign_sections(elements: list[DocumentElement]) -> None:
        """Set each element's `section` to its nearest ancestor heading."""
        current = ""
        for el in elements:
            if el.type == ElementType.HEADING:
                current = el.text
                el.section = el.text
            else:
                el.section = current


def _iter_docx_blocks(document):
    """Yield paragraphs and tables in true document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = document.element.body
    for child in parent.iterchildren():
        if child.tag.endswith("}p"):
            yield {"kind": "paragraph", "item": Paragraph(child, document)}
        elif child.tag.endswith("}tbl"):
            yield {"kind": "table", "item": Table(child, document)}


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _sanitize_filename(filename: str) -> str:
    """Strip path components and special characters (TDD security 7.2)."""
    name = Path(filename).name
    return re.sub(r"[^\w.\- ()\[\]]", "_", name)
