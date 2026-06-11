"""Stage 2 - Semantic chunking.

Heading-aware chunk segmentation: groups elements under their nearest
heading, splits on element boundaries, and falls back to token-window
splitting only when a single element exceeds the max chunk size.
Implements the TDD chunking algorithm verbatim.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import tiktoken

from app.models import Chunk, DocumentElement, ElementType, ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")  # tokenizer used by text-embedding-3-small


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class SemanticChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: ParsedDocument, doc_type: str = "document") -> list[Chunk]:
        uploaded_at = datetime.now(timezone.utc).isoformat()
        chunks: list[Chunk] = []

        section = ""
        page = 0
        buf_texts: list[str] = []
        buf_types: list[str] = []
        buf_tokens = 0

        def flush(carry_overlap: bool) -> str:
            """Emit the current buffer as a chunk; return overlap carry text."""
            nonlocal buf_texts, buf_types, buf_tokens
            text = "\n".join(buf_texts).strip()
            carry = ""
            if text:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc.id,
                    filename=doc.filename,
                    chunk_index=len(chunks),
                    text=text,
                    token_count=count_tokens(text),
                    section=section,
                    page=page,
                    element_types=sorted(set(buf_types)),
                    doc_type=doc_type,
                    uploaded_at=uploaded_at,
                ))
                if carry_overlap:
                    tokens = _ENCODING.encode(text)
                    carry = _ENCODING.decode(tokens[-self.overlap_tokens:])
            buf_texts, buf_types, buf_tokens = [], [], 0
            return carry

        for el in doc.elements:
            if el.type == ElementType.HEADING:
                # 2a. Flush current chunk; start a new one under this heading.
                flush(carry_overlap=False)
                section = el.text
                page = el.page or page
                buf_texts.append(el.text)
                buf_types.append(el.type.value)
                buf_tokens = count_tokens(el.text)
                continue

            page = el.page or page
            el_tokens = count_tokens(el.text)

            # 2b. Would appending overflow MAX_TOKENS? Flush w/ overlap carry.
            if buf_tokens and buf_tokens + el_tokens > self.max_tokens:
                carry = flush(carry_overlap=True)
                if carry:
                    buf_texts.append(carry)
                    buf_types.append("overlap")
                    buf_tokens = count_tokens(carry)

            # Oversized single element: hard-split on token windows.
            if el_tokens > self.max_tokens:
                for piece in self._split_tokens(el.text):
                    if buf_tokens and buf_tokens + count_tokens(piece) > self.max_tokens:
                        carry = flush(carry_overlap=True)
                        if carry:
                            buf_texts.append(carry)
                            buf_types.append("overlap")
                            buf_tokens = count_tokens(carry)
                    buf_texts.append(piece)
                    buf_types.append(el.type.value)
                    buf_tokens += count_tokens(piece)
                continue

            # 2c. Append element to the current chunk.
            buf_texts.append(el.text)
            buf_types.append(el.type.value)
            buf_tokens += el_tokens

        flush(carry_overlap=False)  # 3. Flush final chunk.
        return chunks

    def _split_tokens(self, text: str) -> list[str]:
        """Split an oversized element into windows of max_tokens with overlap."""
        tokens = _ENCODING.encode(text)
        step = self.max_tokens - self.overlap_tokens
        pieces = []
        for start in range(0, len(tokens), step):
            window = tokens[start:start + self.max_tokens]
            pieces.append(_ENCODING.decode(window))
            if start + self.max_tokens >= len(tokens):
                break
        return pieces
