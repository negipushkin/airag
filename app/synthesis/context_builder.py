"""Stage 6a - Context construction with full provenance.

Chunks are wrapped in XML with controlled attribute names; user-controlled
content never reaches tag keys, attribute values are escaped, and chunk
text is capped (TDD security 7.2).
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from app.config import get_settings
from app.models import RetrievedChunk


class ContextBuilder:
    def __init__(self) -> None:
        self._max_chars = get_settings().max_chunk_chars_in_context

    def build(self, chunks: list[RetrievedChunk]) -> str:
        parts = ["<context>"]
        for i, chunk in enumerate(chunks, start=1):
            score = f"{chunk.rerank_score:.2f}" if chunk.rerank_score is not None else "n/a"
            text = chunk.text[: self._max_chars]
            parts.append(
                f"  <chunk id={quoteattr(str(i))}"
                f" source={quoteattr(chunk.filename)}"
                f" section={quoteattr(chunk.section or 'unknown')}"
                f" page={quoteattr(str(chunk.page))}"
                f" rerank_score={quoteattr(score)}>\n"
                f"{escape(text)}\n"
                f"  </chunk>"
            )
        parts.append("</context>")
        return "\n".join(parts)
