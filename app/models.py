"""Core domain and API models (TDD Stage 1-6 data shapes)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- ingestion

class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"


class DocumentElement(BaseModel):
    """One logical unit produced by the structure-aware parser (TDD Stage 1)."""

    type: ElementType
    level: int = 0          # heading depth; 0 for non-headings
    text: str
    page: int = 0           # source page number (0 for non-paginated formats)
    section: str = ""       # nearest ancestor heading text
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    elements: list[DocumentElement] = Field(default_factory=list)


class Chunk(BaseModel):
    """Semantic chunk with provenance metadata (TDD Stage 2)."""

    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    token_count: int
    section: str = ""
    page: int = 0
    element_types: list[str] = Field(default_factory=list)
    doc_type: str = "document"
    uploaded_at: str = ""


# ---------------------------------------------------------------- retrieval

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    section: str = ""
    page: int = 0
    chunk_index: int = 0
    text: str
    retrieval_score: float = 0.0
    rerank_score: float | None = None


# ---------------------------------------------------------------- documents

class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_bytes: int
    chunk_count: int
    doc_type: str = "document"
    uploaded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------- API

class IngestResponse(BaseModel):
    document: DocumentInfo
    message: str = "ingested"


class QueryRequest(BaseModel):
    query: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    doc_type: str | None = None      # optional metadata filter
    top_k: int | None = None


class Citation(BaseModel):
    chunk_id: str
    filename: str
    section: str = ""
    page: int = 0
    excerpt: str
    rerank_score: float | None = None
    retrieval_score: float = 0.0


ValidationStatus = Literal["clean", "uncited_claims", "rerun", "insufficient_context"]


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[Citation]
    validation_status: ValidationStatus
    sub_queries: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    hyde_applied: bool = False
    total_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str]
