"""API layer (TDD section 5.2)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.models import (
    DocumentInfo,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.pipeline import get_pipeline

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

MAX_FILE_BYTES = 50 * 1024 * 1024  # PRD: 50 MB per document


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = get_settings().api_auth_key
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(file: UploadFile, doc_type: str = "document") -> IngestResponse:
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    loop = asyncio.get_event_loop()
    try:
        info = await loop.run_in_executor(
            None,
            lambda: get_pipeline().ingest(data, file.filename or "upload", doc_type=doc_type),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return IngestResponse(document=info)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: get_pipeline().query(req))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return get_pipeline().registry.list()


@router.get("/debug-registry")
def debug_registry() -> dict:
    """Temporary: diagnose registry list issues."""
    import os
    reg = get_pipeline().registry
    try:
        result = reg._index.query(
            vector=reg._zero_dense(),
            sparse_vector=reg._DUMMY_SPARSE,
            top_k=100,
            namespace=reg._NAMESPACE,
            include_metadata=True,
        )
        matches = result.get("matches", [])
        return {
            "namespace": reg._NAMESPACE,
            "match_count": len(matches),
            "matches": [{"id": m["id"], "metadata": m.get("metadata")} for m in matches],
            "vercel": bool(os.getenv("VERCEL")),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    if not get_pipeline().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")


# Health endpoint has no auth (TDD 5.2)
health_router = APIRouter()


@health_router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    checks = {
        "openai_key": "configured" if s.openai_api_key else "missing",
        "pinecone_key": "configured" if s.pinecone_api_key else "missing",
        "cohere_key": "configured" if s.cohere_api_key else "missing (fallback reranker)",
    }
    if s.pinecone_api_key:
        try:
            checks["pinecone_index"] = get_pipeline().store.health()
        except Exception as exc:
            checks["pinecone_index"] = f"error: {exc}"
    status = "ok" if all(
        v == "configured" for k, v in checks.items()
        if k in ("openai_key", "pinecone_key")
    ) else "degraded"
    return HealthResponse(status=status, checks=checks)
