"""KnowledgeOS API entry point.

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import health_router, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="KnowledgeOS",
    description="Enterprise RAG platform - advanced retrieval pipeline (TDD v2)",
    version=__version__,
)
app.include_router(router)
app.include_router(health_router)

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(_STATIC / "index.html")
