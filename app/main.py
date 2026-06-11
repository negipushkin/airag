"""KnowledgeOS API entry point.

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app import __version__
from app.api.routes import health_router, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="KnowledgeOS",
    description="Enterprise RAG platform - advanced retrieval pipeline (TDD v2)",
    version=__version__,
)
app.include_router(router)
app.include_router(health_router)
