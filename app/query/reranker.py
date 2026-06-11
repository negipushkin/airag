"""Stage 5b - Cross-encoder re-ranking (top-20 -> top-5).

Pluggable: Cohere Rerank v3 when COHERE_API_KEY is set; otherwise a local
cross-encoder (sentence-transformers) if installed; otherwise pass-through
ordering by retrieval score.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.config import get_settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        ...


class CohereReranker(Reranker):
    name = "cohere-rerank-v3"

    def __init__(self) -> None:
        import cohere

        s = get_settings()
        self._client = cohere.Client(api_key=s.cohere_api_key)
        self._model = s.cohere_rerank_model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        resp = self._client.rerank(
            query=query,
            documents=[c.text for c in chunks],
            model=self._model,
            top_n=top_n,
        )
        out = []
        for result in resp.results:
            chunk = chunks[result.index].model_copy()
            chunk.rerank_score = float(result.relevance_score)
            out.append(chunk)
        return out


class LocalCrossEncoderReranker(Reranker):
    name = "local-ms-marco-minilm"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        s = get_settings()
        self._model = CrossEncoder(s.local_cross_encoder)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        import math

        raw = self._model.predict([(query, c.text) for c in chunks])
        scored = []
        for chunk, score in zip(chunks, raw, strict=True):
            c = chunk.model_copy()
            c.rerank_score = 1 / (1 + math.exp(-float(score)))  # sigmoid -> [0,1]
            scored.append(c)
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return scored[:top_n]


class PassthroughReranker(Reranker):
    """No reranking model available: keep retrieval-score order."""

    name = "passthrough"

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        ordered = sorted(chunks, key=lambda c: c.retrieval_score, reverse=True)[:top_n]
        out = []
        for c in ordered:
            c = c.model_copy()
            c.rerank_score = c.retrieval_score  # best available proxy
            out.append(c)
        return out


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is not None:
        return _reranker

    s = get_settings()
    if s.cohere_api_key:
        try:
            _reranker = CohereReranker()
            logger.info("Reranker: Cohere Rerank v3")
            return _reranker
        except Exception as exc:
            logger.warning("Cohere reranker unavailable (%s); falling back", exc)
    try:
        _reranker = LocalCrossEncoderReranker()
        logger.info("Reranker: local cross-encoder (ms-marco-MiniLM)")
        return _reranker
    except Exception:
        logger.warning(
            "No reranker available (set COHERE_API_KEY or install "
            "sentence-transformers); using retrieval-score passthrough"
        )
        _reranker = PassthroughReranker()
        return _reranker
