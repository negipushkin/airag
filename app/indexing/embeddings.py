"""Dense embedding generation via OpenAI text-embedding-3-small (TDD Stage 3)."""

from __future__ import annotations

import httpx
from openai import OpenAI

from app.config import get_settings

_BATCH_SIZE = 96


class EmbeddingService:
    def __init__(self) -> None:
        s = get_settings()
        self._client = OpenAI(
            api_key=s.openai_api_key,
            max_retries=0,
            timeout=6.0,
            http_client=httpx.Client(http2=False),
        )
        self._model = s.embedding_model
        self._dimensions = s.embedding_dimensions

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts in batches."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            resp = self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dimensions,
            )
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._model, input=text, dimensions=self._dimensions,
        )
        return resp.data[0].embedding
