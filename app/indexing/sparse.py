"""Sparse (SPLADE-style) vector generation via Pinecone inference (TDD Stage 3)."""

from __future__ import annotations

from pinecone import Pinecone

from app.config import get_settings

_BATCH_SIZE = 96


class SparseEncoder:
    def __init__(self, pc: Pinecone | None = None) -> None:
        s = get_settings()
        self._pc = pc or Pinecone(api_key=s.pinecone_api_key)
        self._model = s.sparse_model

    def encode_passages(self, texts: list[str]) -> list[dict]:
        return self._encode(texts, input_type="passage")

    def encode_query(self, text: str) -> dict:
        return self._encode([text], input_type="query")[0]

    def _encode(self, texts: list[str], input_type: str) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            resp = self._pc.inference.embed(
                model=self._model,
                inputs=batch,
                parameters={"input_type": input_type, "truncate": "END"},
            )
            for item in resp:
                out.append({
                    "indices": list(item["sparse_indices"]),
                    "values": list(item["sparse_values"]),
                })
        return out
