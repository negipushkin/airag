"""Pinecone hybrid index wrapper (TDD Stage 3 + Stage 5a).

Hybrid weighting uses convex scaling: dense * alpha, sparse * (1 - alpha),
the standard approach for dotproduct sparse-dense indexes.
"""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

_UPSERT_BATCH = 100


def scale_hybrid(
    dense: list[float], sparse: dict, alpha: float
) -> tuple[list[float], dict]:
    """Convex-scale dense and sparse query vectors by alpha (0=sparse, 1=dense)."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


class VectorStore:
    def __init__(self, pc: Pinecone | None = None) -> None:
        s = get_settings()
        self._settings = s
        self._pc = pc or Pinecone(api_key=s.pinecone_api_key)
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = self._pc.Index(self._settings.pinecone_index_name)
        return self._index

    def ensure_index(self) -> None:
        """Create the serverless index if it doesn't exist (dotproduct for hybrid)."""
        s = self._settings
        existing = {ix["name"] for ix in self._pc.list_indexes()}
        if s.pinecone_index_name not in existing:
            self._pc.create_index(
                name=s.pinecone_index_name,
                dimension=s.embedding_dimensions,
                metric="dotproduct",
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )

    # ------------------------------------------------------------- write

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict],
    ) -> None:
        vectors = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            vectors.append({
                "id": chunk.chunk_id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": {
                    "docId": chunk.doc_id,
                    "filename": chunk.filename,
                    "section": chunk.section[:500],
                    "page": chunk.page,
                    "docType": chunk.doc_type,
                    "uploadedAt": chunk.uploaded_at,
                    "chunkIndex": chunk.chunk_index,
                    "text": chunk.text[:35000],  # raw text kept for rerank/context
                },
            })
        for i in range(0, len(vectors), _UPSERT_BATCH):
            self.index.upsert(vectors=vectors[i:i + _UPSERT_BATCH])

    def delete_document(self, doc_id: str) -> None:
        self.index.delete(filter={"docId": {"$eq": doc_id}})

    # ------------------------------------------------------------- read

    def hybrid_query(
        self,
        dense_query: list[float],
        sparse_query: dict,
        top_k: int,
        alpha: float,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        dense_scaled, sparse_scaled = scale_hybrid(dense_query, sparse_query, alpha)
        result = self.index.query(
            vector=dense_scaled,
            sparse_vector=sparse_scaled,
            top_k=top_k,
            filter=metadata_filter,
            include_metadata=True,
        )
        out: list[RetrievedChunk] = []
        for match in result["matches"]:
            md = match.get("metadata") or {}
            out.append(RetrievedChunk(
                chunk_id=match["id"],
                doc_id=md.get("docId", ""),
                filename=md.get("filename", ""),
                section=md.get("section", ""),
                page=int(md.get("page", 0)),
                chunk_index=int(md.get("chunkIndex", 0)),
                text=md.get("text", ""),
                retrieval_score=float(match.get("score", 0.0)),
            ))
        return out

    def health(self) -> str:
        try:
            self.index.describe_index_stats()
            return "ok"
        except Exception as exc:  # pragma: no cover
            return f"error: {exc}"
