"""Stage 5a - Hybrid retrieval with Reciprocal Rank Fusion.

Per sub-query: hybrid Pinecone search (top-k candidates). For multiple
sub-queries, result sets are merged via RRF and deduplicated by chunk id.
"""

from __future__ import annotations

from app.config import get_settings
from app.indexing import EmbeddingService, SparseEncoder, VectorStore
from app.models import RetrievedChunk


def rrf_merge(
    result_sets: list[list[RetrievedChunk]], k: int = 60, top_n: int = 20
) -> list[RetrievedChunk]:
    """rrf_score(chunk) = sum over sub-queries of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for results in result_sets:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.retrieval_score > existing.retrieval_score:
                best[chunk.chunk_id] = chunk
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    out = []
    for chunk_id, rrf_score in ranked:
        chunk = best[chunk_id].model_copy()
        chunk.retrieval_score = rrf_score
        out.append(chunk)
    return out


class HybridRetriever:
    def __init__(
        self,
        embeddings: EmbeddingService | None = None,
        sparse: SparseEncoder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self._settings = get_settings()
        self._embeddings = embeddings or EmbeddingService()
        self._sparse = sparse or SparseEncoder()
        self._store = store or VectorStore()

    def retrieve(
        self,
        sub_queries: list[str],
        metadata_filter: dict | None = None,
        dense_override: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k candidates across all sub-queries (RRF-merged).

        `dense_override` lets the caller pass a HyDE-fused embedding for a
        single-query retrieval.
        """
        s = self._settings
        result_sets: list[list[RetrievedChunk]] = []

        for i, sub_query in enumerate(sub_queries):
            if dense_override is not None and len(sub_queries) == 1:
                dense = dense_override
            else:
                dense = self._embeddings.embed_query(sub_query)
            sparse = self._sparse.encode_query(sub_query)
            results = self._store.hybrid_query(
                dense_query=dense,
                sparse_query=sparse,
                top_k=s.top_k_candidates,
                alpha=s.retrieval_alpha,
                metadata_filter=metadata_filter,
            )
            result_sets.append(results)

        if len(result_sets) == 1:
            return result_sets[0][: s.top_k_candidates]
        return rrf_merge(result_sets, k=s.rrf_k, top_n=s.top_k_candidates)
