"""End-to-end pipeline orchestration: ingest (Stages 1-3), query (Stages 4-6)."""

from __future__ import annotations

import logging
import time

from app.config import get_settings
import os

from app.indexing import DocumentRegistry, EmbeddingService, PineconeDocumentRegistry, SparseEncoder, VectorStore
from app.ingestion import DocumentParser, SemanticChunker
from app.models import (
    Citation,
    DocumentInfo,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from app.observability import QueryTrace, log_trace
from app.query import HybridRetriever, QueryTransformer, get_reranker
from app.synthesis import AnswerSynthesizer, CitationValidator, ContextBuilder

logger = logging.getLogger(__name__)


class Pipeline:
    """Lazily-wired singleton holding all pipeline components."""

    def __init__(self) -> None:
        s = get_settings()
        self.settings = s
        self.parser = DocumentParser()
        self.chunker = SemanticChunker(s.max_chunk_tokens, s.overlap_tokens)
        self.embeddings = EmbeddingService()
        self.sparse = SparseEncoder()
        self.store = VectorStore()
        # On Vercel (no persistent filesystem) use Pinecone-backed registry.
        if os.getenv("VERCEL"):
            self.registry = PineconeDocumentRegistry(self.store)
        else:
            self.registry = DocumentRegistry(s.data_dir)
        self.retriever = HybridRetriever(self.embeddings, self.sparse, self.store)
        self.transformer = QueryTransformer()
        self.context_builder = ContextBuilder()
        self.synthesizer = AnswerSynthesizer()
        self.validator = CitationValidator()

    # ------------------------------------------------------------- ingest

    def ingest(self, data: bytes, filename: str, doc_type: str = "document") -> DocumentInfo:
        parsed = self.parser.parse(data, filename)
        chunks = self.chunker.chunk(parsed, doc_type=doc_type)
        if not chunks:
            raise ValueError("No extractable text found in document")

        texts = [c.text for c in chunks]
        dense = self.embeddings.embed_passages(texts)
        sparse = self.sparse.encode_passages(texts)
        self.store.upsert_chunks(chunks, dense, sparse)

        info = DocumentInfo(
            id=parsed.id,
            filename=parsed.filename,
            size_bytes=len(data),
            chunk_count=len(chunks),
            doc_type=doc_type,
        )
        self.registry.add(info)
        logger.info("Ingested %s: %d chunks", parsed.filename, len(chunks))
        return info

    def delete_document(self, doc_id: str) -> bool:
        if self.registry.get(doc_id) is None:
            return False
        self.store.delete_document(doc_id)
        return self.registry.remove(doc_id)

    # ------------------------------------------------------------- query

    def query(self, req: QueryRequest) -> QueryResponse:
        s = self.settings
        t0 = time.perf_counter()
        trace = QueryTrace(query_original=req.query)

        # --- Stage 4: query transformation
        query = self.transformer.rewrite(req.query, req.conversation_history)
        trace.query_rewritten = query

        sub_queries = self.transformer.decompose(query)
        trace.query_decomposed = sub_queries

        dense_override = None
        if len(sub_queries) == 1 and self.transformer.should_apply_hyde(query):
            passage = self.transformer.hyde_passage(query)
            if passage:
                q_emb = self.embeddings.embed_query(query)
                h_emb = self.embeddings.embed_query(passage)
                dense_override = self.transformer.fuse_embeddings(
                    q_emb, h_emb, s.hyde_query_weight
                )
                trace.hyde_applied = True

        # --- Stage 5: hybrid retrieval + re-ranking
        if req.doc_id:
            metadata_filter = {"docId": {"$eq": req.doc_id}}
        elif req.doc_type:
            metadata_filter = {"docType": {"$eq": req.doc_type}}
        else:
            metadata_filter = None
        t_ret = time.perf_counter()
        candidates = self.retriever.retrieve(
            sub_queries, metadata_filter=metadata_filter, dense_override=dense_override
        )
        retrieval_ms = int((time.perf_counter() - t_ret) * 1000)

        top_k = req.top_k or s.top_k_final
        t_rr = time.perf_counter()
        reranker = get_reranker()
        top_chunks = reranker.rerank(query, candidates, top_n=top_k)
        rerank_ms = int((time.perf_counter() - t_rr) * 1000)

        trace.retrieval = {
            "candidates_fetched": len(candidates),
            "candidate_scores": [round(c.retrieval_score, 4) for c in candidates],
            "reranked_scores": [
                round(c.rerank_score or 0.0, 4) for c in top_chunks
            ],
            "sources_used": [c.filename for c in top_chunks],
            "reranker": reranker.name,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
        }

        # --- Refusal gate: no sufficiently relevant chunks
        # PassthroughReranker copies RRF retrieval_score (~0.016) into rerank_score;
        # those values are not reranker confidence scores, so skip the threshold.
        best_score = max((c.rerank_score or 0.0 for c in top_chunks), default=0.0)
        score_gate_active = reranker.name != "passthrough"
        if not top_chunks or (score_gate_active and best_score < s.min_rerank_score):
            trace.total_ms = int((time.perf_counter() - t0) * 1000)
            log_trace(trace, s.data_dir)
            return QueryResponse(
                trace_id=trace.trace_id,
                answer=(
                    "INSUFFICIENT_CONTEXT: No sufficiently relevant content was "
                    "found in the knowledge base for this query."
                ),
                citations=[],
                validation_status="insufficient_context",
                sub_queries=sub_queries,
                rewritten_query=query if query != req.query else None,
                hyde_applied=trace.hyde_applied,
                total_ms=trace.total_ms,
            )

        # --- Stage 6: grounded synthesis + citation validation
        context_block = self.context_builder.build(top_chunks)
        t_syn = time.perf_counter()
        answer, usage = self.synthesizer.synthesize(req.query, context_block)
        validation = self.validator.validate(answer, top_chunks)

        if validation.status in ("rerun", "uncited_claims"):
            # Hallucinated citation or uncited claims: re-run once with stricter prompt.
            logger.warning(
                "Validation issue (%s) — re-running with strict prompt. "
                "Hallucinated: %s  Uncited sentences: %d",
                validation.status,
                validation.hallucinated_files,
                len(validation.uncited_sentences),
            )
            answer, usage = self.synthesizer.synthesize(
                req.query, context_block, strict=True
            )
            revalidation = self.validator.validate(answer, top_chunks)
            status = revalidation.status if revalidation.status != "rerun" else "rerun"
            if revalidation.status == "rerun":
                answer = (
                    "INSUFFICIENT_CONTEXT: The answer could not be reliably "
                    "grounded in the retrieved sources."
                )
        else:
            status = validation.status

        synthesis_ms = int((time.perf_counter() - t_syn) * 1000)
        trace.synthesis = {
            **usage,
            "synthesis_ms": synthesis_ms,
            "citation_count": len(validation.cited_files),
            "validation_status": status,
        }
        trace.total_ms = int((time.perf_counter() - t0) * 1000)
        log_trace(trace, s.data_dir)

        citations = [
            Citation(
                chunk_id=c.chunk_id,
                filename=c.filename,
                section=c.section,
                page=c.page,
                excerpt=c.text[:200],
                rerank_score=c.rerank_score,
                retrieval_score=c.retrieval_score,
            )
            for c in top_chunks
        ]
        return QueryResponse(
            trace_id=trace.trace_id,
            answer=answer,
            citations=citations,
            validation_status=status,
            sub_queries=sub_queries,
            rewritten_query=query if query != req.query else None,
            hyde_applied=trace.hyde_applied,
            total_ms=trace.total_ms,
        )


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
