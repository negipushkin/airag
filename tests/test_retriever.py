"""RRF merge + hybrid scaling tests (TDD 10.1 HybridRetriever cases)."""

import pytest

from app.indexing.vector_store import scale_hybrid
from app.models import RetrievedChunk
from app.query.retriever import rrf_merge


def chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id="d1", filename="f.pdf", text=f"text {cid}",
        retrieval_score=score,
    )


def test_rrf_score_calculation():
    # chunk "a" ranked 1st in both sets: score = 2 * 1/(60+1)
    set1 = [chunk("a", 0.9), chunk("b", 0.5)]
    set2 = [chunk("a", 0.8), chunk("c", 0.4)]
    merged = rrf_merge([set1, set2], k=60, top_n=10)
    assert merged[0].chunk_id == "a"
    assert merged[0].retrieval_score == pytest.approx(2 / 61)
    # b and c each got 1/(60+2)
    tail_ids = {c.chunk_id for c in merged[1:]}
    assert tail_ids == {"b", "c"}


def test_rrf_dedupes_across_subqueries():
    set1 = [chunk("x", 0.9)]
    set2 = [chunk("x", 0.7)]
    merged = rrf_merge([set1, set2], k=60, top_n=10)
    assert len(merged) == 1


def test_rrf_respects_top_n():
    sets = [[chunk(f"c{i}", 1.0 - i * 0.01) for i in range(30)]]
    merged = rrf_merge(sets, k=60, top_n=20)
    assert len(merged) == 20


def test_scale_hybrid_alpha_weighting():
    dense = [1.0, 2.0]
    sparse = {"indices": [3, 7], "values": [0.5, 1.0]}
    d, s = scale_hybrid(dense, sparse, alpha=0.7)
    assert d == pytest.approx([0.7, 1.4])
    assert s["values"] == pytest.approx([0.15, 0.3])
    assert s["indices"] == [3, 7]


def test_scale_hybrid_rejects_bad_alpha():
    with pytest.raises(ValueError):
        scale_hybrid([1.0], {"indices": [], "values": []}, alpha=1.5)
