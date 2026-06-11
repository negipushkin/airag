"""CitationValidator tests (TDD 10.1: clean pass, hallucinated filename,
uncited claim, insufficient-context passthrough)."""

from app.models import RetrievedChunk
from app.synthesis import CitationValidator


def chunks():
    return [
        RetrievedChunk(
            chunk_id="c1", doc_id="d1", filename="policy-v3.pdf",
            section="4.2", page=12, text="AES-256 required.",
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="d2", filename="hipaa-sop.pdf",
            section="1", page=1, text="Keys rotate every 90 days.",
        ),
    ]


def test_clean_response_passes():
    answer = (
        "Encryption at rest must use AES-256 "
        "[Source: policy-v3.pdf, Section: 4.2, Page: 12]."
    )
    result = CitationValidator().validate(answer, chunks())
    assert result.status == "clean"
    assert result.cited_files == ["policy-v3.pdf"]


def test_hallucinated_filename_flagged_as_rerun():
    answer = "Backups run nightly [Source: backup-policy.pdf, Section: 2, Page: 3]."
    result = CitationValidator().validate(answer, chunks())
    assert result.status == "rerun"
    assert result.hallucinated_files == ["backup-policy.pdf"]


def test_uncited_substantial_claim_flagged():
    answer = (
        "All patient health information stored in cloud environments must use "
        "strong encryption and keys must be rotated on a fixed schedule by the team."
    )
    result = CitationValidator().validate(answer, chunks())
    assert result.status == "uncited_claims"
    assert result.uncited_sentences


def test_insufficient_context_is_clean():
    answer = "INSUFFICIENT_CONTEXT: the retention period is not in the context."
    result = CitationValidator().validate(answer, chunks())
    assert result.status == "clean"


def test_citation_match_is_case_insensitive():
    answer = "Rotation is 90 days [Source: HIPAA-SOP.PDF, Section: 1, Page: 1]."
    result = CitationValidator().validate(answer, chunks())
    assert result.status in ("clean", "uncited_claims")
    assert not result.hallucinated_files
