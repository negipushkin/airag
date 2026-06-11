"""Stage 6c - Post-generation citation validation.

1. Parse [Source: ...] markers.
2. Verify each cited filename exists in the retrieved set
   (hallucinated citation -> HALLUCINATION flag).
3. Flag factual sentences without citations (UNCITED_CLAIM).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models import RetrievedChunk

_CITATION = re.compile(r"\[Source:\s*([^,\]]+?)(?:,[^\]]*)?\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ValidationResult:
    status: str  # "clean" | "uncited_claims" | "rerun"
    cited_files: list[str] = field(default_factory=list)
    hallucinated_files: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)


class CitationValidator:
    def validate(self, answer: str, chunks: list[RetrievedChunk]) -> ValidationResult:
        if answer.strip().startswith("INSUFFICIENT_CONTEXT"):
            return ValidationResult(status="clean")

        valid_files = {c.filename.lower() for c in chunks}
        cited = [m.group(1).strip() for m in _CITATION.finditer(answer)]
        hallucinated = sorted({
            f for f in cited if f.lower() not in valid_files
        })

        if hallucinated:
            return ValidationResult(
                status="rerun",
                cited_files=sorted(set(cited)),
                hallucinated_files=hallucinated,
            )

        uncited = self._uncited_sentences(answer)
        status = "uncited_claims" if uncited else "clean"
        return ValidationResult(
            status=status,
            cited_files=sorted(set(cited)),
            uncited_sentences=uncited,
        )

    @staticmethod
    def _uncited_sentences(answer: str) -> list[str]:
        """Heuristic: substantial sentences with no citation marker anywhere
        in the same paragraph are flagged as uncited claims."""
        uncited: list[str] = []
        for paragraph in answer.split("\n\n"):
            if _CITATION.search(paragraph):
                continue
            for sentence in _SENTENCE_SPLIT.split(paragraph.strip()):
                s = sentence.strip()
                # Skip short fragments, headers, list markers
                if len(s) >= 60 and not s.startswith(("#", "-", "*", "|")):
                    uncited.append(s)
        return uncited
