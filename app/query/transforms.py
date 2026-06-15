"""Stage 4 - Query transformation: decomposition, HyDE, rewriting.

All three passes run on a small, fast model (gpt-4o-mini) for low latency.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from app.config import get_settings

_DECOMPOSE_SYSTEM = (
    "You decompose questions. Return a JSON array of atomic sub-questions. "
    "If the question is already atomic, return a single-element array. "
    "Return only valid JSON. No preamble."
)

_HYDE_TEMPLATE = (
    "Write a 2-sentence passage that would appear in an enterprise "
    "document and directly answer this question: {query}\n"
    "Respond with only the passage text."
)

_REWRITE_TEMPLATE = (
    "Given this conversation history:\n{history}\n\n"
    "Rewrite the following query to be fully self-contained and unambiguous. "
    "Return only the rewritten query, nothing else.\n"
    "Query: {query}"
)


class QueryTransformer:
    def __init__(self, client: OpenAI | None = None) -> None:
        s = get_settings()
        self._settings = s
        self._client = client or OpenAI(api_key=s.openai_api_key, max_retries=1, timeout=8.0)
        self._model = s.transform_model

    # ------------------------------------------------------- 4a decomposition

    def decompose(self, query: str) -> list[str]:
        """Split multi-part questions into atomic sub-queries."""
        try:
            text = self._complete(_DECOMPOSE_SYSTEM, f"Question: {query}", max_tokens=512)
            parsed = json.loads(_extract_json_array(text))
            subs = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
            return subs or [query]
        except Exception:
            return [query]  # malformed LLM response -> fall back to original

    # ------------------------------------------------------- 4b HyDE

    def should_apply_hyde(self, query: str) -> bool:
        return len(query.split()) < self._settings.hyde_max_query_words

    def hyde_passage(self, query: str) -> str | None:
        """Generate a hypothetical answer passage for vague queries."""
        try:
            return self._complete(
                system=None,
                user=_HYDE_TEMPLATE.format(query=query),
                max_tokens=256,
            ).strip()
        except Exception:
            return None

    @staticmethod
    def fuse_embeddings(
        query_emb: list[float], hyde_emb: list[float], query_weight: float
    ) -> list[float]:
        w = query_weight
        return [w * q + (1 - w) * h for q, h in zip(query_emb, hyde_emb, strict=True)]

    # ------------------------------------------------------- 4c rewriting

    def rewrite(self, query: str, history: list[dict[str, str]]) -> str:
        """Make the query self-contained using conversation history."""
        if not history:
            return query
        formatted = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-10:]
        )
        try:
            rewritten = self._complete(
                system=None,
                user=_REWRITE_TEMPLATE.format(history=formatted, query=query),
                max_tokens=256,
            ).strip()
            return rewritten or query
        except Exception:
            return query

    # ------------------------------------------------------- shared

    def _complete(self, system: str | None, user: str, max_tokens: int) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


def _extract_json_array(text: str) -> str:
    """Tolerate code fences / preamble around the JSON array."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON array in response")
    return m.group(0)
