"""QueryTransformer logic tests with a stubbed OpenAI client (no network)."""

from types import SimpleNamespace

import pytest

from app.query.transforms import QueryTransformer, _extract_json_array


class StubClient:
    """Mimics openai.OpenAI for chat.completions.create."""

    def __init__(self, reply: str):
        self._reply = reply
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        message = SimpleNamespace(content=self._reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_decompose_compound_query():
    reply = '["What is the data retention policy?", "Who enforces the data retention policy?"]'
    t = QueryTransformer(client=StubClient(reply))
    subs = t.decompose("What is the data retention policy and who is responsible?")
    assert len(subs) == 2


def test_decompose_atomic_returns_single():
    t = QueryTransformer(client=StubClient('["What is the policy?"]'))
    assert t.decompose("What is the policy?") == ["What is the policy?"]


def test_decompose_malformed_response_falls_back():
    t = QueryTransformer(client=StubClient("I cannot decompose that, sorry!"))
    assert t.decompose("original query") == ["original query"]


def test_hyde_trigger_threshold():
    t = QueryTransformer(client=StubClient("x"))
    assert t.should_apply_hyde("encryption policy")          # 2 words -> HyDE
    assert not t.should_apply_hyde(
        "what is the exact key rotation interval required for cloud KMS keys"
    )


def test_embedding_fusion_weights():
    fused = QueryTransformer.fuse_embeddings([1.0, 0.0], [0.0, 1.0], query_weight=0.7)
    assert fused == pytest.approx([0.7, 0.3])


def test_rewrite_skipped_without_history():
    t = QueryTransformer(client=StubClient("rewritten"))
    assert t.rewrite("same query", []) == "same query"


def test_rewrite_uses_history():
    t = QueryTransformer(client=StubClient("What is the HIPAA retention period?"))
    out = t.rewrite("what about retention?", [{"role": "user", "content": "Tell me about HIPAA"}])
    assert out == "What is the HIPAA retention period?"


def test_extract_json_array_with_fences():
    assert _extract_json_array('```json\n["a", "b"]\n```') == '["a", "b"]'
