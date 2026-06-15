"""Application configuration loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    openai_api_key: str = ""
    pinecone_api_key: str = ""
    cohere_api_key: str = ""

    # Pinecone
    pinecone_index_name: str = "knowledgeos-dev"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Auth (optional simple header auth)
    api_auth_key: str = ""

    # Models
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    sparse_model: str = "pinecone-sparse-english-v0"
    transform_model: str = "gpt-4.1-mini"  # Stage 4: decomposition / HyDE / rewrite
    synthesis_model: str = "gpt-4.1"      # Stage 6: grounded answer synthesis
    cohere_rerank_model: str = "rerank-english-v3.0"
    local_cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking (TDD Stage 2)
    max_chunk_tokens: int = 512
    overlap_tokens: int = 64

    # Retrieval (TDD Stage 5)
    retrieval_alpha: float = 0.7      # 0 = pure sparse, 1 = pure dense
    top_k_candidates: int = 20
    top_k_final: int = 5
    rrf_k: int = 60
    min_rerank_score: float = 0.15    # below this -> INSUFFICIENT_CONTEXT

    # Query transforms (TDD Stage 4)
    hyde_max_query_words: int = 8     # HyDE applied only to short/vague queries
    hyde_query_weight: float = 0.7    # fused = w*query + (1-w)*hyde

    # Synthesis (TDD Stage 6)
    max_chunk_chars_in_context: int = 2000  # prompt-injection mitigation 7.2

    # Storage
    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
