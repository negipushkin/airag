"""Create the Pinecone serverless index (TDD migration step 1).

Usage:  python scripts/create_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.indexing import VectorStore


def main() -> None:
    s = get_settings()
    if not s.pinecone_api_key:
        sys.exit("PINECONE_API_KEY is not set (copy .env.example to .env)")
    store = VectorStore()
    store.ensure_index()
    print(f"Index '{s.pinecone_index_name}' ready "
          f"({s.embedding_dimensions}-dim, dotproduct, "
          f"{s.pinecone_cloud}/{s.pinecone_region})")


if __name__ == "__main__":
    main()
