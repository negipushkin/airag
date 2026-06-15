"""Document registry — file-based (local dev) or Pinecone-backed (serverless).

PineconeDocumentRegistry stores document metadata in a dedicated
'__registry__' namespace so it survives Vercel cold starts with zero extra
services.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from app.models import DocumentInfo


class DocumentRegistry:
    """File-based registry — default for local dev and testing."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "documents.json"
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {}

    def _save(self, docs: dict[str, dict]) -> None:
        self._path.write_text(
            json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, info: DocumentInfo) -> None:
        with self._lock:
            docs = self._load()
            docs[info.id] = info.model_dump(mode="json")
            self._save(docs)

    def list(self) -> list[DocumentInfo]:
        with self._lock:
            return [DocumentInfo(**d) for d in self._load().values()]

    def get(self, doc_id: str) -> DocumentInfo | None:
        with self._lock:
            d = self._load().get(doc_id)
            return DocumentInfo(**d) if d else None

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            docs = self._load()
            if doc_id in docs:
                del docs[doc_id]
                self._save(docs)
                return True
            return False


class PineconeDocumentRegistry:
    """Registry backed by a dedicated '__registry__' namespace in Pinecone.

    Keeps document metadata consistent with the index across serverless cold
    starts. No extra services required beyond Pinecone itself.
    """

    _NAMESPACE = "__registry__"
    _DUMMY_SPARSE = {"indices": [0], "values": [0.001]}

    def __init__(self, store) -> None:  # store: VectorStore
        self._store = store

    @property
    def _index(self):
        return self._store.index

    def _zero_dense(self) -> list[float]:
        from app.config import get_settings
        dims = get_settings().embedding_dimensions
        v = [0.0] * dims
        v[0] = 1e-9  # Pinecone rejects all-zero dense vectors
        return v

    def add(self, info: DocumentInfo) -> None:
        self._index.upsert(
            vectors=[{
                "id": info.id,
                "values": self._zero_dense(),
                "sparse_values": self._DUMMY_SPARSE,
                "metadata": {
                    "filename": info.filename,
                    "size_bytes": info.size_bytes,
                    "chunk_count": info.chunk_count,
                    "doc_type": info.doc_type,
                    "uploaded_at": info.uploaded_at,
                },
            }],
            namespace=self._NAMESPACE,
        )

    def list(self) -> list[DocumentInfo]:
        try:
            ids = list(self._index.list(namespace=self._NAMESPACE))
            if not ids:
                return []
            fetched = self._index.fetch(ids=ids, namespace=self._NAMESPACE)
            return [
                self._to_info(vid, v.metadata)
                for vid, v in fetched.vectors.items()
            ]
        except Exception:
            return []

    def get(self, doc_id: str) -> DocumentInfo | None:
        try:
            fetched = self._index.fetch(ids=[doc_id], namespace=self._NAMESPACE)
            if not fetched.vectors:
                return None
            return self._to_info(doc_id, fetched.vectors[doc_id].metadata)
        except Exception:
            return None

    def remove(self, doc_id: str) -> bool:
        try:
            self._index.delete(ids=[doc_id], namespace=self._NAMESPACE)
            return True
        except Exception:
            return False

    @staticmethod
    def _to_info(doc_id: str, meta: dict) -> DocumentInfo:
        return DocumentInfo(
            id=doc_id,
            filename=meta.get("filename", ""),
            size_bytes=int(meta.get("size_bytes", 0)),
            chunk_count=int(meta.get("chunk_count", 0)),
            doc_type=meta.get("doc_type", "document"),
            uploaded_at=meta.get("uploaded_at", ""),
        )
