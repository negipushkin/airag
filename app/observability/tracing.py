"""Pipeline trace logging (TDD section 6).

One newline-delimited JSON object per query, written to stdout and to
data/traces.jsonl for offline analysis.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    query_original: str = ""
    query_decomposed: list[str] = field(default_factory=list)
    query_rewritten: str = ""
    hyde_applied: bool = False
    retrieval: dict = field(default_factory=dict)
    synthesis: dict = field(default_factory=dict)
    total_ms: int = 0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, default=str)


def log_trace(trace: QueryTrace, data_dir: Path | None = None) -> None:
    line = trace.to_json()
    print(line, file=sys.stdout, flush=True)
    if data_dir is not None:
        try:
            path = data_dir / "traces.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # read-only filesystem (e.g. Vercel serverless)
