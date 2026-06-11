"""RAGAS evaluation harness (TDD section 4).

Requires extras:  pip install ragas datasets
Dataset format (JSONL), one object per line:
  {"question": "...", "ground_truth": "..."}

Usage:  python scripts/run_ragas.py --dataset eval/dev_questions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAITHFULNESS_GATE = 0.95
CONTEXT_PRECISION_GATE = 0.80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to eval JSONL")
    args = ap.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        sys.exit("Install eval extras first:  pip install ragas datasets")

    from app.models import QueryRequest
    from app.pipeline import get_pipeline

    pipeline = get_pipeline()
    rows = []
    for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        resp = pipeline.query(QueryRequest(query=item["question"]))
        rows.append({
            "question": item["question"],
            "answer": resp.answer,
            "contexts": [c.excerpt for c in resp.citations],
            "ground_truth": item["ground_truth"],
        })
        print(f"  evaluated: {item['question'][:60]}")

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print("\nRAGAS results:", result)

    scores = result.to_pandas().mean(numeric_only=True)
    f = float(scores.get("faithfulness", 0))
    p = float(scores.get("context_precision", 0))
    if f < FAITHFULNESS_GATE:
        sys.exit(f"GATE FAILED: faithfulness {f:.3f} < {FAITHFULNESS_GATE}")
    if p < CONTEXT_PRECISION_GATE:
        sys.exit(f"GATE FAILED: context_precision {p:.3f} < {CONTEXT_PRECISION_GATE}")
    print("CI gate passed.")


if __name__ == "__main__":
    main()
