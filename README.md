# KnowledgeOS — Enterprise RAG Platform (v2)

Production-grade advanced RAG backend implementing the TDD v2 six-stage
pipeline: structure-aware parsing → semantic chunking → dual-path
(dense + sparse) indexing → query transformation → hybrid retrieval with
re-ranking → grounded synthesis with citation validation.

## Stack (TDD Option A)

| Layer | Technology |
|---|---|
| Dense embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Sparse embeddings | Pinecone `pinecone-sparse-english-v0` (SPLADE) |
| Vector store | Pinecone Serverless (dotproduct, hybrid) |
| Re-ranker | Cohere Rerank v3 → local cross-encoder → passthrough (auto-fallback) |
| Query transforms | OpenAI `gpt-4o-mini` (decomposition / HyDE / rewriting) |
| Answer synthesis | OpenAI `gpt-4o`, streaming-capable, citation-validated |
| API | FastAPI |

> Note: the TDD specifies Claude (Anthropic) for transforms and synthesis;
> this build substitutes OpenAI models so a single key covers the whole
> pipeline. The provider is isolated in `app/query/transforms.py` and
> `app/synthesis/synthesizer.py` if you want to switch back.

## Setup

```sh
cd knowledgeos
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env           # then fill in keys
python scripts/create_index.py   # one-time Pinecone index creation

uvicorn app.main:app --reload    # http://localhost:8000/docs
```

Required keys: `OPENAI_API_KEY`, `PINECONE_API_KEY`.
Optional: `COHERE_API_KEY` (better reranking), `API_AUTH_KEY` (header auth).

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/ingest` | POST | multipart upload (PDF/DOCX/TXT/MD, ≤50 MB) |
| `/api/query` | POST | `{"query": "...", "conversation_history": []}` |
| `/api/documents` | GET | list indexed documents |
| `/api/documents/{id}` | DELETE | remove document + its chunks |
| `/api/health` | GET | dependency checks (no auth) |

Example:

```sh
curl -X POST localhost:8000/api/ingest -F "file=@policy.pdf"
curl -X POST localhost:8000/api/query -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the data retention policy?\"}"
```

Every answer carries inline `[Source: file, Section: s, Page: n]` citations,
a citation panel payload (`citations[]`), and a `validation_status`
(`clean` / `uncited_claims` / `rerun` / `insufficient_context`). Hallucinated
citations trigger an automatic strict re-run; persistent failures return
`INSUFFICIENT_CONTEXT` instead of an ungrounded answer.

## Tests

```sh
pytest tests/            # unit tests, no network or API keys needed
```

## Evaluation (RAGAS)

```sh
pip install ragas datasets
python scripts/run_ragas.py --dataset eval/dev_questions.jsonl
```

CI gates: faithfulness ≥ 0.95, context_precision ≥ 0.80.

## Observability

Each query emits one JSON trace line (stdout + `data/traces.jsonl`) with the
TDD §6 schema: decomposition, HyDE flag, candidate/rerank scores, stage
latencies, token usage, and citation validation status.
