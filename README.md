# AdaptiveRAG

A production-grade Adaptive RAG system with a CRAG (Corrective RAG) correction layer: hybrid retrieval (Dense + BM25 + Graph), query planning, evidence grading with a governed recovery policy, streaming generation, and a two-mode grounding check - deployed entirely on free-tier infrastructure. See `docs/01-prd.md` for the full spec.

## Architecture

```
User
 -> Frontend (React + Vite + TS, Vercel)
 -> Backend API (FastAPI, Render)  [POST /v1/query, GET /v1/health, POST /v1/documents]
 -> AdaptiveRAG pipeline:
      Query Understanding & Planning
      -> Retrieval (Dense/OpenSearch + BM25/OpenSearch + Graph/Neo4j) + RRF fusion + reranking
      -> CRAG Evidence Grading -> Recovery Policy (rewrite / re-retrieve / graph expand / web search) if needed
      -> Generation (Groq, `gpt-oss-20b`)
      -> Grounding check (async, non-gating - or sync, blocking, for high-risk queries)
 -> Frontend renders the answer + sources

Document upload: User -> Frontend -> POST /v1/documents -> same ingestion
pipeline (chunk/extract/resolve/validate -> Neo4j + OpenSearch) used by
offline ingestion, just triggered synchronously by one file upload.
```

The frontend is a thin client: it calls the real `/v1/query`/`/v1/health` endpoints and renders what they return. It does not reimplement any retrieval/CRAG/grounding/generation logic. See `docs/01-prd.md` Section 19 for the full frontend spec.

## Local setup

### 1. Backend

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in real values - see Environment variables below
uvicorn adaptive_rag.app:app --reload --port 8000
```

Verify: `curl http://localhost:8000/v1/health`

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`. On first load, click the settings (gear) icon and enter the same value as the backend's `API_KEY`. Use the upload panel at the top to add a `.pdf`/`.txt` document to the knowledge base before asking questions about it - documents are capped at ~8,000 extracted characters per upload (a free-tier-safe synchronous processing limit, see `docs/01-prd.md` FR-ING5).

### 3. Required environment variables

**Backend** (`.env`, see `.env.example` for the full annotated list):

| Variable | Purpose |
|---|---|
| `OPENSEARCH_URL` / `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD` | Dense + BM25 retrieval |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Graph retrieval |
| `REDIS_URL` | Cache + rate limiting |
| `GROQ_API_KEY` / `GROQ_MODEL` / `GROQ_EXTRACTION_MODEL` | Generation + ingestion-time extraction |
| `API_KEY` | The key `/v1/query` requires as `X-API-Key` |
| `RATE_LIMIT_PER_MINUTE` | Per-key rate limit |
| `CACHE_TTL_SECONDS` | Exact-match cache TTL |
| `CORS_ALLOWED_ORIGINS` | Comma-separated browser origins allowed to call the API - must include the frontend's origin |

**Frontend** (`frontend/.env.local` for dev, Vercel project settings for production):

| Variable | Purpose |
|---|---|
| `VITE_API_BASE_URL` | The backend's base URL - no trailing slash |

The frontend's API key is **not** an environment variable - the user enters it in the UI (Settings), and it's stored only in that browser's `localStorage`. No secret of any kind (Groq/Neo4j/OpenSearch/Redis credentials or the backend `API_KEY`) is ever placed in frontend source, `VITE_`-prefixed variables, or the built bundle.

## Production setup

| | |
|---|---|
| Frontend | Vercel (free tier) - framework preset "Vite", build `npm run build`, output `dist`, env var `VITE_API_BASE_URL` set to the deployed Render backend's URL |
| Backend | Render (free tier) - env var `CORS_ALLOWED_ORIGINS` must include the deployed Vercel URL |
| External services | OpenSearch (Aiven free tier) / Neo4j (AuraDB free tier) / Redis / Groq - see `docs/01-prd.md` Section 11 |

No paid infrastructure is required anywhere in this stack.

## Testing

```bash
# Backend
python -m pytest tests/ -q
python -m ruff check src/ tests/

# Frontend
cd frontend
npm run test
npm run lint
npm run build
```
