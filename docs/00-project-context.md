# Project Context: Adaptive RAG (CRAG-based)

**Purpose of this file:** quick orientation for any AI agent working on this repo. Read this first. For full requirements, acceptance criteria, and the phased build checklist, go to `docs/01-prd.md` - this file does not replace it.

---

## 1. What this project is

An adaptive, cost-optimized RAG system with a CRAG-style correction layer: query-plan-based retrieval routing (Dense + BM25 + Graph), a governed evidence-recovery policy, a proper graph ingestion pipeline, and streaming generation - designed to keep exactly one expensive generation-model call per request while everything else runs on small models or rules. Runs entirely on free-tier infrastructure, with no Docker requirement.

## 2. Current status

- PRD finalized (v4 - Docker made optional for local development).
- **Phase 0 mostly complete (2026-09-11):** repo scaffold (`pyproject.toml`, `src/adaptive_rag/`), config via `pydantic-settings` reading all OpenSearch/Neo4j/Redis/Groq connectivity from env (no hardcoded localhost, verified by `tests/test_config.py`), JSON logging (`logging.py`), minimal FastAPI app with `/v1/health` stub. Files are staged but **not yet git-committed** - no git identity configured for this machine yet.
- **Blocked on Phase 0:** Render free-tier service provisioning (requires manual dashboard signup), storing secrets in Render's secrets manager.
- Not yet done in Phase 0: `infra/docker-compose.yml` (optional alternative, deferred - not needed until someone wants the local-Docker path).
- Next actionable step: commit the Phase 0 scaffold once git identity is set, then **Phase 1** in `docs/01-prd.md` Section 17 (API Gateway + cache).

## 3. Read order

1. This file (`00-project-context.md`) - orientation, principles, conventions.
2. `docs/01-prd.md` Section 2 - full architecture (diagrams, component inventory, Retrieval Plan schema, Recovery Policy).
3. `docs/01-prd.md` Section 3 - ingestion/graph-building pipeline.
4. `docs/01-prd.md` Section 7 - functional requirements + acceptance criteria (what "done" means per component).
5. `docs/01-prd.md` Section 11 - deployment and local development (Docker-optional setup).
6. `docs/01-prd.md` Section 17 - the actual phased checklist to execute against.

## 4. Locked tech stack

| Layer | Choice |
|---|---|
| Generation | `openai/gpt-oss-20b` via Groq |
| Backend hosting | Render (free tier) |
| Frontend/edge (optional) | Vercel (free tier) |
| Dense + BM25 retrieval | OpenSearch (default: free-tier hosted sandbox, e.g. Bonsai.io; Docker Compose optional alternative) |
| Graph retrieval | Neo4j (default: AuraDB free tier; Docker Compose optional alternative) |
| Cache | Redis (default: Upstash free tier; Docker Compose optional alternative) |
| Small models (planner, grader, recovery, grounding) | Mix of hosted free-tier APIs and self-hosted fine-tuned models |

**Deployment is free-tier only.** No paid infrastructure is designed, costed, or assumed anywhere in this project.

**Local development requires no Docker.** All three data services (OpenSearch, Neo4j, Redis) are reached via configurable environment variables, defaulting to free-tier hosted remote instances so the app runs with zero local installs or virtualization.

## 5. Non-negotiable architectural principles

These came out of a full architecture review - do not silently redesign around them without updating the PRD first.

1. **Exactly one expensive generation-model call per request** (default path). This is defined by *role in the pipeline*, not by parameter count or marketing label ("large"/"frontier" is not the criterion). Never let the planner, recovery, or grading steps quietly turn into a second generation-tier call.
2. **The Query Planner outputs a structured plan**, not free text - `{dense, bm25, graph, freshness, apply_filters, top_k}` - via a small multi-label classifier. It replaces the old fixed 5-class router.
3. **Never trust planner output blindly.** `top_k` and `apply_filters` are always clamped/validated downstream regardless of what the planner says (defense in depth).
4. **CRAG correction uses a governed Recovery Planner**, not automatic web search. Order of preference: query rewrite → internal re-retrieval → graph expansion → external web search (only if no authoritative internal source scored even moderately). Max 2 recovery attempts total, ever.
5. **Authoritative internal sources must never be silently overridden by web results.** Web results are always tagged `supplementary`.
6. **The knowledge graph (Neo4j) is an index, not evidence.** Every graph node/edge must resolve back to its source document + passage before being used as generation evidence. The LLM grounds on real document text, never on a paraphrased graph edge label.
7. **Grounding checks have two distinct modes** - don't conflate them:
   - Default/low-risk: async, detector-only. Can flag/log but cannot block an answer already streaming.
   - High-risk (medical/legal/financial): buffered, synchronous, blocking. Response is only released to the user after a PASS.
8. **Docker is optional, never required for local development.** All connections to OpenSearch/Neo4j/Redis must be configurable via environment variables (host/port/credentials) - never hardcoded to `localhost` or assumed to run through Docker networking. The default local setup points at free-tier hosted remote instances; `infra/docker-compose.yml` is kept only as an alternative for those who have Docker available.
9. **All performance and capacity claims must be measured on the actual free-tier deployment.** The 1k-50k req/day range is an architecture design target, not a guarantee of current capacity - never state it as if it's already proven.

## 6. Folder structure (current)

```
AdaptiveRAG1/
  CLAUDE.md                  <- currently empty
  pyproject.toml
  .env.example
  .gitignore
  src/
    adaptive_rag/
      __init__.py
      app.py                 <- minimal FastAPI app, /v1/health stub only
      config.py
      logging.py
  tests/
    test_config.py
  docs/
    00-project-context.md    <- this file
    01-prd.md                <- full PRD
```

`infra/docker-compose.yml` (optional local-Docker alternative) not yet created - deferred until needed.

## 7. Working conventions for agents

- In `01-prd.md` Section 17, mark a checklist item `[x]` only after it's implemented, tested, and verified - not just written.
- If a task is blocked, mark it `[ ] BLOCKED: <reason>` and move to the next independent task rather than stalling.
- If an implementation decision changes the architecture (a new component, a changed data flow, a new dependency), update `01-prd.md` - don't let architectural decisions live only in chat history or code comments.
- Any acceptance-criteria threshold that's currently a placeholder (e.g. "F1 >= 0.80, baseline, revisit in Phase 9") should be treated as provisional until real labeled data exists - don't hard-code final logic against a number that was flagged as a placeholder.
- Never hardcode `localhost`, `127.0.0.1`, or a Docker Compose service name when connecting to OpenSearch, Neo4j, or Redis - always read the connection details from configuration (FR27, NFR9), so the same code works against either a remote free-tier instance or a local Docker container.

## 8. Quick glossary

- **CRAG** - Corrective RAG: evidence grading + a bounded correction loop before generation.
- **RRF** - Reciprocal Rank Fusion, used to combine ranked lists from multiple retrievers.
- **TTFT** - Time to first token.
- **Trust tier** - classification of a source as `authoritative` (internal) vs `supplementary` (e.g. web), governing the Recovery Planner's policy gate.
- **Retrieval Plan** - the structured JSON object the Query Planner outputs, replacing the old fixed-class router output.

## 9. Open decisions still pending (not yet locked)

- Exact hosted-API vs self-hosted split for the reranker and grounding-check models (PRD leans hosted-first for speed, self-hosted for the evidence grader specifically since it's domain-specific).
- Compliance scope (PII/GDPR/HIPAA) - currently undetermined; baseline PII hygiene only until this is resolved (see `01-prd.md` Section 10).
- Specific provider choice for the free-tier hosted OpenSearch sandbox (candidates: Bonsai.io or equivalent) - not yet finalized.