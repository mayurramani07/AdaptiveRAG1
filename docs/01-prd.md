# PRD: Production-Grade Adaptive RAG System with CRAG Correction Layer

**Status:** Draft v4 (Docker made optional for local development)
**Deployment stance:** Free-tier only. The architecture below is designed to scale to 1k-50k req/day, but that range is an **architectural design target, not a guaranteed capacity of the current free-tier deployment**. All performance numbers in this document must be measured on the actual free-tier deployment, not assumed from the design target.
**Local development stance:** Docker is **optional, never required**. All service connections (OpenSearch, Neo4j, Redis) must be configurable via environment variables pointing to either a local or a remote endpoint - no code may assume `localhost` or Docker networking. The default recommended local setup uses **free-tier hosted remote instances** (no install, no virtualization needed): Neo4j AuraDB Free, Upstash Redis free tier, and a free-tier hosted OpenSearch sandbox (e.g. Bonsai.io). `infra/docker-compose.yml` remains available as an **alternative** for anyone who prefers local containers, but it is not on the default path.
**Stack:** Generation - `openai/gpt-oss-20b` via Groq. Deployment - Render (free tier) for backend orchestration, Vercel (free tier) optional for frontend/edge. Retrieval - OpenSearch (BM25 + Dense k-NN, free-tier hosted by default) + Neo4j (Graph, AuraDB free tier by default). Small models - mix of hosted free-tier APIs and self-hosted. Compliance - not yet determined; baseline PII hygiene only (see SS9).

---

## 1. Overview

Build a production-grade RAG pipeline that combines hybrid retrieval (Dense + BM25 + Graph), a query-planning routing layer, CRAG-style evidence correction with a governed recovery policy, a proper graph ingestion pipeline, and streaming generation - optimized for cost, latency, retrieval quality, and observability simultaneously, and deployed entirely on free-tier infrastructure.

**Core principle:** exactly **one expensive generation-model inference call per request in the default path**. (Note: we deliberately avoid the label "large/frontier LLM" - what matters architecturally is that there is exactly one costly generation call in the critical path, not how any specific model is marketed or how many parameters it has.) Every judgment/scoring step upstream (planning, reranking, evidence grading, grounding check) uses a small, cheap, fast model or a rule-based method - never the generation-tier model.

---

## 2. Full System Architecture

### 2.1 End-to-End Flow Diagram

```
                              +-------------+
                              |    USER     |
                              +------+------+
                                     v
                              +-------------+
                              | API Gateway |  (Render free-tier service: auth, rate limit, validation)
                              +------+------+
                                     v
                       +--------------------------+
                       |   Semantic Cache Check    |
                       | exact match + embedding   |  (Redis free tier)
                       +---------+--------+--------+
                                 |HIT      |MISS
                                 |         v
                                 |  +----------------------+
                                 |  | Query Understanding   |
                                 |  | (entities, intent,    |
                                 |  |  filter candidates)   |
                                 |  +-----------+-----------+
                                 |              v
                                 |  +----------------------+
                                 |  |   Query Planner        | (small multi-label classifier,
                                 |  +-----------+-----------+  NOT free-text LLM generation)
                                 |              v
                                 |  +----------------------+
                                 |  |   Retrieval Plan       | {dense, bm25, graph, freshness,
                                 |  |   (structured object)  |  apply_filters, top_k}
                                 |  +-----------+-----------+
                                 |              v
                                 |  +----------------------+
                                 |  |   Adaptive Router      | (executes the plan; clamps
                                 |  +-----------+-----------+  top_k regardless of plan value)
                                 |              v
                                 |   +----------+----------+--------------+
                                 |   v          v          v              v
                                 | (skip if  Dense       Dense+BM25   + Graph
                                 |  all off) only        (OpenSearch) (Neo4j, if plan.graph)
                                 |   |          |          |              |
                                 |   +----------+----+-----+--------------+
                                 |                   v
                                 |           +----------------+
                                 |           |  RRF Fusion     |
                                 |           +--------+-------+
                                 |                    v
                                 |           +----------------+
                                 |           |   Reranker      | (hosted free-tier API or
                                 |           |  top 20-30      |  self-hosted cross-encoder,
                                 |           |  (hard clamp)   |  clamped regardless of plan.top_k)
                                 |           +--------+-------+
                                 |                    v
                                 |           +----------------+
                                 |           |Evidence Grader  | (self-hosted small model - CRAG)
                                 |           +---+--------+----+
                                 |       Correct |        | Ambiguous / Incorrect
                                 |               |        v
                                 |               |  +--------------------------+
                                 |               |  |   Recovery Planner        |
                                 |               |  |  (policy-gated, max 2     |
                                 |               |  |   strategies attempted)   |
                                 |               |  +---+----+-----+-----------+
                                 |               |      |    |     |
                                 |               |  Query  Internal Graph   External Web*
                                 |               |  Rewrite Re-retr Expand  Search
                                 |               |      |    |     |         |
                                 |               |      +----+-----+---------+
                                 |               |               v
                                 |               |      (* only if no authoritative
                                 |               |         internal source scored even
                                 |               |         moderately - see SS2.6 trust tiers)
                                 |               v              v
                                 |           +------------------------+
                                 |           |  Context Builder        | (resolves any graph hits
                                 |           |                         |  back to source document
                                 |           |                         |  text - see SS3)
                                 |           +-----------+------------+
                                 |                       v
                                 |           +------------------------+
                                 |           |  Generation call         | <- THE ONE expensive
                                 |           |  gpt-oss-20b via Groq    |    generation call
                                 |           +-----------+------------+
                                 |                       |
                                 |            +----------+-----------+
                                 |            v                      v
                                 |   [DEFAULT / low-risk]     [HIGH-RISK category]
                                 |   Stream tokens to user    Buffer full response
                                 |   in parallel with:        internally, run BLOCKING
                                 |   async grounding check    grounding check first:
                                 |   (detect/flag/log only -  PASS -> release to user
                                 |    cannot stop an answer   FAIL -> return fallback
                                 |    already streaming)          response (not the
                                 |                                 ungrounded answer)
                                 |                       v
                                 +--------------->+----------------+
                                                  | Answer+Sources |
                                                  +-------+--------+
                                                          v
                                                     +---------+
                                                     |  USER   |
                                                     +---------+

        =============== OBSERVABILITY (runs alongside every stage) ===============
              Every stage -> Trace -> Metrics -> Logs -> Online Eval + Offline Eval (CI gate)
```

### 2.2 Cost/Model Tier Map

```
+---------------------------------------------------------------+
|  TIER 1 - No model, rule-based, near-free                     |
|  Semantic cache | Adaptive router | RRF fusion                |
+---------------------------------+-------------------------------+
                                   v
+---------------------------------------------------------------+
|  TIER 2 - Small model, cheap and fast (mix of hosted + self)   |
|  Query Planner | Reranker | Evidence grader | Recovery Planner |
|  Grounding check                                                |
+---------------------------------+-------------------------------+
                                   v
+---------------------------------------------------------------+
|  TIER 3 - The one expensive generation-model call               |
|  Final answer generation (gpt-oss-20b via Groq, streamed        |
|  by default / buffered for high-risk)                           |
+---------------------------------------------------------------+
```

### 2.3 Component Inventory & Tech Choices

| # | Component | Layer | Tech choice | Model tier | Blocking? |
|---|---|---|---|---|---|
| 1 | API Gateway | Ingress | Render free-tier web service | None | Yes |
| 2 | Semantic cache | Ingress | Redis free tier + embedding similarity | None | Yes |
| 3 | Query understanding | Planning | NER/intent tagger + filter-candidate extraction | None/small | Yes |
| 4 | Query Planner | Planning | Small multi-label classifier (self-hosted) | Small | Yes |
| 5 | Adaptive router | Planning | Executes plan, clamps top_k/filters | None | Yes |
| 6 | Dense retriever | Retrieval | OpenSearch k-NN | None | Yes (parallel) |
| 7 | BM25 retriever | Retrieval | OpenSearch BM25 | None | Yes (parallel) |
| 8 | Graph retriever | Retrieval | Neo4j Cypher traversal | None | Yes (parallel, only if plan.graph) |
| 9 | RRF fusion | Retrieval | Rule-based | None | Yes |
| 10 | Reranker | Retrieval | Hosted free-tier API or self-hosted cross-encoder | Small | Yes |
| 11 | Evidence grader | Correction (CRAG) | Self-hosted fine-tuned model (domain-specific) | Small | Yes |
| 12 | Recovery Planner | Correction (CRAG) | Small classifier + trust-tier config (rule-based gate) | Small/None | Yes, on bad evidence only |
| 13 | Web search fallback | Correction (CRAG) | External search API (free tier), PII-redacted queries | None | Only when policy-gated ON |
| 14 | Context builder | Generation | Rule-based; resolves graph hits to source text | None | Yes |
| 15 | Generation call | Generation | `openai/gpt-oss-20b` via Groq API | The one expensive call | Yes (streamed default / buffered high-risk) |
| 16 | Grounding check | Post-gen | Hosted free-tier NLI API or self-hosted small model | Small | Sync (high-risk, gates release) / Async (default, detector only) |
| 17 | Ingestion pipeline | Indexing (offline) | Chunking + extraction + resolution + validation (see SS3) | Small (extraction models) | N/A (offline batch) |
| 18 | Observability sidecar | Cross-cutting | OpenTelemetry + free-tier log aggregator | None | No |
| 19 | Online eval | Cross-cutting | Aggregation service | None | No |
| 20 | Offline eval / CI gate | Cross-cutting | Batch job in CI pipeline | None | No (pre-deploy) |

**Note on reranker/grounding "mix of hosted + self-hosted":** start with free-tier hosted APIs for reranker and grounding check to ship faster. The evidence grader must be self-hosted and fine-tuned on your own labeled data from day one - a generic hosted model cannot learn your domain's correctness signal.

**Note on Render vs Vercel:** Vercel serverless functions are optimized for short-lived edge/frontend logic, not a pipeline holding persistent connections to OpenSearch/Neo4j/Redis across a streaming request. Run the core orchestration backend on Render (free tier); use Vercel only for a separate frontend, if any.

### 2.4 Retrieval Plan Schema

Replaces the old fixed 5-class router output. The Query Planner is a **small multi-label classifier** producing a structured plan - not free text generated by a large model.

```json
{
  "dense": true,
  "bm25": true,
  "graph": false,
  "freshness": false,
  "apply_filters": false,
  "top_k": 20
}
```

Rules governing this plan:
- **The planner decides `apply_filters` as a boolean only.** It does not invent filter contents. Actual filter values (dates, document types, departments) come from the upstream Query Understanding step's entity/date extraction - the planner just decides whether to use them.
- **`top_k` is always clamped downstream** (hard cap of 30) regardless of what the planner outputs. Never trust the plan value blindly - this is defense in depth.
- **If all retrieval flags are false**, the query is treated as chitchat/meta and retrieval is skipped entirely (replaces the old `chitchat` class).
- **Combinatorial testing requirement:** with 3 boolean retrieval flags there are up to 8 reachable combinations (including "all off"). All 8 must have integration test coverage, not just the combinations that map to the old 5 classes.
- **Training data:** each dimension (`dense`, `bm25`, `graph`, `freshness`, `apply_filters`) needs independent ground-truth labels, not one categorical label per query. If bootstrapping from legacy 5-class labels, map each old class to its equivalent plan object as a starting point, then collect fresh per-dimension labels over time.

### 2.5 CRAG Recovery Policy

Replaces automatic web search on bad evidence. When the Evidence Grader returns Ambiguous or Incorrect, a **Recovery Planner** (small classifier + a rule-based trust gate) decides which recovery strategy to attempt - not a fixed pipe straight to the web.

**Recovery strategies, in order of preference:**
1. **Query Rewrite** - reformulate the query and re-run retrieval (cheapest, no new external calls).
2. **Internal Re-retrieval** - widen internal search (different retriever weights, larger candidate pool).
3. **Graph Expansion** - if Graph was used, expand traversal depth (e.g. 1-hop to 2-hop) before concluding the graph has nothing.
4. **External Web Search** - only attempted when the trust-tier gate below permits it.

**Trust-tier gate (must be explicit config, not a vague "if permitted" flag):**
```json
{
  "internal_policy_docs": "authoritative",
  "internal_knowledge_base": "authoritative",
  "web_search_results": "supplementary"
}
```
Rule: external web search may only be attempted if **no authoritative internal source scored even moderately** on the evidence grader. An authoritative internal source, even if imperfect, must never be silently overridden by a web result. Web results are always tagged `supplementary` in the final answer's source list, never presented at the same trust level as internal authoritative sources.

**Hard cap:** at most **2 recovery strategies attempted total** per request (not "try all 4 until something works"). If both attempts fail, return the "insufficient evidence" fallback response - same discipline as the original single-retry cap, just applied across a richer set of options.

**PII redaction:** any query text sent to the external web search step must be redacted per SS9 before it leaves the system boundary.

---

## 3. Indexing / Ingestion Architecture (Graph + Document Pipeline)

This section was previously undefined - the query-time diagram assumed Neo4j "just exists." It doesn't; it's built by this pipeline, and it must be re-run on the same scheduled cadence as document reindexing (see NG4).

```
Documents
   |
   v
Chunking
   |
   v
Entity Extraction        (identify people, orgs, policies, dates, etc.)
   |
   v
Entity Resolution        (dedup: "Apple Inc." and "Apple" -> one node)
   |
   v
Relationship Extraction  (infer edges: "reports to", "relates to", etc.)
   |
   v
Entity + Relationship Validation   (filter low-confidence/hallucinated edges
   |                                 before they enter the graph)
   v
Neo4j
```

**Critical rule - the graph is an index, not the evidence:**
- Every Neo4j node and edge relevant to a query result must carry a **pointer back to its source document + passage** (document ID, chunk ID, offset).
- The Graph retriever's job at query time is to find *which* documents/passages are relevant via relationship traversal - not to hand the LLM a paraphrased relationship label as if it were fact.
- The **Context Builder (FR16)** must resolve every graph hit back to the original source text before it enters the generation prompt. The LLM should ground its answer in real document text, not synthesized graph structure.
- The **Evidence Grader** grades the resolved source text, not the graph edge label - consistent with the rule above.

**Relationship extraction validation:** relationship extraction (LLM- or NER-based) is error-prone. The validation step must reject low-confidence or contradictory edges before they load into Neo4j - do not treat "the extractor said so" as sufficient to create a permanent graph edge.

**Freshness / re-ingestion policy:** ties to NG4 (no real-time index updates in v1). When a source document changes or is deleted, its derived graph nodes/edges are only updated on the same scheduled reindexing cadence as the rest of the index - document graph edges must not silently outlive a deleted or superseded source document past that cadence.

---

## 4. Goals

- **G1 - Retrieval quality:** Use a structured Retrieval Plan (SS2.4) to select the retriever combination best suited to each query, rather than a fixed enumeration of combinations.
- **G2 - Latency:** meet route-specific latency targets (SS5) as measured on the actual free-tier deployment.
- **G3 - Cost efficiency:** exactly one expensive generation-model inference call per request in the default path (SS1). All planning/grading/reranking/grounding done via small models or rules.
- **G4 - Correction / hallucination reduction:** CRAG-style evidence grading with a governed Recovery Policy (SS2.5) - internal recovery preferred, external web search only when policy-gated, capped at 2 total attempts.
- **G5 - Evaluation:** both online and offline evaluation loops exist from day one.
- **G6 - Observability:** every stage emits trace + metrics + logs; sampled at scale, full trace on errors/flagged cases.
- **G7 - Graph integrity:** the knowledge graph is built through a defined ingestion pipeline (SS3) and is always used as a routing/relationship index, never as a substitute for authoritative source documents.
- **G8 - No Docker dependency:** the application must run locally end-to-end using free-tier hosted remote services, with zero requirement for Docker or local virtualization.

## 5. Non-Goals

- **NG1:** Not aiming to eliminate hallucinations entirely - reduce and catch, not guarantee zero.
- **NG2:** Not a general-purpose agent framework or multi-turn planning system - single-turn (or simple follow-up) RAG only.
- **NG3:** Not training foundation models from scratch - small models are fine-tuned/distilled, generation uses an existing hosted model.
- **NG4:** No real-time index updates in v1 - scheduled reindexing only, and this cadence governs both the document index and the derived knowledge graph (SS3).
- **NG5:** Not building a custom vector/graph database engine - OpenSearch and Neo4j are used as-is.
- **NG6:** No multi-language support in v1.
- **NG7:** No UI/frontend build in this PRD - backend pipeline and API only.
- **NG8:** No user-uploaded/on-the-fly document ingestion in v1 - knowledge base is pre-indexed via the pipeline in SS3.
- **NG9:** No formal compliance certification (HIPAA/GDPR/SOC2) in v1 - baseline-only approach (SS9), explicit revisit trigger before regulated data.
- **NG10:** This project does NOT guarantee production-scale capacity (1k-50k req/day) on its free-tier deployment. That range is the architecture's design target for a future paid tier, not a claim about what the current free-tier deployment can sustain. No paid infrastructure is designed or costed in this version.
- **NG11:** Docker is not a dependency of this project. `infra/docker-compose.yml` is an optional convenience for contributors who have Docker available - it is never a required step, and no part of the application, tests, or CI may assume it is present.

## 6. Success Metrics (Route-Specific, Measured on Free-Tier Deployment)

The old single blended "p50 < 500ms" target hid enormous variance between a cache hit and a full CRAG-recovery path. Replaced with per-route targets, each tracked as p50/p95/p99:

| Route | p50 target | p95 target | p99 target | Notes |
|---|---|---|---|---|
| Cache HIT | < 100ms | < 200ms | < 400ms | Vulnerable to Render free-tier cold starts - measure and report honestly |
| Simple RAG (dense only) | < 500ms | < 900ms | < 1.5s | |
| Hybrid RAG (dense+BM25) | < 800ms | < 1.3s | < 2s | |
| Graph RAG (+ Neo4j) | < 1s | < 1.6s | < 2.5s | Neo4j free-tier query latency is a known variable |
| CRAG recovery | < 2s | < 3.5s | not bounded* | *p99 may exceed target under external web-search dependency slowness - p50/p95 should still hold |

Blended headline number (`overall p50 < 500ms`) may still be reported to stakeholders but must be labeled explicitly as a **blended, non-SLA figure** - the per-route table above is the actual engineering target.

**All numbers in this table must be measured on the real free-tier deployment**, not projected from the design target scale. If free-tier constraints (cold starts, AuraDB node caps, shared CPU) make a target unreachable, that is a documented finding, not a target to quietly redefine.

| Other metric | Target |
|---|---|
| Cache hit rate | Tracked; no hard target |
| Generation-model calls per request | Exactly 1 (default path) |
| Grounding check pass rate | Tracked; alert if it drops below baseline set during Phase 9 tuning |
| CRAG recovery trigger rate | Tracked; alert if consistently high |
| Recovery strategy distribution (rewrite vs re-retrieval vs graph expansion vs web) | Tracked - a route relying heavily on web search signals a retrieval/ingestion quality gap |
| Planner per-dimension accuracy | Tracked per flag (dense/bm25/graph/freshness/apply_filters), not one blended accuracy number |

---

## 7. Functional Requirements by Layer (with Acceptance Criteria)

### 7.1 Ingress Layer
- **FR1:** API Gateway handles auth, rate limiting, request validation.
  - *Acceptance:* Unauthenticated/malformed requests rejected with correct 4xx codes in 100% of test cases.
- **FR2:** Semantic cache checks exact string match + embedding similarity before falling through.
  - *Acceptance:* False-positive HIT rate < 1% on a labeled paraphrase test set.
- **FR3:** On cache HIT, return immediately - no downstream stage executes.
  - *Acceptance:* Trace confirms zero calls to planner/retrievers/generation on a HIT.

### 7.2 Query Understanding & Planning
- **FR4:** Extract entities, intent tags, and filter candidates (dates, document types, departments).
  - *Acceptance:* Entity extraction F1 >= 0.85 on a labeled validation set (baseline, revisit after real data exists).
- **FR5:** Query Planner outputs a structured Retrieval Plan per SS2.4 (`dense`, `bm25`, `graph`, `freshness`, `apply_filters`, `top_k`) via a small multi-label classifier - not free-text generation from a large model.
  - *Acceptance:* Per-flag accuracy >= 90% against a held-out labeled set, measured independently per dimension, before go-live.
- **FR6:** Adaptive Router executes the plan: fires only the retrievers flagged true, applies filters only if `apply_filters` is true (using upstream-extracted values), and clamps `top_k` to the hard cap regardless of the planner's output.
  - *Acceptance:* Integration tests cover all 8 reachable flag combinations (SS2.4), 100% correct retriever set invoked in each case. A test asserting a plan requesting `top_k: 500` is clamped to the configured cap must pass.
- **FR7:** Plan decisions logged with grounding/quality scores for retraining.
  - *Acceptance:* Every request's plan + final grounding score queryable by request ID within 5 minutes.

### 7.3 Retrieval & Fusion
- **FR8:** Dense (OpenSearch k-NN), BM25 (OpenSearch), Graph (Neo4j) retrievers run in parallel when multiple are flagged true.
  - *Acceptance:* Load test confirms parallel execution - total retrieval latency approx max(individual retriever latency), not sum.
- **FR9:** RRF fusion combines results from fired retrievers.
  - *Acceptance:* Fusion output is deterministic given identical inputs.
- **FR10:** Reranker re-scores only the clamped top-N candidates (default 20-30).
  - *Acceptance:* Reranker never receives more than the configured cap - enforced in code, verified by test.

### 7.4 Correction Layer (CRAG + Recovery Policy)
- **FR11:** Evidence grader scores evidence as Correct/Ambiguous/Incorrect using tuned thresholds, grading resolved source text (not graph edge labels - see SS3).
  - *Acceptance:* Grader F1 >= 0.80 against a labeled relevance dataset (baseline, revisit in Phase 9).
- **FR12:** On Correct, evidence passes directly to Context Builder.
  - *Acceptance:* Verified by unit test with synthetic graded input.
- **FR13:** On Ambiguous/Incorrect, the Recovery Planner selects from {Query Rewrite, Internal Re-retrieval, Graph Expansion, External Web Search} per the trust-tier gate in SS2.5.
  - *Acceptance:* Unit tests confirm External Web Search is never attempted while an authoritative internal source still scores at or above the moderate threshold.
- **FR14:** Recovery is capped at 2 total strategy attempts; failing both, return the "insufficient evidence" fallback.
  - *Acceptance:* Chaos test forcing the grader to always return Incorrect confirms exactly 2 recovery attempts occur, then a fallback response is returned - no infinite loop, no crash.
- **FR15:** Web search fallback queries are PII-redacted before leaving the system boundary.
  - *Acceptance:* Automated scan of outbound web-search request logs confirms no unredacted PII patterns present.

### 7.5 Ingestion Pipeline
- **FR-ING1:** Documents are chunked, entities extracted, entities resolved (deduplicated), relationships extracted, and both entities and relationships validated before loading into Neo4j (SS3).
  - *Acceptance:* A test document with a known duplicate entity (e.g. "Apple Inc." and "Apple" in different chunks) resolves to a single graph node.
- **FR-ING2:** Every graph node/edge carries a pointer to its source document + passage.
  - *Acceptance:* 100% of graph nodes queried in an integration test resolve back to valid source text.
- **FR-ING3:** Relationship extraction validation rejects low-confidence/contradictory edges before they load into Neo4j.
  - *Acceptance:* A deliberately low-confidence extracted relationship in a test fixture is rejected and does not appear in the graph.
- **FR-ING4:** Re-ingestion follows the same scheduled cadence as the rest of the index (NG4); stale graph edges from deleted/changed source documents do not persist past that cadence.
  - *Acceptance:* A document deleted and a subsequent scheduled reindex run together remove its derived graph nodes/edges.

### 7.6 Generation
- **FR16:** Context builder assembles the prompt from graded evidence only, resolving any graph-derived evidence back to its original source document text (SS3) before inclusion.
  - *Acceptance:* Prompt inspection in test confirms no graph edge label is included as evidence without its resolved source text, and no evidence graded Incorrect is included.
- **FR17:** `gpt-oss-20b` (via Groq) streams tokens to the client by default.
  - *Acceptance:* Time-to-first-token measured per route per SS6, on the actual free-tier deployment.
- **FR18:** Generation is the only stage permitted to make the one expensive generation-model call.
  - *Acceptance:* Automated lint/code-scan check confirms no other module invokes the Groq generation endpoint or any equivalent generation-tier API.

### 7.7 Post-Generation Checks (Two Explicit Modes)
- **FR19 (Mode 1 - Default, low/medium-risk):** Answer streams to the user immediately; grounding check runs asynchronously in parallel. This mode is a **detector, not a gate** - it can detect, flag, log, and feed the eval loop, but it cannot prevent an already-streaming answer from reaching the user.
  - *Acceptance:* Load test confirms response latency for default-mode queries is unaffected by grounding check completion time; a flagged-but-already-delivered answer is correctly logged for review.
- **FR20 (Mode 2 - High-risk categories: medical, legal, financial):** The full response is buffered internally (not streamed) while a synchronous grounding check runs. PASS releases the buffered response to the user; FAIL returns the fallback response instead of the ungrounded answer.
  - *Acceptance:* No high-risk query result reaches the user without a passed grounding check, verified by test. A test forcing FAIL confirms the fallback response is returned and the ungrounded buffered answer is never sent.
- **FR21:** High-risk category detection determines routing between Mode 1 and Mode 2 before generation begins.
  - *Acceptance:* Category classifier test set confirms correct mode selection for known high-risk query patterns.

### 7.8 Observability & Evaluation
- **FR22-26:** Per SS2 architecture - tracing, metrics, online eval, offline eval, drift detection.
  - *Acceptance:* A deliberately regressed component (e.g. reranker returning random order) fails the offline eval CI gate and blocks deployment.

### 7.9 Local Development Connectivity
- **FR27:** All connections to OpenSearch, Neo4j, and Redis are established using host, port, and credential values read from configuration (environment variables) - never hardcoded to `localhost` or any Docker-internal hostname.
  - *Acceptance:* A code-scan/lint check confirms no source file contains a hardcoded `localhost`, `127.0.0.1`, or Docker Compose service name for these three connections. Switching `.env` from a remote endpoint to a local Docker endpoint (or vice versa) requires no code change, only a config change.
- **FR28:** The application starts and runs its full request path successfully against free-tier hosted remote instances of OpenSearch, Neo4j, and Redis, with no local Docker containers running.
  - *Acceptance:* A documented local run (see Section 11.1) completes a full query end-to-end using only remote free-tier services.

---

## 8. Non-Functional Requirements

- **NFR1 (Timeouts):** Every external call has an explicit timeout with defined degradation (e.g. Neo4j timeout -> proceed with OpenSearch results only).
- **NFR2 (Circuit breakers):** Each dependency (OpenSearch, Neo4j, Redis, Groq, web search API) has a circuit breaker.
- **NFR3 (No unbounded loops):** CRAG recovery is capped at 2 total strategy attempts (SS2.5); this cap is enforced in code, not just by convention.
- **NFR4 (Cost ceiling):** No stage other than the single generation call may invoke a generation-tier model in the default path - defined by role in the pipeline, not by parameter count or marketing label.
- **NFR5 (Statelessness):** The orchestration service should be stateless where possible (session state in Redis) to support scaling beyond free tier later.
- **NFR6 (Plan trust boundary):** Any value produced by the Query Planner (top_k, apply_filters) must be validated/clamped downstream before use - the plan is never trusted blindly.
- **NFR7 (Streaming/blocking boundary):** Once tokens begin streaming to a user (Mode 1), the system must not attempt to retract or edit already-sent tokens. High-risk queries (Mode 2) must be identified and routed to buffered mode *before* generation begins, not mid-stream.
- **NFR8 (Free-tier honesty):** No performance or capacity claim in this document may be asserted without being measured on the actual free-tier deployment. Design-target capacity (1k-50k req/day) must always be labeled as such, never presented as current capacity.
- **NFR9 (Docker independence):** No application code, test, or CI step may assume Docker is installed or running. All infrastructure connectivity is configuration-driven (FR27); `infra/docker-compose.yml` is an optional alternative path, never a prerequisite.

---

## 9. API Contract

### 9.1 `POST /v1/query`

**Request:**
```json
{
  "query": "string, required",
  "session_id": "string, optional",
  "user_id": "string, optional",
  "stream": true
}
```

**Response (streaming, Server-Sent Events) - Mode 1 (default):**
```
event: token
data: {"text": "partial answer chunk"}

event: sources
data: {"sources": [{"id": "doc_123", "title": "...", "trust_tier": "authoritative", "score": 0.87}]}

event: grounding
data: {"mode": "async", "status": "pending" | "passed" | "flagged", "confidence": 0.91}

event: done
data: {"request_id": "uuid", "latency_ms": 842, "cache_hit": false, "route": "hybrid-rag", "recovery_used": false}
```

**Response (non-streaming until release) - Mode 2 (high-risk):**
```json
{
  "request_id": "uuid",
  "answer": "full text, released only after grounding PASS",
  "sources": [{"id": "doc_456", "trust_tier": "authoritative"}],
  "grounding": {"mode": "sync", "status": "passed", "confidence": 0.94},
  "route": "graph-rag",
  "recovery_used": true,
  "latency_ms": 1840
}
```

**Error responses:**
| Code | Meaning |
|---|---|
| 400 | Invalid/empty query |
| 401 | Auth failure |
| 429 | Rate limit exceeded |
| 503 | Downstream dependency unavailable, no graceful degradation possible |
| 504 | Pipeline exceeded max allowed latency |

### 9.2 `GET /v1/health`
Returns per-dependency health (OpenSearch, Neo4j, Redis, Groq reachability) - regardless of whether each dependency is a local Docker container or a remote free-tier instance.

---

## 10. Security & Privacy

- **Auth:** token-based (API key or JWT) on all requests.
- **PII handling (baseline, compliance TBD):**
  - Redact common PII patterns from logs and traces before storage.
  - Redact PII from outbound web search queries during CRAG recovery (SS2.5, FR15).
  - Entities extracted during ingestion (SS3) may contain PII (names, contact info) - apply the same redaction policy to logged extraction outputs.
  - Cache keys hashed, not stored as raw plaintext query text beyond a 30-day default retention window.
- **Explicit revisit trigger:** before onboarding healthcare, financial, or EU-resident user data, re-scope this section for HIPAA/GDPR/SOC2 as applicable.
- **Secrets management:** Groq API key, OpenSearch/Neo4j/Redis credentials stored in `.env` locally (never committed - see `.gitignore`) and in Render's environment/secrets manager in deployment. The same variable names must work whether the value points at a local Docker container or a remote free-tier instance.

---

## 11. Deployment (Free-Tier Only)

There is a single deployment target for this project: **free-tier infrastructure end to end.** No paid production tier is designed or costed in this version.

| Component | Free-tier choice (default, remote/hosted) | Alternative (local, Docker - optional) |
|---|---|---|
| Backend orchestration | Render free web service | N/A |
| Dense + BM25 | Free-tier hosted OpenSearch sandbox (e.g. Bonsai.io) | Self-hosted OpenSearch via `infra/docker-compose.yml` |
| Graph | Neo4j AuraDB free tier (mind node/relationship caps) | Neo4j Community Edition via `infra/docker-compose.yml` |
| Cache | Upstash Redis free tier | Redis via `infra/docker-compose.yml` |
| Generation | Groq (usage-based, cheap at low volume - not tier-gated) | N/A |
| Reranker/Grounding | Free-tier hosted APIs where available, else self-hosted small models | Same |
| Web search fallback | Free-tier search API | N/A |
| Frontend (optional) | Vercel free tier | N/A |

### 11.1 Local development - Docker is optional, never required

Docker/virtualization is not assumed anywhere in this project. All service connections are configured via environment variables (FR27), and either path below works without any code change:

- **Default (recommended, no install needed - use this on machines where Docker/virtualization is restricted):** point `.env` at free-tier hosted remote instances - Neo4j AuraDB Free, Upstash Redis free tier, a free-tier hosted OpenSearch sandbox. Nothing runs on the local machine except the application itself.
- **Alternative (for contributors who have Docker available):** run `infra/docker-compose.yml` to spin up local containers for OpenSearch/Neo4j/Redis, and point `.env` at `localhost` instead.

**Known free-tier constraints to expect and document, not hide:**
- Render free-tier services sleep on inactivity - cold starts will affect the Cache HIT <100ms target; report the real number.
- AuraDB free tier caps node/relationship counts - ingestion pipeline (SS3) must respect this limit and document what fraction of the intended knowledge base actually fits.
- Free-tier hosted OpenSearch sandboxes and Upstash Redis may have their own request-rate or storage caps - document these once selected.
- Shared/limited CPU on free tiers will affect reranker/grader inference latency versus a dedicated instance.

**Feature flags:** Graph retrieval, web search fallback, and Mode 2 blocking grounding should each be independently toggleable, so a failing free-tier dependency (e.g. AuraDB rate limit hit) can be disabled without a redeploy.

**No canary/paid-tier rollout process** is defined in this version - single free-tier environment, deploy directly, monitor via observability (SS2). Canary/staged rollout is deferred to a future paid-tier phase, not designed here.

---

## 12. Cost Estimate (Free-Tier Deployment)

| Component | Cost |
|---|---|
| Render, Upstash Redis, hosted OpenSearch sandbox, Neo4j AuraDB free tier, Vercel | $0/month baseline, subject to free-tier limits above |
| Groq generation (`gpt-oss-20b`) | Usage-based, real cost - not free-tier gated. At ~2,000 input / ~400 output tokens per request (~$0.075/1M in, ~$0.30/1M out, verify current pricing at implementation time): approx **$0.00027 per request** |
| Reranker/grounding hosted APIs (if free tiers are exceeded) | Track actual usage; free-tier quotas may cap real throughput before cost becomes an issue |
| Web search fallback API | Free-tier quota; track recovery-trigger rate (SS6) since this is the metric that predicts when this quota gets exhausted |

**No paid production infrastructure cost is estimated or invented in this document.** If the free-tier deployment's actual sustainable request volume is meaningfully below the 1k-50k/day design target (very likely, given the constraints above), that gap must be reported honestly as a finding from Phase 9 testing, not smoothed over.

---

## 13. Ownership & External Dependencies

| Dependency | Type | Fallback if fully down |
|---|---|---|
| Groq (`gpt-oss-20b`) | Generation | No fallback in v1 - return 503 |
| OpenSearch | Dense + BM25 | Degrade to Neo4j-only if plan allows, else 503 |
| Neo4j | Graph | Degrade to OpenSearch-only (skip Graph), log degraded mode |
| Redis | Cache | Bypass cache, proceed to full pipeline |
| Web search API | CRAG recovery | Skip that recovery strategy, fall through to remaining recovery options or fallback response |
| Hosted reranker/grounding API | Small model | Fall back to self-hosted equivalent, or skip with a logged warning |

---

## 14. Model Lifecycle & Retraining

- Query Planner, Evidence Grader, Recovery Planner, and any self-hosted reranker/grounding models are versioned artifacts.
- **Planner retraining note:** since the planner now predicts multiple independent flags (SS2.4), retraining/evaluation must track per-flag accuracy, not one blended number - a regression in the `graph` flag alone could hide behind a healthy overall accuracy score.
- Retrain trigger: drift detection firing, or per-flag/grader accuracy dropping below threshold in online eval over a rolling window.
- Maintain a labeled evaluation dataset that grows from flagged/reviewed production queries (PII-redacted per SS10) - the same dataset the offline eval CI gate runs against.

---

## 15. Key Tunable Thresholds

- [ ] Semantic cache similarity threshold
- [ ] Query Planner per-flag confidence thresholds (dense/bm25/graph/freshness/apply_filters)
- [ ] `top_k` hard clamp value (default 30)
- [ ] Evidence grader Correct/Ambiguous/Incorrect score boundaries
- [ ] Recovery Planner trust-tier "moderate score" threshold (below which external web search becomes eligible)
- [ ] Max recovery strategy attempts (default 2)
- [ ] Grounding check pass/fail threshold (separately tunable for Mode 1 vs Mode 2, if needed)
- [ ] High-risk category list (governs Mode 1 vs Mode 2 routing)

---

## 16. Definition of Done (v1, Free-Tier Deployment)

v1 is considered shipped when, on the actual free-tier deployment:
- Route-specific latency (SS6) has been measured (not assumed) for at least a 1-2 week soak at whatever request volume the free tier can honestly sustain - document the sustainable volume found, do not assume 25k/day.
- Exactly one generation-model call occurs per request in 100% of sampled traces.
- Offline eval CI gate has run on every merge with no unreviewed regressions.
- CRAG recovery cap (2 strategies max) passes chaos testing - no infinite loop, no crash.
- Query Planner per-flag accuracy and evidence grader accuracy meet the baselines in SS7's acceptance criteria.
- Ingestion pipeline (SS3) validated end-to-end on a test document set, with graph nodes correctly resolving to source text.
- Mode 1 / Mode 2 grounding split verified: async never blocks, sync never releases an unpassed answer.
- Security baseline (SS10) implemented and PII redaction verified on a log sample.
- Free-tier constraints and their measured impact (cold starts, node caps, etc.) are documented as findings, not omitted.
- The application has been run and verified end-to-end with zero local Docker containers running, using only free-tier hosted remote services (FR28).

---

## 17. Implementation Checklist for AI Agent

**Instructions for the agent:** Work through tasks in order, phase by phase. Mark a task `[x]` only after it is implemented, tested, and verified working - do not proceed to the next task with an unchecked box above it unless explicitly told to skip. If a task is blocked, mark it `[ ] BLOCKED: <reason>` and move to the next independent task, then return.

### Phase 0 - Setup
- [x] Set up project repo structure, config management, environment variables
- [ ] BLOCKED: Provision Render free-tier service — requires manual Render dashboard signup; Vercel free-tier project if frontend needed
- [x] Provision OpenSearch and Neo4j connectivity — config fields exist in `config.py` reading from `.env` (host/creds only, no client wired yet); real connectivity happens when the OpenSearch/Neo4j clients are built in Phase 2/4
- [x] Provision Redis — config field exists in `config.py` reading from `.env`; real client wired in Phase 1 (cache)
- [x] Set up logging/tracing infrastructure before building pipeline logic — JSON logging in `src/adaptive_rag/logging.py`; distributed tracing deferred to Phase 8
- [ ] BLOCKED: Store Groq API key and DB credentials in Render's secrets manager — requires Render dashboard access
- [x] Audit `config.py` and any client initialization code to confirm no hardcoded `localhost`/Docker-service-name connections remain (FR27) — verified by `tests/test_config.py`

### Phase 1 - Ingress & Cache
- [x] Implement API Gateway with auth + rate limiting (FR1)
- [x] Implement exact-match + semantic cache lookup (FR2) — exact match via Redis with hashed keys; semantic lookup config-gated on `EMBEDDING_PROVIDER` (no provider wired yet, pending decision in project context SS9)
- [x] Verify: cache HIT returns without invoking any downstream stage (FR3)

### Phase 2 - Ingestion Pipeline (build before query-time Graph retrieval)
- [ ] Implement document chunking
- [ ] Implement entity extraction
- [ ] Implement entity resolution (dedup)
- [ ] Implement relationship extraction
- [ ] Implement entity + relationship validation (reject low-confidence edges)
- [ ] Load validated graph into Neo4j with source document/passage pointers on every node/edge (FR-ING1, FR-ING2)
- [ ] Implement scheduled re-ingestion tied to document reindex cadence (FR-ING4)
- [ ] Verify: a test document with duplicate entity mentions resolves to one graph node (FR-ING1 acceptance)

### Phase 3 - Query Understanding & Planning
- [ ] Implement entity/intent/filter-candidate extraction (FR4)
- [ ] Train/integrate the Query Planner as a multi-label classifier outputting the Retrieval Plan schema (FR5, SS2.4)
- [ ] Implement Adaptive Router executing the plan, with top_k/filter clamping (FR6, NFR6)
- [ ] Log plan decisions with request ID (FR7)
- [ ] Write integration tests covering all 8 reachable plan combinations
- [ ] Verify: a plan requesting top_k above the cap is clamped

### Phase 4 - Retrieval & Fusion
- [ ] Implement Dense (OpenSearch k-NN), BM25 (OpenSearch), Graph (Neo4j) retrievers
- [ ] Implement parallel execution when multiple flags are true (FR8)
- [ ] Implement RRF fusion (FR9)
- [ ] Integrate reranker, capped at top 20-30 (FR10)
- [ ] Add per-retriever timeout + circuit breaker (NFR1, NFR2)
- [ ] Verify: correct retriever subset fires per plan

### Phase 5 - CRAG Correction & Recovery Policy
- [ ] Train/fine-tune self-hosted evidence grader (FR11), grading resolved source text not graph labels
- [ ] Implement Recovery Planner with the 4 strategies and trust-tier gate (FR13, SS2.5)
- [ ] Implement hard cap of 2 total recovery attempts (FR14)
- [ ] Implement PII redaction on outbound web search queries (FR15)
- [ ] Chaos-test: grader always returns Incorrect -> confirm exactly 2 recovery attempts then fallback, no loop

### Phase 6 - Generation
- [ ] Implement Context Builder resolving graph hits to source text before prompting (FR16)
- [ ] Integrate `gpt-oss-20b` generation call via Groq
- [ ] Implement SSE streaming for Mode 1 (default)
- [ ] Implement response buffering for Mode 2 (high-risk) per NFR7
- [ ] Audit: confirm generation call is the only generation-tier invocation in the codebase (FR18)

### Phase 7 - Grounding (Two Modes)
- [ ] Implement async, non-blocking grounding check for Mode 1 (FR19)
- [ ] Implement synchronous, blocking grounding check for Mode 2 (FR20)
- [ ] Implement high-risk category detection routing to the correct mode before generation begins (FR21)
- [ ] Verify: Mode 1 never delays response; Mode 2 never releases an unpassed answer

### Phase 8 - Observability & Evaluation
- [ ] Instrument every stage with trace + metrics + structured logs
- [ ] Implement PII redaction on logs/traces (SS10)
- [ ] Build online eval collection (grounding scores, feedback)
- [ ] Build offline eval suite (golden-set queries, per-flag planner accuracy, grader accuracy)
- [ ] Wire offline eval into CI/CD as a deploy gate
- [ ] Implement drift detection for embedding/index/graph staleness

### Phase 9 - Free-Tier Measurement & Hardening
- [ ] Run offline eval to tune all thresholds in SS15
- [ ] Load test each route separately (SS6 table) on the actual free-tier deployment; record real p50/p95/p99 per route
- [ ] Document free-tier constraints found (cold starts, node caps, shared CPU impact) as explicit findings
- [ ] Reverify current Groq pricing against SS12 estimate
- [ ] Chaos-test: kill Neo4j mid-request, confirm graceful degradation
- [ ] Verify Definition of Done (SS16) over the soak period at the actual sustainable free-tier volume
- [ ] Verify FR28: run the full application against only remote free-tier services with zero local Docker containers running
- [ ] Final review against Non-Goals (SS5) - confirm none have been inadvertently built, including NG10 (no paid infra claims) and NG11 (no Docker dependency)

---

## 18. Open Risks / Tradeoffs

- Query Planner and Evidence Grader quality is bounded by training data - misclassification on any single flag can silently degrade retrieval quality; per-flag monitoring (SS6) exists specifically to catch this.
- The Recovery Planner's trust-tier gate is only as good as its configuration - if internal sources are mis-tagged as `supplementary`, the "never override authoritative internal evidence" guarantee breaks silently. Config for this must be reviewed, not just set once.
- Graph ingestion quality depends entirely on relationship extraction validation (SS3) - an under-validated pipeline will populate Neo4j with confident-sounding but wrong edges, which is worse than no graph at all.
- Mode 2 (buffered, blocking grounding) trades away streaming latency for safety on high-risk queries by design - this is an accepted tradeoff, not an oversight.
- **Free-tier capacity is the single biggest unknown in this project.** The 1k-50k req/day design target may not be realistically testable on free infrastructure; Phase 9 exists specifically to surface this honestly rather than assume the architecture's theoretical scalability equals the deployment's actual capacity.
- No fallback generation provider is defined if Groq has an outage - accepted risk for v1; revisit if/when moving beyond free tier.
- Compliance scope remains undefined (SS10) - a real risk if the user base later includes regulated data.
- Free-tier hosted OpenSearch/Redis sandboxes (chosen to avoid Docker) may have tighter storage/throughput caps than a locally-run Docker instance would - this is a deliberate tradeoff (no virtualization required vs. slightly lower local capacity) and should be documented once specific providers are selected.