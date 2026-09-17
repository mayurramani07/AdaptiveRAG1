/**
 * Types matching the REAL backend contract exactly (src/adaptive_rag/app.py,
 * src/adaptive_rag/generation.py - verified against source, not assumed;
 * see docs/01-prd.md Section 9.1). Do not add fields the backend doesn't
 * actually return.
 */

export interface Source {
  id: string | null
  doc_id: string | null
  trust_tier: string
  score: number | null
  /** Snippet of the source chunk's own text (Phase 10 addition). */
  text: string
}

export interface GroundingInfo {
  mode: 'sync' | 'async'
  status: 'pending' | 'passed' | 'failed'
  confidence?: number
}

/** The shape of a completed (buffered or fully-streamed) query result. */
export interface QueryResult {
  request_id: string
  answer: string
  sources: Source[]
  route: string
  recovery_used: boolean
  cache_hit: boolean
  latency_ms: number
  grounding?: GroundingInfo
}

export interface QueryRequestBody {
  query: string
  session_id?: string
  user_id?: string
  stream?: boolean
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  dependencies: Record<string, string>
}

/** Phase 11 (FR-ING5) - POST /v1/documents response. */
export interface UploadResult {
  doc_id: string
  filename: string
  chunks_indexed: number
}
