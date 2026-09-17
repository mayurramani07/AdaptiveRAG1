import { parseSSEStream } from '../lib/sse'
import { getStoredApiKey } from '../lib/storage'
import type { HealthResponse, QueryRequestBody, QueryResult, Source, UploadResult } from '../types/api'

/**
 * VITE_API_BASE_URL is the only source of the backend URL - never hardcode
 * localhost (docs/01-prd.md NFR-FE2). Vite only inlines VITE_-prefixed vars
 * into the client bundle at build time, so this is safe to read here.
 */
function getApiBaseUrl(): string {
  const base = import.meta.env.VITE_API_BASE_URL
  if (!base) {
    throw new ApiError(0, 'VITE_API_BASE_URL is not configured - set it in your .env.local (dev) or Vercel project settings (production).')
  }
  return base.replace(/\/$/, '')
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function extractErrorMessage(resp: Response): Promise<string> {
  try {
    const body = await resp.json()
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((d) => d?.msg ?? JSON.stringify(d)).join('; ')
  } catch {
    // response body wasn't JSON - fall through to a generic message
  }
  switch (resp.status) {
    case 401:
      return 'Invalid or missing API key. Check your API key in Settings.'
    case 413:
      return 'That document is too large to upload.'
    case 429:
      return 'Rate limit exceeded. Please wait a moment and try again.'
    case 503:
      return 'A required backend service is unavailable right now.'
    default:
      return `Request failed (HTTP ${resp.status}).`
  }
}

export interface StreamCallbacks {
  onSources?: (sources: Source[]) => void
  onToken?: (text: string) => void
  onDone?: (meta: { request_id: string; latency_ms: number; cache_hit: boolean; route: string; recovery_used: boolean }) => void
}

/**
 * Submits a query. The backend decides the actual response shape (buffered
 * JSON vs SSE) independent of the request's `stream` flag - Mode 2
 * (high-risk) always buffers regardless (see docs/01-prd.md 19.2/FR-FE5).
 * This function inspects Content-Type rather than assuming.
 */
export async function submitQuery(body: QueryRequestBody, callbacks: StreamCallbacks = {}): Promise<QueryResult> {
  const apiKey = getStoredApiKey()
  let response: Response
  try {
    response = await fetch(`${getApiBaseUrl()}/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError(0, 'Could not reach the backend. Check that it is running and VITE_API_BASE_URL is correct.')
  }

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response))
  }

  const contentType = response.headers.get('content-type') ?? ''
  if (contentType.includes('text/event-stream')) {
    return consumeSSE(response, callbacks)
  }
  const result = (await response.json()) as QueryResult
  callbacks.onSources?.(result.sources)
  callbacks.onToken?.(result.answer)
  return result
}

type DoneMeta = { request_id: string; latency_ms: number; cache_hit: boolean; route: string; recovery_used: boolean }

async function consumeSSE(response: Response, callbacks: StreamCallbacks): Promise<QueryResult> {
  let sources: Source[] = []
  let answer = ''
  let done: DoneMeta | null = null

  for await (const { event, data } of parseSSEStream(response)) {
    if (event === 'sources') {
      sources = (data as { sources: Source[] }).sources
      callbacks.onSources?.(sources)
    } else if (event === 'token') {
      const text = (data as { text: string }).text
      answer += text
      callbacks.onToken?.(text)
    } else if (event === 'done') {
      done = data as DoneMeta
      callbacks.onDone?.(done)
    }
  }

  if (!done) {
    throw new ApiError(0, 'Stream ended unexpectedly before completion.')
  }
  return { ...done, answer, sources }
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${getApiBaseUrl()}/v1/health`)
  if (!response.ok) {
    throw new ApiError(response.status, `Health check failed (HTTP ${response.status}).`)
  }
  return (await response.json()) as HealthResponse
}

/** Phase 11 (FR-FE10): uploads a single .pdf/.txt file to POST /v1/documents.
 * Blocking - the request only resolves once ingestion fully completes
 * (docs/01-prd.md Section 9.3 - there is no processing status to poll). */
export async function uploadDocument(file: File): Promise<UploadResult> {
  const apiKey = getStoredApiKey()
  const formData = new FormData()
  formData.append('file', file)

  let response: Response
  try {
    // No Content-Type header set manually - the browser fills in the
    // multipart boundary itself; setting it by hand breaks the upload.
    response = await fetch(`${getApiBaseUrl()}/v1/documents`, {
      method: 'POST',
      headers: { 'X-API-Key': apiKey },
      body: formData,
    })
  } catch {
    throw new ApiError(0, 'Could not reach the backend. Check that it is running and VITE_API_BASE_URL is correct.')
  }

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response))
  }
  return (await response.json()) as UploadResult
}
