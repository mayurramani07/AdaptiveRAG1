import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { setStoredApiKey } from '../lib/storage'
import { ApiError, checkHealth, submitQuery } from './api'

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE_URL', 'http://api.test')
  setStoredApiKey('test-key')
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('submitQuery - buffered JSON response', () => {
  it('parses a buffered (non-streaming) response and reports sources/answer', async () => {
    const body = {
      request_id: 'r1',
      answer: 'the answer',
      sources: [{ id: 'c1', doc_id: 'd1', trust_tier: 'authoritative', score: 3.2, text: 'snippet' }],
      route: 'hybrid-rag',
      recovery_used: false,
      cache_hit: false,
      latency_ms: 120,
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(body)))

    const onSources = vi.fn()
    const onToken = vi.fn()
    const result = await submitQuery({ query: 'hello', stream: false }, { onSources, onToken })

    expect(result.answer).toBe('the answer')
    expect(result.sources).toHaveLength(1)
    expect(onSources).toHaveBeenCalledWith(body.sources)
    expect(onToken).toHaveBeenCalledWith('the answer')
  })
})

describe('submitQuery - SSE streaming response', () => {
  it('parses sources/token/done events from an SSE stream', async () => {
    const sse =
      'event: sources\ndata: {"sources":[{"id":"c1","doc_id":"d1","trust_tier":"authoritative","score":1.0,"text":"t"}]}\n\n' +
      'event: token\ndata: {"text":"Hel"}\n\n' +
      'event: token\ndata: {"text":"lo"}\n\n' +
      'event: grounding\ndata: {"mode":"async","status":"pending"}\n\n' +
      'event: done\ndata: {"request_id":"r2","latency_ms":50,"cache_hit":false,"route":"hybrid-rag","recovery_used":false}\n\n'

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse))
        controller.close()
      },
    })
    const response = new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))

    const tokens: string[] = []
    const result = await submitQuery({ query: 'hi', stream: true }, { onToken: (t) => tokens.push(t) })

    expect(tokens.join('')).toBe('Hello')
    expect(result.answer).toBe('Hello')
    expect(result.request_id).toBe('r2')
    expect(result.sources[0].doc_id).toBe('d1')
  })
})

describe('submitQuery - error handling', () => {
  it('throws ApiError with the backend detail message on a non-2xx response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'invalid or missing API key' }, 401)))

    await expect(submitQuery({ query: 'hi' })).rejects.toMatchObject({ status: 401, message: 'invalid or missing API key' } satisfies Partial<ApiError>)
  })

  it('throws a friendly ApiError when the network request fails entirely', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network error')))

    await expect(submitQuery({ query: 'hi' })).rejects.toThrow(/could not reach the backend/i)
  })
})

describe('checkHealth', () => {
  it('returns the parsed health payload', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok', dependencies: { redis: 'ok' } })))
    const health = await checkHealth()
    expect(health.status).toBe('ok')
  })
})
