import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { setStoredApiKey } from '../lib/storage'
import { ApiError, uploadDocument } from './api'

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

describe('uploadDocument', () => {
  it('posts the file as multipart form data and returns the parsed result', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ doc_id: 'notes-abc123', filename: 'notes.txt', chunks_indexed: 2 }))
    vi.stubGlobal('fetch', fetchMock)

    const file = new File(['hello world'], 'notes.txt', { type: 'text/plain' })
    const result = await uploadDocument(file)

    expect(result).toEqual({ doc_id: 'notes-abc123', filename: 'notes.txt', chunks_indexed: 2 })
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://api.test/v1/documents')
    expect(init.headers['X-API-Key']).toBe('test-key')
    expect(init.body).toBeInstanceOf(FormData)
  })

  it('throws ApiError with the backend detail on a 413 response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'document too large (9000 chars, max 8000)' }, 413)))
    const file = new File(['x'.repeat(9000)], 'big.txt', { type: 'text/plain' })

    await expect(uploadDocument(file)).rejects.toMatchObject({ status: 413 } satisfies Partial<ApiError>)
  })

  it('throws ApiError on a 400 unsupported-type response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'unsupported file type' }, 400)))
    const file = new File(['x'], 'image.png', { type: 'image/png' })

    await expect(uploadDocument(file)).rejects.toMatchObject({ status: 400, message: 'unsupported file type' } satisfies Partial<ApiError>)
  })
})
