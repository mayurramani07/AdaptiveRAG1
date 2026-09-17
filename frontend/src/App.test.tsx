import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { ApiError } from './services/api'

const { submitQuery, checkHealth, uploadDocument } = vi.hoisted(() => ({
  submitQuery: vi.fn(),
  checkHealth: vi.fn(),
  uploadDocument: vi.fn(),
}))

vi.mock('./services/api', async () => {
  const actual = await vi.importActual<typeof import('./services/api')>('./services/api')
  return { ...actual, submitQuery, checkHealth, uploadDocument }
})

beforeEach(() => {
  checkHealth.mockResolvedValue({ status: 'ok', dependencies: { redis: 'ok', neo4j: 'ok', opensearch: 'ok', groq: 'ok' } })
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('App', () => {
  it('renders the empty state on load', async () => {
    render(<App />)
    expect(screen.getByText(/AdaptiveRAG/)).toBeInTheDocument()
    expect(await screen.findByTestId('empty-state')).toBeInTheDocument()
  })

  it('shows the system-operational status once the health check resolves', async () => {
    render(<App />)
    expect(await screen.findByText(/system operational/i)).toBeInTheDocument()
  })

  it('shows backend-unavailable status when the health check fails', async () => {
    checkHealth.mockRejectedValue(new Error('network error'))
    render(<App />)
    expect(await screen.findByText(/backend unavailable/i)).toBeInTheDocument()
  })

  it('submits a query, shows loading, then renders the answer and sources', async () => {
    const user = userEvent.setup()
    submitQuery.mockImplementation(async (_body, callbacks) => {
      callbacks?.onSources?.([{ id: 'c1', doc_id: 'doc1', trust_tier: 'authoritative', score: 2.1, text: 'source snippet' }])
      callbacks?.onToken?.('the full answer')
      return {
        request_id: 'r1',
        answer: 'the full answer',
        sources: [{ id: 'c1', doc_id: 'doc1', trust_tier: 'authoritative', score: 2.1, text: 'source snippet' }],
        route: 'hybrid-rag',
        recovery_used: false,
        cache_hit: false,
        latency_ms: 100,
      }
    })

    render(<App />)
    await user.type(screen.getByLabelText(/query/i), 'What is Acme Corporation?')
    await user.click(screen.getByRole('button', { name: /ask/i }))

    expect(await screen.findByText('the full answer')).toBeInTheDocument()
    expect(await screen.findByTestId('source-card')).toHaveTextContent('doc1')
    expect(screen.getByText('source snippet')).toBeInTheDocument()
  })

  it('shows an error banner with a retry option when the query fails', async () => {
    const user = userEvent.setup()
    submitQuery.mockRejectedValue(new ApiError(401, 'invalid or missing API key'))

    render(<App />)
    await user.type(screen.getByLabelText(/query/i), 'hello')
    await user.click(screen.getByRole('button', { name: /ask/i }))

    expect(await screen.findByTestId('error-banner')).toHaveTextContent(/invalid or missing api key/i)

    submitQuery.mockResolvedValueOnce({
      request_id: 'r2',
      answer: 'recovered answer',
      sources: [],
      route: 'hybrid-rag',
      recovery_used: false,
      cache_hit: false,
      latency_ms: 90,
    })
    await user.click(screen.getByRole('button', { name: /retry/i }))
    expect(await screen.findByText('recovered answer')).toBeInTheDocument()
  })

  it('disables the submit button while a request is in flight', async () => {
    const user = userEvent.setup()
    let resolveFn: (value: unknown) => void = () => {}
    submitQuery.mockReturnValue(new Promise((resolve) => (resolveFn = resolve)))

    render(<App />)
    await user.type(screen.getByLabelText(/query/i), 'hello')
    await user.click(screen.getByRole('button', { name: /ask/i }))

    expect(screen.getByRole('button', { name: /asking/i })).toBeDisabled()

    resolveFn({ request_id: 'r3', answer: 'done', sources: [], route: 'hybrid-rag', recovery_used: false, cache_hit: false, latency_ms: 10 })
    await waitFor(() => expect(screen.getByRole('button', { name: /ask/i })).not.toBeDisabled())
  })

  it('uploads a document and shows a success message', async () => {
    const user = userEvent.setup()
    uploadDocument.mockResolvedValue({ doc_id: 'notes-abc123', filename: 'notes.txt', chunks_indexed: 3 })

    render(<App />)
    const file = new File(['hello world'], 'notes.txt', { type: 'text/plain' })
    const input = screen.getByLabelText(/document file/i)
    await user.upload(input, file)

    expect(await screen.findByTestId('upload-success')).toHaveTextContent('notes.txt')
    expect(await screen.findByTestId('upload-success')).toHaveTextContent('3 chunks')
  })

  it('shows an error message when a document upload fails', async () => {
    const user = userEvent.setup()
    uploadDocument.mockRejectedValue(new ApiError(413, 'document too large (9000 chars, max 8000)'))

    render(<App />)
    const file = new File(['x'], 'big.txt', { type: 'text/plain' })
    const input = screen.getByLabelText(/document file/i)
    await user.upload(input, file)

    expect(await screen.findByTestId('upload-error')).toHaveTextContent(/too large/i)
  })
})
