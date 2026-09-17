import { useCallback, useRef, useState } from 'react'
import { ApiError, submitQuery } from '../services/api'
import type { QueryResult, Source } from '../types/api'

export type AskStatus = 'idle' | 'loading' | 'streaming' | 'success' | 'error'

export interface AskState {
  status: AskStatus
  answer: string
  sources: Source[]
  meta: Pick<QueryResult, 'route' | 'recovery_used' | 'cache_hit' | 'latency_ms' | 'grounding'> | null
  error: string | null
}

const initialState: AskState = { status: 'idle', answer: '', sources: [], meta: null, error: null }

export function useAskQuery() {
  const [state, setState] = useState<AskState>(initialState)
  const lastQueryRef = useRef<string>('')

  const ask = useCallback(async (query: string) => {
    lastQueryRef.current = query
    setState({ status: 'loading', answer: '', sources: [], meta: null, error: null })

    try {
      const result = await submitQuery(
        { query, stream: true },
        {
          onSources: (sources) => setState((prev) => ({ ...prev, status: 'streaming', sources })),
          onToken: (text) => setState((prev) => ({ ...prev, status: 'streaming', answer: prev.answer + text })),
        },
      )
      setState({
        status: 'success',
        answer: result.answer,
        sources: result.sources,
        meta: { route: result.route, recovery_used: result.recovery_used, cache_hit: result.cache_hit, latency_ms: result.latency_ms, grounding: result.grounding },
        error: null,
      })
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.'
      setState({ status: 'error', answer: '', sources: [], meta: null, error: message })
    }
  }, [])

  const retry = useCallback(() => {
    if (lastQueryRef.current) ask(lastQueryRef.current)
  }, [ask])

  const reset = useCallback(() => setState(initialState), [])

  return { state, ask, retry, reset }
}
