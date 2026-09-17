import { useEffect, useState } from 'react'
import { checkHealth } from '../services/api'
import type { HealthResponse } from '../types/api'

const POLL_INTERVAL_MS = 60_000

export type HealthState = { status: 'checking' | 'ok' | 'degraded' | 'unreachable'; dependencies?: Record<string, string> }

/** Polls GET /v1/health on mount and every 60s - PRD FR-FE7: infrequent,
 * never on every keystroke/render. */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ status: 'checking' })

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const result: HealthResponse = await checkHealth()
        if (!cancelled) setState({ status: result.status, dependencies: result.dependencies })
      } catch {
        if (!cancelled) setState({ status: 'unreachable' })
      }
    }

    poll()
    const id = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return state
}
