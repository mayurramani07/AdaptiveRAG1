import { useHealth } from '../hooks/useHealth'

const STYLES: Record<string, { dot: string; label: string }> = {
  checking: { dot: 'bg-gray-400 animate-pulse', label: 'Checking…' },
  ok: { dot: 'bg-emerald-500', label: 'System operational' },
  degraded: { dot: 'bg-amber-500', label: 'Degraded' },
  unreachable: { dot: 'bg-red-500', label: 'Backend unavailable' },
}

export function StatusIndicator() {
  const health = useHealth()
  const style = STYLES[health.status]
  const title =
    health.status === 'ok' || health.status === 'degraded'
      ? Object.entries(health.dependencies ?? {})
          .map(([dep, value]) => `${dep}: ${value}`)
          .join('\n')
      : undefined

  return (
    <div className="flex items-center gap-2 text-sm text-gray-600" title={title} data-testid="status-indicator">
      <span className={`h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
      <span>{style.label}</span>
    </div>
  )
}
