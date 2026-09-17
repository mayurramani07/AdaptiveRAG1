import { GitBranch, RefreshCcw, ShieldCheck, ShieldAlert, Zap } from 'lucide-react'
import type { AskState } from '../hooks/useAskQuery'

const ROUTE_LABELS: Record<string, string> = {
  'hybrid-rag': 'Hybrid RAG (dense + BM25)',
  'graph-rag': 'Graph RAG',
  'bm25-rag': 'BM25 only',
  'simple-rag': 'Dense only',
  chitchat: 'No retrieval (chitchat)',
}

/** Step 4: surfaces real CRAG/retrieval metadata the API actually returns -
 * never fabricated. Nothing here is invented beyond what `meta` carries. */
export function MetaBadges({ meta }: { meta: NonNullable<AskState['meta']> }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
      <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-1">
        <GitBranch className="h-3 w-3" />
        {ROUTE_LABELS[meta.route] ?? meta.route}
      </span>
      {meta.recovery_used && (
        <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-1 text-amber-700">
          <RefreshCcw className="h-3 w-3" />
          Recovery triggered
        </span>
      )}
      {meta.cache_hit && (
        <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2.5 py-1 text-blue-700">
          <Zap className="h-3 w-3" />
          Cached
        </span>
      )}
      {meta.grounding &&
        (meta.grounding.status === 'passed' ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-700">
            <ShieldCheck className="h-3 w-3" />
            Grounded
          </span>
        ) : meta.grounding.status === 'failed' ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2.5 py-1 text-red-700">
            <ShieldAlert className="h-3 w-3" />
            Grounding failed
          </span>
        ) : null)}
      <span className="ml-auto text-gray-400">{meta.latency_ms.toFixed(0)}ms</span>
    </div>
  )
}
