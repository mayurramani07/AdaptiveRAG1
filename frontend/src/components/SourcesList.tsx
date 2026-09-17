import { FileText, ShieldCheck } from 'lucide-react'
import type { Source } from '../types/api'

interface SourcesListProps {
  sources: Source[]
}

/** Step 4: only fields the real API returns (id, doc_id, trust_tier, score,
 * text) - no fabricated title/date/author. */
export function SourcesList({ sources }: SourcesListProps) {
  if (sources.length === 0) return null

  return (
    <section aria-label="Sources">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">Sources / Evidence</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        {sources.map((source, i) => (
          <article key={source.id ?? i} className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm" data-testid="source-card">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 truncate text-sm font-medium text-gray-800" title={source.doc_id ?? undefined}>
                <FileText className="h-3.5 w-3.5 shrink-0 text-gray-400" />
                {source.doc_id ?? 'unknown document'}
              </span>
              {source.trust_tier === 'authoritative' && (
                <span className="flex items-center gap-1 shrink-0 rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">
                  <ShieldCheck className="h-3 w-3" />
                  authoritative
                </span>
              )}
              {source.trust_tier && source.trust_tier !== 'authoritative' && (
                <span className="shrink-0 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">{source.trust_tier}</span>
              )}
            </div>
            {source.text && <p className="line-clamp-3 text-sm text-gray-600">{source.text}</p>}
            {source.score !== null && source.score !== undefined && <p className="mt-2 text-xs text-gray-400">relevance score: {source.score.toFixed(2)}</p>}
          </article>
        ))}
      </div>
    </section>
  )
}
