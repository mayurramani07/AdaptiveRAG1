import { Loader2, Send } from 'lucide-react'
import { type FormEvent, useState } from 'react'

interface QueryFormProps {
  onSubmit: (query: string) => void
  busy: boolean
}

export function QueryForm({ onSubmit, busy }: QueryFormProps) {
  const [query, setQuery] = useState('')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmed = query.trim()
    if (!trimmed || busy) return
    onSubmit(trimmed)
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask anything about your knowledge base…"
        disabled={busy}
        aria-label="Query"
        className="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-3 text-base text-gray-900 placeholder:text-gray-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 disabled:bg-gray-50 disabled:text-gray-400"
      />
      <button
        type="submit"
        disabled={busy || !query.trim()}
        className="flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-6 py-3 font-medium text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300"
      >
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        {busy ? 'Asking…' : 'Ask'}
      </button>
    </form>
  )
}
