import { Sparkles } from 'lucide-react'

export function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center text-gray-400" data-testid="empty-state">
      <Sparkles className="h-8 w-8" />
      <p className="text-sm">Ask a question to search your knowledge base.</p>
    </div>
  )
}
