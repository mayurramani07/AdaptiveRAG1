import { Settings } from 'lucide-react'
import { StatusIndicator } from './StatusIndicator'

interface HeaderProps {
  onOpenSettings: () => void
}

export function Header({ onOpenSettings }: HeaderProps) {
  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-5">
        <div>
          <h1 className="text-xl font-bold text-gray-900">AdaptiveRAG</h1>
          <p className="text-sm text-gray-500">Production RAG + CRAG Research Assistant</p>
        </div>
        <div className="flex items-center gap-4">
          <StatusIndicator />
          <button onClick={onOpenSettings} aria-label="Settings" className="rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600">
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </div>
    </header>
  )
}
