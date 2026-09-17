import { AlertTriangle, RotateCw } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onRetry: () => void
}

export function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4" role="alert" data-testid="error-banner">
      <AlertTriangle className="h-5 w-5 shrink-0 text-red-500" />
      <div className="flex-1">
        <p className="text-sm text-red-800">{message}</p>
      </div>
      <button
        onClick={onRetry}
        className="flex shrink-0 items-center gap-1.5 rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100"
      >
        <RotateCw className="h-3.5 w-3.5" />
        Retry
      </button>
    </div>
  )
}
