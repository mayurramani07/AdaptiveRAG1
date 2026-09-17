import { Check, Copy } from 'lucide-react'
import { useState } from 'react'
import type { AskState } from '../hooks/useAskQuery'
import { MetaBadges } from './MetaBadges'

interface AnswerPanelProps {
  answer: string
  status: AskState['status']
  meta: AskState['meta']
}

export function AnswerPanel({ answer, status, meta }: AnswerPanelProps) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard permission denied - not worth surfacing an error for
    }
  }

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm" aria-label="Answer">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">Answer</h2>
        {status === 'success' && answer && (
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-700"
            aria-label="Copy answer"
          >
            {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </div>

      <p className="whitespace-pre-wrap text-base leading-relaxed text-gray-900">
        {answer}
        {status === 'streaming' && <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-gray-400 align-middle" aria-hidden="true" />}
      </p>

      {status === 'success' && meta && (
        <div className="mt-4 border-t border-gray-100 pt-3">
          <MetaBadges meta={meta} />
        </div>
      )}
    </section>
  )
}
