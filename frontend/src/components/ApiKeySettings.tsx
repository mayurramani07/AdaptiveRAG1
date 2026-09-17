import { KeyRound, X } from 'lucide-react'
import { useState } from 'react'

interface ApiKeySettingsProps {
  apiKey: string
  onSave: (key: string) => void
  onClose: () => void
}

/** Step 6: the user supplies their own API key client-side - never
 * hardcoded into source, persisted to localStorage only. */
export function ApiKeySettings({ apiKey, onSave, onClose }: ApiKeySettingsProps) {
  const [value, setValue] = useState(apiKey)

  function handleSave() {
    onSave(value.trim())
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" role="dialog" aria-modal="true" aria-label="API key settings">
      <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-base font-semibold text-gray-900">
            <KeyRound className="h-4 w-4" />
            API Key
          </h2>
          <button onClick={onClose} aria-label="Close" className="rounded-md p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mb-3 text-sm text-gray-500">
          Stored only in this browser (<code>localStorage</code>) and sent as the <code>X-API-Key</code> header. Never bundled into application code.
        </p>
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Your backend API key"
          className="mb-4 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
        />
        <button onClick={handleSave} className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700">
          Save
        </button>
      </div>
    </div>
  )
}
