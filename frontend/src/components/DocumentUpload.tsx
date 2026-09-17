import { CheckCircle2, FileUp, Loader2, XCircle } from 'lucide-react'
import { type ChangeEvent, useRef } from 'react'
import { useUploadDocument } from '../hooks/useUploadDocument'

/** Step 4/FR-FE10: lets the user add a document to the knowledge base via
 * the real POST /v1/documents endpoint - one file at a time, no progress
 * bar (the request is a single blocking call - see docs/01-prd.md 19.1). */
export function DocumentUpload() {
  const { state, upload, reset } = useUploadDocument()
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) upload(file)
    e.target.value = '' // allow re-selecting the same file later
  }

  const busy = state.status === 'uploading'

  return (
    <section className="rounded-xl border border-dashed border-gray-300 bg-white p-4" aria-label="Upload document">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <FileUp className="h-4 w-4 shrink-0 text-gray-400" />
          <span>Add a document (.pdf or .txt) to the knowledge base</span>
        </div>
        <button
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="flex shrink-0 items-center gap-1.5 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-400"
        >
          {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {busy ? 'Uploading…' : 'Choose file'}
        </button>
        <input ref={inputRef} type="file" accept=".pdf,.txt" onChange={handleFileChange} className="hidden" aria-label="Document file" />
      </div>

      {state.status === 'success' && state.result && (
        <p className="mt-3 flex items-center gap-1.5 text-sm text-emerald-700" data-testid="upload-success">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          Indexed "{state.result.filename}" as {state.result.chunks_indexed} chunk{state.result.chunks_indexed === 1 ? '' : 's'}. You can ask about it now.
          <button onClick={reset} className="ml-1 text-gray-400 underline hover:text-gray-600">
            dismiss
          </button>
        </p>
      )}

      {state.status === 'error' && state.error && (
        <p className="mt-3 flex items-center gap-1.5 text-sm text-red-700" data-testid="upload-error">
          <XCircle className="h-4 w-4 shrink-0" />
          {state.error}
          <button onClick={reset} className="ml-1 text-gray-400 underline hover:text-gray-600">
            dismiss
          </button>
        </p>
      )}
    </section>
  )
}
