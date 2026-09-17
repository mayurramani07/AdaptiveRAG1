import { useState } from 'react'
import { AnswerPanel } from './components/AnswerPanel'
import { ApiKeySettings } from './components/ApiKeySettings'
import { DocumentUpload } from './components/DocumentUpload'
import { EmptyState } from './components/EmptyState'
import { ErrorBanner } from './components/ErrorBanner'
import { Header } from './components/Header'
import { QueryForm } from './components/QueryForm'
import { SourcesList } from './components/SourcesList'
import { useApiKey } from './hooks/useApiKey'
import { useAskQuery } from './hooks/useAskQuery'

function App() {
  const { apiKey, setApiKey } = useApiKey()
  const { state, ask, retry } = useAskQuery()
  const [settingsOpen, setSettingsOpen] = useState(false)

  const busy = state.status === 'loading' || state.status === 'streaming'
  const hasResult = state.status === 'success' || state.status === 'streaming'

  return (
    <div className="min-h-screen bg-gray-50">
      <Header onOpenSettings={() => setSettingsOpen(true)} />

      <main className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <DocumentUpload />

        <QueryForm onSubmit={ask} busy={busy} />

        {state.status === 'error' && state.error && <ErrorBanner message={state.error} onRetry={retry} />}

        {hasResult && <AnswerPanel answer={state.answer} status={state.status} meta={state.meta} />}

        {hasResult && state.sources.length > 0 && <SourcesList sources={state.sources} />}

        {state.status === 'idle' && <EmptyState />}
      </main>

      {settingsOpen && <ApiKeySettings apiKey={apiKey} onSave={setApiKey} onClose={() => setSettingsOpen(false)} />}
    </div>
  )
}

export default App
