import { useCallback, useState } from 'react'
import { getStoredApiKey, setStoredApiKey } from '../lib/storage'

export function useApiKey() {
  const [apiKey, setApiKeyState] = useState(getStoredApiKey)

  const setApiKey = useCallback((key: string) => {
    setStoredApiKey(key)
    setApiKeyState(key)
  }, [])

  return { apiKey, setApiKey }
}
