import { useCallback, useState } from 'react'
import { ApiError, uploadDocument } from '../services/api'
import type { UploadResult } from '../types/api'

export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

export interface UploadState {
  status: UploadStatus
  result: UploadResult | null
  error: string | null
}

const initialState: UploadState = { status: 'idle', result: null, error: null }

export function useUploadDocument() {
  const [state, setState] = useState<UploadState>(initialState)

  const upload = useCallback(async (file: File) => {
    setState({ status: 'uploading', result: null, error: null })
    try {
      const result = await uploadDocument(file)
      setState({ status: 'success', result, error: null })
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Upload failed. Please try again.'
      setState({ status: 'error', result: null, error: message })
    }
  }, [])

  const reset = useCallback(() => setState(initialState), [])

  return { state, upload, reset }
}
