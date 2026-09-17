const API_KEY_STORAGE_KEY = 'adaptive-rag:api-key'

/** localStorage can throw (private browsing, blocked site data) - never let
 * a storage failure break the app; the caller just gets an empty key back. */
export function getStoredApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setStoredApiKey(key: string): void {
  try {
    localStorage.setItem(API_KEY_STORAGE_KEY, key)
  } catch {
    // ponytail: silent - a settings panel toast for this edge case isn't worth it
  }
}
