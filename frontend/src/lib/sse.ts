/**
 * Minimal Server-Sent Events parser over a fetch() Response body.
 *
 * Not using the browser's EventSource: it only supports GET with no custom
 * headers, and /v1/query needs POST + X-API-Key (see docs/01-prd.md 19.2).
 * Wire format is exactly generation.format_sse_event's output:
 * "event: <name>\ndata: <json>\n\n" - verified against src/adaptive_rag/generation.py.
 */

export interface ParsedSSEEvent {
  event: string
  data: unknown
}

export async function* parseSSEStream(response: Response): AsyncGenerator<ParsedSSEEvent> {
  const body = response.body
  if (!body) return

  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let separatorIndex: number
      while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
        const rawMessage = buffer.slice(0, separatorIndex)
        buffer = buffer.slice(separatorIndex + 2)
        const parsed = parseOneMessage(rawMessage)
        if (parsed) yield parsed
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function parseOneMessage(raw: string): ParsedSSEEvent | null {
  let event = 'message'
  let dataLine = ''
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim()
    else if (line.startsWith('data:')) dataLine += line.slice('data:'.length).trim()
  }
  if (!dataLine) return null
  try {
    return { event, data: JSON.parse(dataLine) }
  } catch {
    return null
  }
}
