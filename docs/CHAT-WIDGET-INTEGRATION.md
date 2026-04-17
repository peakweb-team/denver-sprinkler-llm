# Chat Widget Integration Guide

Instructions for connecting the Denver Sprinkler Next.js chat widget (`peakweb-team/denver-sprinkler`) to the BitNet inference API (`peakweb-team/denver-sprinkler-llm`).

**Related issues:**
- `peakweb-team/denver-sprinkler-llm#10` -- Integration issue
- `peakweb-team/denver-sprinkler#8` -- Chat widget issue

## API Reference

### Endpoints

**Health check:**
```
GET /health
Response: { "status": "ok", "model": "bitnet-b1.58-2B-4T", "version": "0.1.0" }
```

**Chat:**
```
POST /chat
Content-Type: application/json

Request:
{
  "messages": [
    { "role": "user", "content": "Do you offer sprinkler repair?" },
    { "role": "assistant", "content": "Yes we do! Call (303) 993-8717." },
    { "role": "user", "content": "What about in Aurora?" }
  ]
}

Response:
{ "response": "Absolutely, Aurora is part of our service area..." }
```

The `messages` array supports multi-turn conversation. Send the full conversation history with each request -- the API is stateless.

### Rate limiting
- 10 requests per minute per IP
- Returns `429 Too Many Requests` when exceeded

### Response time
- Expect 30-45 seconds per response (model loads from disk per request)
- The server returns `504` if inference exceeds 120 seconds

## Integration Steps

### 1. Add environment variable

In the `peakweb-team/denver-sprinkler` project:

**.env.local** (local development):
```
NEXT_PUBLIC_CHAT_API_URL=http://<elastic-ip>
```

**.env.example** (documentation):
```
# BitNet inference API URL (from denver-sprinkler-llm project)
NEXT_PUBLIC_CHAT_API_URL=
```

**Vercel project settings** (production):
- Go to Settings > Environment Variables
- Add `NEXT_PUBLIC_CHAT_API_URL` with the Elastic IP or domain
- Set for both Preview and Production environments

### 2. Update chat-widget.tsx

Replace the current mock `setTimeout` response with a real API call. Here is the complete updated component:

```tsx
'use client'

import { useState, useRef, useEffect } from 'react'
import { phone, phoneHref } from '@/lib/site-config'

interface Message {
  role: 'bot' | 'user'
  text: string
}

const WELCOME_MESSAGE =
  'Welcome to our site, if you need help simply reply to this message, we are online and ready to help.'

const FALLBACK_MESSAGE = `We'll get back to you soon. Call us at ${phone} for immediate help.`

const CHAT_API_URL = process.env.NEXT_PUBLIC_CHAT_API_URL || ''

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    { role: 'bot', text: WELCOME_MESSAGE },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus input when chat opens
  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && open) setOpen(false)
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [open])

  async function handleSend(e: React.FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || isLoading) return

    // Add user message
    const userMessage: Message = { role: 'user', text }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      // Build conversation history for the API
      // Map our 'bot'/'user' roles to the API's 'assistant'/'user' roles
      // Skip the welcome message (index 0)
      const history = [...messages.slice(1), userMessage].map((m) => ({
        role: m.role === 'bot' ? 'assistant' : 'user',
        content: m.text,
      }))

      const response = await fetchWithTimeout(
        `${CHAT_API_URL}/chat`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: history }),
        },
        60000 // 60 second timeout
      )

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`)
      }

      const data = await response.json()

      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: data.response },
      ])
    } catch (error) {
      console.error('Chat API error:', error)
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: FALLBACK_MESSAGE },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  // ... rest of the component (JSX) stays the same, with one addition:
  // Add a loading indicator after the last message when isLoading is true
```

### 3. Add the fetch timeout helper

Add this utility function above the component (or in a shared utils file):

```tsx
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } finally {
    clearTimeout(timeout)
  }
}
```

### 4. Add loading indicator to the chat UI

Inside the messages list JSX, add a typing indicator when the model is generating:

```tsx
{/* After the messages map, before messagesEndRef */}
{isLoading && (
  <div className="flex items-start gap-2">
    <div className="max-w-[85%] rounded-lg bg-gray-100 px-3 py-2 text-sm text-gray-900">
      <span className="inline-flex gap-1">
        <span className="animate-bounce" style={{ animationDelay: '0ms' }}>.</span>
        <span className="animate-bounce" style={{ animationDelay: '150ms' }}>.</span>
        <span className="animate-bounce" style={{ animationDelay: '300ms' }}>.</span>
      </span>
    </div>
  </div>
)}
```

### 5. Disable the send button while loading

Update the send button to show a disabled state:

```tsx
<button
  type="submit"
  disabled={isLoading}
  aria-label="Send message"
  className={`rounded-md p-2 text-white ${
    isLoading
      ? 'bg-gray-400 cursor-not-allowed'
      : 'bg-green-700 hover:bg-green-800'
  }`}
>
```

## Error Handling

The integration handles these failure modes:

| Scenario | Behavior |
|----------|----------|
| API unreachable | Shows fallback message with phone number |
| Response timeout (>60s) | Shows fallback message with phone number |
| 429 rate limited | Shows fallback message (next request works after 1 min) |
| 5xx server error | Shows fallback message with phone number |
| Network failure | Shows fallback message with phone number |

The fallback message is: *"We'll get back to you soon. Call us at (303) 993-8717 for immediate help."*

## CORS

The inference server allows requests from:
- `https://denversprinklerservices.com`
- `https://*.vercel.app`

If your Vercel deployment uses a different domain, update `CORS_ORIGINS` in `server/config.py`.

## Testing

### Local testing (mock mode)

Start the inference server in mock mode for frontend development without AWS:

```bash
# In the denver-sprinkler-llm repo
cd server
MOCK_MODE=true python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then in the denver-sprinkler repo:
```
NEXT_PUBLIC_CHAT_API_URL=http://localhost:8000
```

Mock mode returns instant canned responses -- useful for UI development.

### Testing against the live API

```bash
# Start the inference server on AWS
cd terraform && terraform apply -var="launch_inference=true"
# SSH in and start the server (see README for details)

# Set the env var to the Elastic IP
NEXT_PUBLIC_CHAT_API_URL=http://<elastic-ip>
```

### Verifying the integration

1. Open the chat widget on the site
2. Type "What services do you offer?" and send
3. You should see the loading indicator (bouncing dots) for 30-45 seconds
4. The response should mention Denver Sprinkler's services and phone number
5. Send a follow-up question to test multi-turn conversation
6. Verify the phone number in responses is always (303) 993-8717

## Architecture Notes

- The API is **stateless** -- conversation history must be sent with each request
- The `messages` array should include the full conversation (excluding the welcome message)
- Maximum 20 messages per request (enforced by the API)
- Maximum message length: 2,000 characters per message
- Response time is 30-45 seconds due to per-request model loading on t3.small
- The 60-second client timeout gives enough buffer over the typical response time

## Shutting Down the Server

When not actively testing, shut down the inference server to avoid charges:

```bash
cd terraform && terraform apply -var="launch_inference=false"
```

Cost while running: ~$0.02/hour (t3.small). Cost while stopped: $0.
