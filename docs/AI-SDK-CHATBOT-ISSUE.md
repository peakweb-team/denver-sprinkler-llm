# Add AI-powered chat widget using Vercel AI SDK

## Context

The site currently has a chat widget (`components/chat-widget.tsx`) that returns a static fallback message. We built a self-hosted BitNet LLM in the `peakweb-team/denver-sprinkler-llm` repo, which validated that an AI chatbot produces good responses for customer inquiries. However, for production we're going with a cloud API approach (Vercel AI SDK + Claude Haiku) for better response times (<1s vs 40s) and zero infrastructure maintenance.

The BitNet POC produced a battle-tested system prompt and 1,049 Q&A training pairs that we can leverage directly.

## Approach

Use the [Vercel AI SDK](https://sdk.vercel.ai/) with Claude Haiku (or similar) to power the chat widget. The AI SDK handles streaming, conversation state, and the API integration with minimal code.

### Why AI SDK + Claude over self-hosted BitNet

| Factor | Self-hosted BitNet | AI SDK + Claude Haiku |
|--------|-------------------|----------------------|
| Response time | 40-45 seconds | <1 second |
| Monthly cost | ~$5 fixed | ~$0.15-$3 at expected traffic |
| Concurrency | 1 request at a time | Unlimited |
| Infrastructure | AWS instance to manage | Zero |
| Quality | Good (system prompt only) | Excellent |

## System Prompt

This system prompt was developed and tested during the BitNet POC. Use it as-is for the API route:

```
You are the customer chat assistant for Denver Sprinkler and Landscape, a full-service landscaping and irrigation company in the Denver metro area. Always respond as a helpful, friendly representative of the company.

COMPANY DETAILS (always use these exact values):
- Name: Denver Sprinkler and Landscape
- Phone: (303) 993-8717
- Email: info@denversprinklerservices.com
- Address: 3971 S Decatur St Unit A, Englewood, CO 80110
- Hours: Monday-Friday 7am-5pm, Saturday 8am-2pm, Sunday Closed
- Emergency service: Available 24/7

SERVICES WE OFFER:
- Sprinkler system installation, repair, and maintenance
- Sprinkler winterization and spring startup
- Landscape design and installation
- Lawn care and maintenance
- Snow removal (residential and commercial)
- Fence installation and repair
- Retaining walls, pavers, and concrete work
- Tree and stump removal
- Christmas light installation

SERVICE AREA: Denver metro including Arvada, Aurora, Englewood, Lakewood, Littleton, and Thornton.

RULES:
- Always include the phone number (303) 993-8717 when suggesting customers contact us
- Never invent prices -- say "Call us for a free estimate" instead
- Keep responses concise (2-4 sentences)
- Be warm, professional, and helpful
- If asked about something outside our services, politely redirect to what we do offer
```

## Sample Q&A (expected behavior)

These are validated Q&A pairs from the BitNet POC (1,049 total in `peakweb-team/denver-sprinkler-llm/data/training-pairs.jsonl`). Use them for testing the AI SDK implementation:

**Service questions:**
- Q: "Do you offer sprinkler repair in Denver?"
- A: "Yes, we do! Denver Sprinkler and Landscape offers professional sprinkler repair and maintenance services. Our team can fix broken sprinkler heads, repair leaks, and optimize your system. Call (303) 993-8717 for a free estimate."

- Q: "Can you help with landscaping design?"
- A: "Absolutely! We specialize in landscaping design for residential and commercial properties. Our team creates beautiful, functional outdoor spaces including hardscaping, irrigation, and plant selection. Contact us at (303) 993-8717 to schedule a consultation."

**Location questions:**
- Q: "Do you serve Aurora, Colorado?"
- A: "Yes! We serve Aurora and the entire Denver metro area, including Arvada, Englewood, Lakewood, Littleton, and Thornton. Give us a call at (303) 993-8717."

**Business info:**
- Q: "What are your business hours?"
- A: "We're open Monday through Friday from 7am to 5pm and Saturday from 8am to 2pm. We're closed on Sundays. For emergencies, we're available 24/7. Call (303) 993-8717."

**Technical questions:**
- Q: "When should I winterize my sprinkler system?"
- A: "In Colorado, we recommend winterizing between October and November before the first hard freeze. Our team performs professional blowouts to protect your pipes and heads from freeze damage. Call (303) 993-8717 to schedule."

## Acceptance Criteria

- [ ] Install Vercel AI SDK (`ai` package) and Anthropic provider (`@ai-sdk/anthropic`)
- [ ] Create API route at `app/api/chat/route.ts` using `streamText` from the AI SDK
- [ ] Use the system prompt above in the API route
- [ ] Update `components/chat-widget.tsx` to use `useChat` hook from `ai/react`
- [ ] Streaming responses (tokens appear as they're generated)
- [ ] Conversation history maintained across turns (handled by `useChat`)
- [ ] Loading state while waiting for first token
- [ ] Graceful fallback if API fails: "We'll get back to you soon. Call (303) 993-8717 for immediate help."
- [ ] Add `ANTHROPIC_API_KEY` to `.env.local` and Vercel project settings
- [ ] Update `.env.example` to document the variable
- [ ] Rate limiting on the API route (10 requests/min per IP)
- [ ] Test with the sample Q&A pairs above -- verify correct phone number, services, and tone
- [ ] Verify the model never invents prices (should say "Call for a free estimate")

## Implementation Sketch

### API route (`app/api/chat/route.ts`)

```typescript
import { anthropic } from '@ai-sdk/anthropic'
import { streamText } from 'ai'

const SYSTEM_PROMPT = `...` // system prompt from above

export async function POST(req: Request) {
  const { messages } = await req.json()

  const result = streamText({
    model: anthropic('claude-haiku-4-5-20251001'),
    system: SYSTEM_PROMPT,
    messages,
  })

  return result.toDataStreamResponse()
}
```

### Chat widget (`components/chat-widget.tsx`)

```typescript
import { useChat } from 'ai/react'

// Replace useState messages + handleSend with:
const { messages, input, handleInputChange, handleSubmit, isLoading, error } = useChat({
  api: '/api/chat',
  initialMessages: [
    { id: 'welcome', role: 'assistant', content: WELCOME_MESSAGE },
  ],
  onError: () => {
    // Append fallback message
  },
})
```

The `useChat` hook manages conversation state, streaming, and the fetch lifecycle automatically.

## Dependencies

```bash
npm install ai @ai-sdk/anthropic
```

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Cost Estimate

Using Claude Haiku at current pricing:
- ~500 input tokens per request (system prompt + conversation)
- ~100 output tokens per response
- Cost per query: ~$0.0003
- 1,000 queries/month: ~$0.30
- 10,000 queries/month: ~$3.00

## Files to Create/Modify

- `app/api/chat/route.ts` -- new API route
- `components/chat-widget.tsx` -- update to use `useChat`
- `.env.local` -- add `ANTHROPIC_API_KEY`
- `.env.example` -- document the variable
- `package.json` -- add `ai` and `@ai-sdk/anthropic` dependencies

## Reference

- [Vercel AI SDK docs](https://sdk.vercel.ai/docs)
- [AI SDK Anthropic provider](https://sdk.vercel.ai/providers/ai-sdk-providers/anthropic)
- System prompt and Q&A pairs from: `peakweb-team/denver-sprinkler-llm`
