# Denver Sprinkler LLM

A BitNet 1-bit LLM fine-tuned on Denver Sprinkler & Landscape site content and authoritative landscaping sources. Powers the customer chat widget on [denversprinklerservices.com](https://denversprinklerservices.com).

## Architecture

```
┌─────────────────────┐     HTTPS      ┌──────────────────────┐
│  Next.js Chat Widget │ ──────────────▶│  FastAPI + bitnet.cpp │
│  (Vercel)            │                │  (AWS t3.medium)      │
└─────────────────────┘                └──────────────────────┘
                                                │
                                        ┌───────┴───────┐
                                        │ denver-sprinkler │
                                        │ -3b-1bit.gguf    │
                                        └─────────────────┘
```

## Model

| Property | Value |
|----------|-------|
| Base model | Llama 3.2 3B |
| Fine-tuning | LoRA (rank 16-32) |
| Quantization | BitNet 1-bit |
| Training data | Site content + RAG-sourced landscaping knowledge |
| Inference | bitnet.cpp (CPU-only, no GPU needed) |

## Development

### Prerequisites
- Docker
- AWS credentials (access key + secret key)
- Claude Code subscription (Max/Pro) for OAuth login

### Quick Start

```bash
# Clone
git clone https://github.com/peakweb-team/denver-sprinkler-llm.git
cd denver-sprinkler-llm

# Copy env vars
cp .env.example .env
# Edit .env with your AWS credentials

# Build and run dev container
docker build -t denver-sprinkler-llm .devcontainer/
docker run -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --env-file .env \
  denver-sprinkler-llm

# Inside the container — authenticate Claude Code
claude --dangerously-skip-permissions
# Follow the OAuth flow to sign in
```

### Project Milestones

See [GitHub Milestones](https://github.com/peakweb-team/denver-sprinkler-llm/milestones) for the full roadmap.

1. **M1: Foundation** — Dev container + Terraform infrastructure
2. **M2: Training Data** — Site extraction, RAG sources, synthetic Q&A
3. **M3: Model Training** — LoRA fine-tuning + BitNet quantization
4. **M4: Deployment** — Inference server on AWS
5. **M5: Integration** — Connect to Denver Sprinkler chat widget
