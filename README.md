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

### Site Corpus Extraction

The training corpus is built from the site's crawl data (`docs/crawl-raw.json` in `peakweb-team/denver-sprinkler`).

**Pipeline:**
1. `scripts/extract_corpus.py` fetches crawl JSON via GitHub API (or accepts a local file path)
2. Filters out PDFs (status 0) and deduplicates trailing-slash URL variants
3. Builds a frequency map of section texts; sections appearing on >50% of pages are classified as shared content (nav, header bar, footer, copyright) and removed
4. Extracts unique page content preserving heading structure (h1/h2/h3 + section text)
5. Categorizes pages by URL pattern: `service` (Denver-specific service pages), `city` (city landing pages for Littleton, Lakewood, etc.), `info` (about, contact, blog posts, testimonials)
6. Outputs JSONL to `data/site-corpus.jsonl`

**Validation:** `scripts/validate_corpus.py` checks page count, empty content, JSX/HTML artifacts, business detail accuracy, and content quality metrics.

**Schema:** Each line in `data/site-corpus.jsonl` is:
```json
{ "page": "/path/", "title": "Page Title", "content": "# Heading\n\nBody text...", "category": "service|city|info" }
```

### AWS Infrastructure (Terraform)

The `terraform/` directory contains POC-grade AWS infrastructure:

```bash
cd terraform/
terraform init
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your S3 bucket name and alert email
terraform plan
terraform apply
```

**AWS credentials** must be set before running Terraform:
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

Compute instances are **off by default** to avoid costs. Enable with:
```bash
terraform apply -var="launch_inference=true"   # Start inference server
terraform apply -var="launch_training=true"     # Start training instance
```

See [`terraform/README.md`](terraform/README.md) for full details and cost estimates.
### Project Milestones

See [GitHub Milestones](https://github.com/peakweb-team/denver-sprinkler-llm/milestones) for the full roadmap.

1. **M1: Foundation** — Dev container + Terraform infrastructure
2. **M2: Training Data** — Site extraction, RAG sources, synthetic Q&A
3. **M3: Model Training** — LoRA fine-tuning + BitNet quantization
4. **M4: Deployment** — Inference server on AWS
5. **M5: Integration** — Connect to Denver Sprinkler chat widget
