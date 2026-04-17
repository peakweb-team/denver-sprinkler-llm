# Denver Sprinkler LLM

A BitNet 1.58-bit LLM powering the customer chat widget for [Denver Sprinkler & Landscape](https://denversprinklerservices.com). Uses Microsoft's BitNet-b1.58-2B-4T model with a domain-specific system prompt, served via bitnet.cpp on a small AWS instance.

## Architecture

```
┌─────────────────────┐     HTTPS      ┌───────────────────────────┐
│  Next.js Chat Widget │ ──────────────▶│  FastAPI + bitnet.cpp      │
│  (Vercel)            │                │  (AWS t3.small, on-demand) │
└─────────────────────┘                └───────────────────────────┘
                                                │
                                        ┌───────┴────────────┐
                                        │ BitNet-b1.58-2B-4T  │
                                        │ ggml-model-i2_s.gguf│
                                        │ (1.2 GB, 1.58-bit)  │
                                        └────────────────────┘
```

## Model

| Property | Value |
|----------|-------|
| Base model | Microsoft BitNet-b1.58-2B-4T |
| Parameters | 2.4B |
| Quantization | 1.58-bit ternary ({-1, 0, 1}) -- native, not post-training |
| Model size | 1.2 GB (GGUF i2_s format) |
| Domain adaptation | System prompt with company details, services, and rules |
| Inference | bitnet.cpp with optimized ternary kernels (CPU-only) |
| Response time | ~40-45s on t3.small (includes model loading per request) |

## Cost Analysis

### Self-hosted BitNet (current approach)

Running on t3.small during business hours only (Mon-Fri 7am-5pm, Sat 8am-2pm):

| Item | Cost |
|------|------|
| t3.small x 56 hrs/week x 4.3 weeks | $4.82/month |
| S3 model storage (1.2 GB) | $0.03/month |
| **Total** | **~$5/month** |

Unlimited queries while running. Zero per-query cost.

### vs. Cloud LLM API (e.g., Claude Haiku via Vercel AI SDK)

| Monthly queries | API cost |
|----------------|----------|
| 500 | $0.15 |
| 2,000 | $0.60 |
| 10,000 | $3.00 |
| 50,000 | $15.00 |

### Comparison

| Factor | BitNet self-hosted | Cloud API |
|--------|-------------------|-----------|
| Monthly cost | ~$5 fixed | $0.15-$15 usage-based |
| Response time | 40-45 seconds | <1 second |
| Concurrency | 1 request at a time | Unlimited |
| Infrastructure | You manage AWS | Zero |
| Privacy | Data stays on your AWS | Data goes to API provider |
| Quality | Good (system prompt) | Excellent |
| Vendor dependency | None | API provider |

**Bottom line:** BitNet is cost-competitive at ~$5/month for a low-traffic chat widget where users already expect a wait (current widget connects to a human agent with multi-minute response times). The 40-45s model response is a significant improvement over the status quo.

### Future: Fine-Tuned BitNet via QAT

The current deployment uses the base (unmodified) BitNet model with a comprehensive system prompt. Standard LoRA fine-tuning followed by ternary re-quantization destroys the fine-tuning signal. True domain-specific fine-tuning requires Quantization-Aware Training (QAT). See [`docs/FUTURE-QAT-FINE-TUNING.md`](docs/FUTURE-QAT-FINE-TUNING.md) for details.

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

# Inside the container -- authenticate Claude Code
claude --dangerously-skip-permissions
# Follow the OAuth flow to sign in
```

### Testing the Chat CLI

```bash
# Start the inference server (on-demand)
cd terraform && terraform apply -var="launch_inference=true"
# SSH into the instance and run: bash /tmp/start-server.sh

# Interactive chat
python3 scripts/chat-cli.py --url http://<elastic-ip> --timeout 120

# Batch test with sample prompts
python3 scripts/chat-cli.py --url http://<elastic-ip> --timeout 120 --batch data/test-prompts.txt

# Shut down when done
cd terraform && terraform apply -var="launch_inference=false"
```

### Training Data Pipeline

The training data pipeline extracts content from the Denver Sprinkler website and authoritative landscaping sources:

1. **Site corpus** (`scripts/extract_corpus.py`) -- 58 pages from denversprinklerservices.com
2. **RAG sources** (`scripts/crawl_rag_sources.py`) -- 906 chunks from CSU Extension, Rain Bird, EPA WaterSense, Denver Water
3. **Q&A pairs** (`data/training-pairs.jsonl`) -- 1,049 synthetic instruction/response pairs

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

Compute instances are **off by default** to avoid costs:
```bash
terraform apply -var="launch_inference=true"   # Start inference server (~$0.02/hr)
terraform apply -var="launch_training=true"    # Start GPU training instance (~$1.00/hr)
terraform apply -var="launch_inference=false"  # Stop inference server
terraform apply -var="launch_training=false"   # Stop training instance
```

### Project Structure

```
configs/              Training and quantization configuration
data/                 Training data (corpus, RAG sources, Q&A pairs)
docs/                 Technical documentation and future plans
models/               Model card (artifacts in S3, not git)
scripts/              Data extraction, training, quantization, CLI tools
server/               FastAPI inference server + Dockerfile
terraform/            AWS infrastructure (VPC, S3, EC2, IAM, budgets)
```
