# Denver Sprinkler LLM

Fine-tuned BitNet 1-bit LLM for the Denver Sprinkler & Landscape customer chat widget.

## Project Overview
- **Base model:** Llama 3.2 3B, fine-tuned with LoRA, quantized to 1-bit via BitNet
- **Training data:** Site content from peakweb-team/denver-sprinkler + RAG-sourced landscaping knowledge
- **Inference:** bitnet.cpp behind a FastAPI REST API on AWS
- **Consumer:** Chat widget in the Denver Sprinkler Next.js site

## Directory Structure
- `.devcontainer/` — Anthropic Claude Code container + ML/infra tooling
- `data/` — Training data (site corpus, RAG sources, Q&A pairs)
- `models/` — Model artifacts (large files go to S3, not git)
- `scripts/` — Data extraction, training, quantization scripts
- `server/` — FastAPI inference server wrapping bitnet.cpp
- `terraform/` — AWS infrastructure (training instances, inference server, S3, budgets)

## Autonomous Workflow Rules
When running with `--dangerously-skip-permissions`:
1. **Terraform apply is allowed** if `terraform plan` shows estimated cost under $50/month
2. **If cost exceeds $50/month**, stop and ask the user
3. **Always tear down** training instances (g5) after training completes and artifacts are in S3
4. **Never commit** `.env`, AWS credentials, or model binary files to git
5. **Log all AWS resource provisioning** to a run journal in the issue comments

## Key References
- BitNet: https://github.com/microsoft/BitNet
- Site repo: https://github.com/peakweb-team/denver-sprinkler
- Site crawl data: `peakweb-team/denver-sprinkler/docs/crawl-raw.json`
- Site config (phone, address, hours): `peakweb-team/denver-sprinkler/lib/site-config.ts`

## Business Details (must be accurate in all training data and model outputs)
- **Name:** Denver Sprinkler and Landscape
- **Phone:** (303) 993-8717
- **Email:** info@denversprinklerservices.com
- **Address:** 3971 S Decatur St Unit A, Englewood, CO 80110
- **Hours:** Mon-Fri 7am-5pm, Sat 8am-2pm, Sun Closed
- **Emergency:** Available 24/7
