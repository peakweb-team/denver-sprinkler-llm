# Model Card: Denver Sprinkler & Landscape Chat Assistant

## Model Details

- **Model name:** Denver Sprinkler & Landscape Chat Assistant
- **Base model:** [Llama 3.2 3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (Meta)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj
- **Training framework:** Hugging Face Transformers + TRL (SFTTrainer)
- **License:** [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)

## Intended Use

This model powers the customer chat widget on the Denver Sprinkler and Landscape website. It answers questions about:

- Sprinkler system installation, repair, and maintenance
- Landscape design and hardscaping services
- Seasonal lawn care and irrigation scheduling
- Service areas, pricing, and scheduling
- Company contact information and hours of operation

**Not intended for:** General-purpose chat, medical/legal advice, or any use outside the Denver Sprinkler and Landscape business context.

## Training Data

### Site Corpus
- **Source:** 58 pages crawled from denversprinklerservices.com
- **Content:** Service descriptions, FAQs, blog posts, location and contact details

### RAG-Sourced Knowledge
- **Sources:** 906 chunks from authoritative landscaping and irrigation references:
  - Colorado State University Extension guides
  - Rain Bird technical documentation
  - EPA WaterSense program materials
  - Denver Water conservation guidelines
- **Purpose:** Supplement site-specific content with domain expertise for Colorado Front Range landscaping

### Training Pairs
- **Total:** 1,049 Q&A pairs in instruction/response format
- **Split:** 90% train / 10% validation (20 held out for post-training evaluation)
- **Format:** JSONL with `instruction` and `response` fields, formatted using Llama 3.2 chat template

## Quantization

This model uses a dual-path quantization strategy:

### Path A: BitNet Ternary (Experimental)

Post-training quantization to ternary weights {-1, 0, +1} using the [microsoft/BitNet](https://github.com/microsoft/BitNet) tooling:

1. Weight preprocessing via `preprocess-huggingface-bitnet.py` (absmean quantization)
2. Conversion to GGUF via `convert-hf-to-gguf-bitnet.py`
3. Final quantization with `llama-quantize` using `i2_s` type

**Output:** `ggml-model-i2_s.gguf` (compatible with bitnet.cpp inference)

### Path B: GGUF Standard (Production Recommended)

Standard quantization via [llama.cpp](https://github.com/ggerganov/llama.cpp):

1. HF-to-GGUF conversion via `convert_hf_to_gguf.py`
2. Quantization with `llama-quantize` at multiple levels

**Variants produced:**

| Variant | Bits/Weight | Estimated Size | Use Case |
|---------|-------------|----------------|----------|
| Q2_K    | ~2.5        | ~1.0 GB        | Smallest, most aggressive compression |
| Q3_K_S  | ~3.0        | ~1.3 GB        | Small with better quality than Q2 |
| Q4_K_M  | ~4.5        | ~1.8 GB        | **Recommended** -- best quality/size tradeoff |

*Note: Size values are estimates for a 3B parameter model and will be updated after quantization runs.*

### Size Comparison

| Format          | Estimated Size | Compression |
|-----------------|----------------|-------------|
| FP16 (original) | ~6.4 GB        | 1.0x        |
| BitNet i2_s     | ~0.6 GB        | ~10x        |
| GGUF Q2_K       | ~1.0 GB        | ~6x         |
| GGUF Q3_K_S     | ~1.3 GB        | ~5x         |
| GGUF Q4_K_M     | ~1.8 GB        | ~3.5x       |

*These are placeholder estimates. Actual sizes will be filled after quantization completes.*

## Known Limitations and Caveats

### Post-Training BitNet Quantization

The BitNet ternary quantization path applies post-training quantization to a model that was trained with FP16 weights. This is fundamentally different from true BitNet models (such as `microsoft/BitNet-b1.58-2B-4T`), which are trained from scratch with quantization-aware training (QAT). Post-training quantization to ternary weights will cause significant quality degradation because the model never learned to operate with such extreme weight constraints during training.

**For true BitNet 1-bit inference performance**, the correct approach would be to either:
- Fine-tune a model that was already trained with BitNet architecture (e.g., start from a BitNet-native base model)
- Perform quantization-aware training (QAT) during fine-tuning to gradually adapt weights to ternary values

### Domain Specificity
- This model is trained exclusively on Denver Sprinkler and Landscape content and Colorado Front Range landscaping knowledge
- It may produce incorrect or fabricated answers for topics outside its training domain
- It should not be used as a general-purpose assistant

### Business Details Accuracy
The model is trained to provide accurate business contact information:
- **Phone:** (303) 993-8717
- **Email:** info@denversprinklerservices.com
- **Address:** 3971 S Decatur St Unit A, Englewood, CO 80110
- **Hours:** Mon-Fri 7am-5pm, Sat 8am-2pm, Sun Closed
- **Emergency:** Available 24/7

If any of these details change, the model must be retrained or the serving layer must intercept and correct them.

### Small Model Limitations
- Llama 3.2 3B is a relatively small language model
- It may struggle with complex multi-turn conversations
- Response quality will vary; a production deployment should include guardrails and fallback mechanisms

## Evaluation

Quality is evaluated on 20 held-out Q&A pairs not seen during training. Evaluation compares:
- Base model (Llama 3.2 3B-Instruct, no fine-tuning) responses
- Fine-tuned model (FP16) responses
- Each quantized variant's responses

Quality checks include:
- **Response coherence:** Minimum length, repetition detection
- **Factual accuracy:** Business details verified when referenced
- **Regression testing:** Comparison against pre-quantization fine-tuned outputs

Results are stored in `models/quantized_eval_results.json`.

## Infrastructure

- **Training hardware:** AWS g5.xlarge (NVIDIA A10G, 24 GB VRAM)
- **Training time:** Estimated 30-45 minutes
- **Training cost:** Estimated $0.25-0.40 (spot pricing)
- **Inference:** bitnet.cpp or llama.cpp server behind FastAPI on AWS
- **Model storage:** S3 bucket

## Ethical Considerations

- This model is designed for a narrow commercial use case (customer support chat)
- It does not generate content about sensitive topics
- Business information accuracy is verified during evaluation
- The model should always offer to connect customers with a human representative for complex issues

## Citation

```text
Base model: Meta Llama 3.2 3B-Instruct
BitNet tooling: microsoft/BitNet (https://github.com/microsoft/BitNet)
GGUF tooling: ggerganov/llama.cpp (https://github.com/ggerganov/llama.cpp)
```
