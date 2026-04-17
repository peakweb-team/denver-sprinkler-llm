# Future: Quantization-Aware Training (QAT) for BitNet Fine-Tuning

## Background

The current deployment uses Microsoft's official BitNet-b1.58-2B-4T model (unmodified) with a detailed system prompt for Denver Sprinkler domain adaptation. This works for a POC but the model has no domain-specific training data baked into its weights.

During the initial attempt to fine-tune BitNet, we discovered that the standard approach of LoRA fine-tuning on BF16 weights followed by post-training ternary quantization **destroys the fine-tuning signal**. Ternary rounding ({-1, 0, 1}) is too aggressive — it wipes out the subtle weight adjustments that LoRA produces.

## What Happened

1. Fine-tuned `microsoft/bitnet-b1.58-2B-4T-bf16` with LoRA (rank 16, all attention + MLP projections)
2. Training succeeded: loss converged from ~2.5 to ~1.4 over 3 epochs
3. Merged LoRA adapters into base BF16 weights
4. Applied BitNet's `preprocess-huggingface-bitnet.py` to quantize to ternary
5. Converted to GGUF i2_s format for bitnet.cpp
6. **Result: model output was garbage** ("GGGGG..." repeated tokens)

The ternary quantization rounds each weight to `{-1, 0, 1}` using `round(weight * scale).clamp(-1, 1)`. The LoRA fine-tuning adjustments are small relative to this rounding, so they're effectively erased.

## The Solution: Quantization-Aware Training (QAT)

QAT simulates the quantization during training, so the model learns weights that remain meaningful after ternary rounding. The approach:

1. **Forward pass**: Apply ternary quantization to weights (straight-through estimator)
2. **Backward pass**: Compute gradients on the full-precision weights (gradient flows through the quantization step)
3. **Update**: Adjust full-precision weights based on gradients
4. **Effect**: The model learns to "work around" the ternary constraint

### Implementation Sketch

```python
class StraightThroughTernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        scale = 1.0 / weight.abs().mean().clamp(min=1e-5)
        return (weight * scale).round().clamp(-1, 1) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-through estimator

# During training, replace linear layer forward:
def quantized_forward(self, x):
    quantized_weight = StraightThroughTernary.apply(self.weight)
    return F.linear(x, quantized_weight, self.bias)
```

### What This Would Require

- Custom training loop (can't use standard SFTTrainer without modification)
- Replace the forward pass of all linear layers with the quantized version
- May need longer training (5-10 epochs) since the optimization landscape is harder
- Full fine-tuning recommended over LoRA (since the quantization affects all weights)
- Estimated VRAM: ~12-16 GB on A10G (2.4B model in BF16 + gradients)
- Estimated training time: 30-60 minutes on g5.xlarge

### References

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) — original BitNet paper
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) — BitNet b1.58 paper
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285) — Microsoft's 2B training report
- Microsoft's BF16 training variant: `microsoft/bitnet-b1.58-2B-4T-bf16`

### Alternative: LoRA Adapter at Runtime

Another approach that may become viable as tooling matures:
- Keep the base BitNet model in ternary (i2_s GGUF)
- Load LoRA adapter at inference time via `--lora` flag
- Requires llama.cpp/bitnet.cpp to support LoRA on BitNet architecture
- As of April 2026, the `convert_lora_to_gguf.py` script does not support `BitNetForCausalLM`

## Current Approach (POC)

The POC uses the unmodified BitNet-b1.58-2B-4T model with a system prompt that includes:
- Company identity (Denver Sprinkler and Landscape)
- Business details (phone, email, address, hours)
- Service descriptions
- Behavioral instructions (helpful, grounded, conversational)

This relies on the model's instruction-following capability rather than domain-specific fine-tuning. For many POC use cases, this may be sufficient.

## When to Revisit QAT

Consider implementing QAT if:
- The system-prompt-only approach doesn't produce satisfactory responses
- The model frequently hallucinates services or details not in the prompt
- Response quality needs to match the level seen in the LoRA-fine-tuned (pre-quantization) BF16 model
- The project moves beyond POC to production
