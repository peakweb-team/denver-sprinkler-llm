#!/usr/bin/env python3
"""
Denver Sprinkler LLM — Standalone Evaluation Script

Compares base model vs fine-tuned model responses on held-out questions.
Can be run independently after training completes.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --finetuned_model models/merged --held_out models/held_out_questions.json
    python scripts/evaluate.py --config configs/training.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "configs/training.yaml"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for Denver Sprinkler and Landscape, "
    "a landscaping and sprinkler company in Englewood, Colorado. "
    "Phone: (303) 993-8717. Email: info@denversprinklerservices.com. "
    "Address: 3971 S Decatur St Unit A, Englewood, CO 80110. "
    "Hours: Mon-Fri 7am-5pm, Sat 8am-2pm, Sun Closed. Emergency service available 24/7."
)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_model(model_path: str, hf_token: str | None = None):
    """Load a model and tokenizer from a local path or HuggingFace hub."""
    logger.info("Loading model from: %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    temperature: float = 0.7,
    max_new_tokens: int = 300,
) -> str:
    """Generate a response from a model given chat messages."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    base_model_name: str,
    finetuned_model_dir: str,
    held_out: list[dict],
    system_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 300,
    output_path: str = "models/eval_results.json",
) -> list[dict]:
    """Run side-by-side evaluation of base vs fine-tuned model."""
    hf_token = os.environ.get("HF_TOKEN")
    start_time = time.time()

    # Load both models
    logger.info("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_model(finetuned_model_dir)

    logger.info("Loading base model...")
    base_model, base_tokenizer = load_model(base_model_name, hf_token)

    results = []

    for i, pair in enumerate(held_out):
        logger.info(
            "Evaluating question %d/%d: %s",
            i + 1,
            len(held_out),
            pair["instruction"][:80],
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["instruction"]},
        ]

        # Generate responses
        ft_answer = generate_response(
            ft_model, ft_tokenizer, messages, temperature, max_new_tokens
        )
        base_answer = generate_response(
            base_model, base_tokenizer, messages, temperature, max_new_tokens
        )

        result = {
            "question_index": i,
            "question": pair["instruction"],
            "reference_answer": pair["response"],
            "base_model_answer": base_answer,
            "finetuned_model_answer": ft_answer,
            "source": pair.get("source", ""),
            "source_ref": pair.get("source_ref", ""),
        }
        results.append(result)

        # Print side-by-side comparison
        logger.info("-" * 60)
        logger.info("Q: %s", pair["instruction"])
        logger.info("Reference: %s", pair["response"][:200])
        logger.info("Base model: %s", base_answer[:200])
        logger.info("Fine-tuned: %s", ft_answer[:200])
        logger.info("-" * 60)

    # Clean up
    del base_model
    del ft_model
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info(
        "Evaluation complete: %d questions in %.1f seconds", len(results), elapsed
    )

    # Save results
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "base_model": base_model_name,
                    "finetuned_model": finetuned_model_dir,
                    "num_questions": len(results),
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "elapsed_seconds": round(elapsed, 1),
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Results saved to %s", output_path_obj)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model vs base model on held-out questions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (overrides individual args)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="models/merged",
        help="Path to fine-tuned (merged) model directory",
    )
    parser.add_argument(
        "--held_out",
        type=str,
        default="models/held_out_questions.json",
        help="Path to held-out questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt (default: from config or built-in)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum new tokens to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Load config if provided
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = str(project_root / config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        args.base_model = config.get("base_model", args.base_model)
        args.finetuned_model = config.get("merged_output_dir", args.finetuned_model)
        args.temperature = config.get("eval_temperature", args.temperature)
        args.max_new_tokens = config.get("eval_max_new_tokens", args.max_new_tokens)
        system_prompt = config.get("system_prompt", system_prompt)

    if args.system_prompt:
        system_prompt = args.system_prompt

    # Resolve paths relative to project root
    finetuned_path = args.finetuned_model
    if not os.path.isabs(finetuned_path):
        finetuned_path = str(project_root / finetuned_path)

    held_out_path = args.held_out
    if not os.path.isabs(held_out_path):
        held_out_path = str(project_root / held_out_path)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = str(project_root / output_path)

    # Load held-out questions
    logger.info("Loading held-out questions from %s", held_out_path)
    with open(held_out_path, "r", encoding="utf-8") as f:
        held_out = json.load(f)
    logger.info("Loaded %d held-out questions", len(held_out))

    # Run evaluation
    evaluate(
        base_model_name=args.base_model,
        finetuned_model_dir=finetuned_path,
        held_out=held_out,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
