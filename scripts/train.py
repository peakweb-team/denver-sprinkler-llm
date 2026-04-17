#!/usr/bin/env python3
"""
Denver Sprinkler LLM — LoRA Fine-Tuning Script

Fine-tunes Llama 3.2 3B-Instruct with LoRA on the Denver Sprinkler Q&A dataset.
Designed to run on a g5.xlarge (NVIDIA A10G, 24 GB VRAM).

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training.yaml
    python scripts/train.py --config configs/training.yaml --num_epochs 5 --learning_rate 1e-4
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

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
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "configs/training.yaml"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, overrides: dict) -> dict:
    """Apply CLI argument overrides to config. Only override non-None values."""
    for key, value in overrides.items():
        if value is not None and key in config:
            # Coerce type to match config
            original = config[key]
            if isinstance(original, bool):
                config[key] = str(value).lower() in ("true", "1", "yes")
            elif isinstance(original, int):
                config[key] = int(value)
            elif isinstance(original, float):
                config[key] = float(value)
            else:
                config[key] = value
    return config


# ---------------------------------------------------------------------------
# Data loading & formatting
# ---------------------------------------------------------------------------


def load_data(data_path: str) -> list[dict]:
    """Load Q&A pairs from JSONL file."""
    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON on line %d: %s", line_num, e)
    logger.info("Loaded %d Q&A pairs from %s", len(pairs), data_path)
    return pairs


def split_data(
    pairs: list[dict], train_ratio: float, held_out_count: int, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data into train, validation, and held-out eval sets.

    The held-out eval set is drawn from the validation portion so that
    training data is never contaminated.
    """
    import random

    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    # Reserve held-out questions from validation set
    held_out = val_data[:held_out_count]
    val_data = val_data[held_out_count:]

    logger.info(
        "Split: %d train, %d validation, %d held-out eval",
        len(train_data),
        len(val_data),
        len(held_out),
    )
    return train_data, val_data, held_out


def format_chat_examples(
    pairs: list[dict], tokenizer, system_prompt: str
) -> list[str]:
    """Format Q&A pairs into Llama 3.2 chat template strings."""
    formatted = []
    for pair in pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append(text)
    return formatted


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def load_base_model(model_name: str, fp16: bool):
    """Load the base model and tokenizer."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Model download may fail for gated models.")

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model: %s (fp16=%s)", model_name, fp16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    return model, tokenizer


def apply_lora(model, config: dict):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: %s trainable params / %s total (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config: dict):
    """Run the full training pipeline."""
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / config["data_path"]
    output_dir = project_root / config["output_dir"]
    merged_dir = project_root / config["merged_output_dir"]
    logging_dir = project_root / config["logging_dir"]

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load and split data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Loading and splitting data")
    logger.info("=" * 60)

    pairs = load_data(str(data_path))
    train_data, val_data, held_out = split_data(
        pairs, config["train_split"], config["held_out_count"], config["seed"]
    )

    # Save held-out questions for later evaluation
    held_out_path = project_root / "models" / "held_out_questions.json"
    with open(held_out_path, "w", encoding="utf-8") as f:
        json.dump(held_out, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d held-out questions to %s", len(held_out), held_out_path)

    # -----------------------------------------------------------------------
    # Step 2: Load model and tokenizer
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Loading base model and tokenizer")
    logger.info("=" * 60)

    model, tokenizer = load_base_model(config["base_model"], config["fp16"])

    # -----------------------------------------------------------------------
    # Step 3: Format data into chat template
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Formatting data into chat template")
    logger.info("=" * 60)

    train_texts = format_chat_examples(train_data, tokenizer, config["system_prompt"])
    val_texts = format_chat_examples(val_data, tokenizer, config["system_prompt"])

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    logger.info("Train dataset: %d examples", len(train_dataset))
    logger.info("Validation dataset: %d examples", len(val_dataset))

    # Log a sample for sanity check
    if len(train_texts) > 0:
        logger.info("Sample formatted example (first 500 chars):\n%s", train_texts[0][:500])

    # -----------------------------------------------------------------------
    # Step 4: Apply LoRA
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Applying LoRA adapters")
    logger.info("=" * 60)

    model = apply_lora(model, config)

    # -----------------------------------------------------------------------
    # Step 5: Configure training
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Configuring training")
    logger.info("=" * 60)

    import os as _os
    _os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(logging_dir))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=max(1, int(config["warmup_ratio"] * 177)),
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        max_grad_norm=config["max_grad_norm"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=config["seed"],
        remove_unused_columns=False,
        max_length=config["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # -----------------------------------------------------------------------
    # Step 6: Train
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6: Training")
    logger.info("=" * 60)

    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # -----------------------------------------------------------------------
    # Step 7: Save LoRA adapter
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 7: Saving LoRA adapter")
    logger.info("=" * 60)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("LoRA adapter saved to %s", output_dir)

    # -----------------------------------------------------------------------
    # Step 8: Merge and save full model
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 8: Merging LoRA weights and saving full model")
    logger.info("=" * 60)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("Merged model saved to %s", merged_dir)

    # Free training VRAM before loading eval models
    del merged_model
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Freed training VRAM for evaluation")

    # -----------------------------------------------------------------------
    # Step 9: Run held-out evaluation
    # -----------------------------------------------------------------------
    eval_path = project_root / "models" / "eval_results.json"
    if config.get("_skip_eval"):
        logger.info("Skipping held-out evaluation (--skip_eval flag set)")
    else:
        logger.info("=" * 60)
        logger.info("Step 9: Running held-out evaluation")
        logger.info("=" * 60)

        eval_results = run_held_out_eval(
            base_model_name=config["base_model"],
            finetuned_model_dir=str(merged_dir),
            held_out=held_out,
            system_prompt=config["system_prompt"],
            temperature=config.get("eval_temperature", 0.7),
            max_new_tokens=config.get("eval_max_new_tokens", 300),
        )

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation results saved to %s", eval_path)

    # -----------------------------------------------------------------------
    # Step 10: Upload artifacts to S3
    # -----------------------------------------------------------------------
    if config.get("_skip_s3"):
        logger.info("Skipping S3 upload (--skip_s3 flag set)")
    else:
        logger.info("=" * 60)
        logger.info("Step 10: Uploading artifacts to S3")
        logger.info("=" * 60)

        upload_to_s3(config, project_root, timestamp)

    # -----------------------------------------------------------------------
    # Step 11: Print cost summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    elapsed_hours = elapsed / 3600
    spot_rate = config.get("spot_price_per_hour", 0.50)
    estimated_cost = elapsed_hours * spot_rate

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Elapsed time: %.1f minutes (%.2f hours)", elapsed / 60, elapsed_hours)
    logger.info("Spot rate: $%.2f/hr", spot_rate)
    logger.info("Estimated cost: $%.2f", estimated_cost)
    logger.info("Adapter saved to: %s", output_dir)
    logger.info("Merged model saved to: %s", merged_dir)
    logger.info("Eval results: %s", eval_path)

    # Save cost summary
    cost_summary = {
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed_hours, 4),
        "spot_rate_per_hour": spot_rate,
        "estimated_cost_usd": round(estimated_cost, 2),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_samples": len(train_dataset),
        "eval_samples": len(val_dataset),
        "held_out_count": len(held_out),
    }
    cost_path = project_root / "models" / "cost_summary.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2, ensure_ascii=False)
    logger.info("Cost summary saved to %s", cost_path)


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------


def run_held_out_eval(
    base_model_name: str,
    finetuned_model_dir: str,
    held_out: list[dict],
    system_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 300,
) -> list[dict]:
    """Run inference on held-out questions with both base and fine-tuned models."""
    hf_token = os.environ.get("HF_TOKEN")

    results = []

    # Load fine-tuned model (already merged)
    logger.info("Loading fine-tuned model from %s for evaluation", finetuned_model_dir)
    ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)
    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load base model
    logger.info("Loading base model %s for comparison", base_model_name)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for i, pair in enumerate(held_out):
        logger.info("Evaluating held-out question %d/%d", i + 1, len(held_out))
        question = pair["instruction"]
        reference = pair["response"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Generate from fine-tuned model
        ft_answer = generate_response(
            ft_model, ft_tokenizer, messages, temperature, max_new_tokens
        )

        # Generate from base model
        base_answer = generate_response(
            base_model, base_tokenizer, messages, temperature, max_new_tokens
        )

        results.append(
            {
                "question_index": i,
                "question": question,
                "reference_answer": reference,
                "base_model_answer": base_answer,
                "finetuned_model_answer": ft_answer,
                "source": pair.get("source", ""),
                "source_ref": pair.get("source_ref", ""),
            }
        )

    # Clean up base model to free VRAM
    del base_model
    del ft_model
    torch.cuda.empty_cache()

    return results


def generate_response(
    model, tokenizer, messages: list[dict], temperature: float, max_new_tokens: int
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

    # Decode only the generated tokens (exclude the prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def upload_to_s3(config: dict, project_root: Path, timestamp: str):
    """Upload training artifacts to S3."""
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed. Skipping S3 upload.")
        return

    s3_bucket = os.environ.get("S3_BUCKET_NAME")
    if not s3_bucket:
        logger.warning("S3_BUCKET_NAME not set. Skipping S3 upload.")
        return

    s3_prefix = config.get("s3_prefix", "models/llama-3.2-3b-lora")
    s3_prefix = f"{s3_prefix}/{timestamp}"

    s3 = boto3.client("s3")

    upload_dirs = [
        (project_root / config["output_dir"], "adapter"),
        (project_root / config["merged_output_dir"], "merged"),
        (project_root / config["logging_dir"], "training_logs"),
    ]

    upload_files = [
        (project_root / "models" / "eval_results.json", "eval_results.json"),
        (project_root / "models" / "cost_summary.json", "cost_summary.json"),
        (project_root / "models" / "held_out_questions.json", "held_out_questions.json"),
        (project_root / "configs" / "training.yaml", "training_config.yaml"),
    ]

    total_uploaded = 0

    # Upload directories
    for local_dir, s3_subdir in upload_dirs:
        if not local_dir.exists():
            logger.warning("Directory not found, skipping: %s", local_dir)
            continue
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{s3_subdir}/{rel_path}"
                logger.info("Uploading %s -> s3://%s/%s", file_path, s3_bucket, s3_key)
                s3.upload_file(str(file_path), s3_bucket, s3_key)
                total_uploaded += 1

    # Upload individual files
    for local_file, s3_name in upload_files:
        if not local_file.exists():
            logger.warning("File not found, skipping: %s", local_file)
            continue
        s3_key = f"{s3_prefix}/{s3_name}"
        logger.info("Uploading %s -> s3://%s/%s", local_file, s3_bucket, s3_key)
        s3.upload_file(str(local_file), s3_bucket, s3_key)
        total_uploaded += 1

    logger.info("S3 upload complete: %d files to s3://%s/%s/", total_uploaded, s3_bucket, s3_prefix)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 3B with LoRA for Denver Sprinkler"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config YAML (default: configs/training.yaml)",
    )
    # Allow any config key to be overridden via CLI
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--merged_output_dir", type=str, default=None)
    parser.add_argument("--skip_eval", action="store_true", help="Skip held-out evaluation")
    parser.add_argument("--skip_s3", action="store_true", help="Skip S3 upload")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        project_root = Path(__file__).resolve().parent.parent
        config_path = str(project_root / config_path)

    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    # Apply CLI overrides
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("config", "skip_eval", "skip_s3")
    }
    config = apply_cli_overrides(config, overrides)

    # Store flags in config for downstream use
    config["_skip_eval"] = args.skip_eval
    config["_skip_s3"] = args.skip_s3

    # Log final config
    logger.info("Final training config:")
    for key, value in sorted(config.items()):
        if not key.startswith("_"):
            logger.info("  %s: %s", key, value)

    # Verify CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
    else:
        logger.warning("No CUDA GPU detected! Training will be very slow on CPU.")

    # Run training
    train(config)


if __name__ == "__main__":
    main()
