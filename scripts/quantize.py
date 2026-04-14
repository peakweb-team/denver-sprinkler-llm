#!/usr/bin/env python3
"""
Denver Sprinkler LLM — Dual-Path Quantization Script

Quantizes the merged fine-tuned Llama 3.2 3B model via two paths:
  - BitNet: Post-training ternary quantization using microsoft/BitNet tooling
  - GGUF:   Standard llama.cpp quantization (Q2_K, Q3_K_S, Q4_K_M)

The BitNet path is experimental (post-training quantization to ternary degrades
quality significantly). The GGUF path is recommended for production.

Usage:
    python scripts/quantize.py --method bitnet
    python scripts/quantize.py --method gguf --quant-type Q4_K_M
    python scripts/quantize.py --method all
    python scripts/quantize.py --method all --skip-eval --skip-s3
    python scripts/quantize.py --method all --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

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
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "quantization.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    """Load quantization YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """Validate that all required config keys are present.

    Raises ``SystemExit`` with a clear message when a required key is missing.
    """
    required_keys: list[tuple[str, ...]] = [
        ("merged_model_dir",),
        ("bitnet", "repo_url"),
        ("bitnet", "cache_dir"),
        ("bitnet", "output_dir"),
        ("bitnet", "quant_type"),
        ("bitnet", "preprocess_script"),
        ("bitnet", "convert_script"),
        ("gguf", "repo_url"),
        ("gguf", "cache_dir"),
        ("gguf", "output_dir"),
        ("gguf", "convert_script"),
        ("gguf", "quant_types"),
        ("eval", "held_out_questions_path"),
        ("eval", "reference_eval_path"),
        ("eval", "output_path"),
        ("eval", "max_new_tokens"),
        ("eval", "temperature"),
        ("eval", "system_prompt"),
        ("quality", "min_response_length"),
        ("quality", "max_repetition_ratio"),
        ("quality", "min_coherence_pass_rate"),
        ("s3", "prefix"),
    ]

    missing: list[str] = []
    for key_path in required_keys:
        node = config
        for part in key_path:
            if not isinstance(node, dict) or part not in node:
                missing.append(".".join(key_path))
                break
            node = node[part]

    if missing:
        logger.error("Missing required config keys: %s", ", ".join(missing))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_merged_model(merged_dir: Path, dry_run: bool) -> bool:
    """Check that the merged model directory exists and contains expected files."""
    if not merged_dir.exists():
        if dry_run:
            logger.warning(
                "[DRY RUN] Merged model dir not found: %s (expected — training not yet complete)",
                merged_dir,
            )
            return True
        logger.error("Merged model directory not found: %s", merged_dir)
        logger.error("Run training first (scripts/train.py) to produce the merged model.")
        return False

    # Check for safetensors or bin files
    has_weights = (
        list(merged_dir.glob("*.safetensors"))
        or list(merged_dir.glob("*.bin"))
    )
    has_config = (merged_dir / "config.json").exists()

    if not has_weights:
        if dry_run:
            logger.warning("[DRY RUN] No weight files found in %s", merged_dir)
            return True
        logger.error("No model weight files (.safetensors or .bin) in %s", merged_dir)
        return False

    if not has_config:
        if dry_run:
            logger.warning("[DRY RUN] No config.json found in %s", merged_dir)
            return True
        logger.error("No config.json found in %s", merged_dir)
        return False

    logger.info("Merged model validated at %s", merged_dir)
    return True


def get_dir_size_bytes(directory: Path) -> int:
    """Calculate total size of all files in a directory."""
    total = 0
    if directory.exists():
        for f in directory.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def run_command(cmd: list[str], cwd: str | Path | None = None, dry_run: bool = False) -> int:
    """Run a shell command, logging the invocation. Returns exit code."""
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        logger.info("[DRY RUN] Would run: %s", cmd_str)
        return 0

    logger.info("Running: %s", cmd_str)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Command failed with exit code %d: %s", result.returncode, cmd_str)
    return result.returncode


# ---------------------------------------------------------------------------
# BitNet quantization
# ---------------------------------------------------------------------------


def clone_bitnet_repo(
    cache_dir: Path, repo_url: str, revision: str | None, dry_run: bool
) -> bool:
    """Clone the microsoft/BitNet repo if not already cached."""
    if cache_dir.exists():
        logger.info("BitNet repo already cached at %s", cache_dir)
        return True

    logger.info("Cloning BitNet repo to %s", cache_dir)
    if not dry_run:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
    rc = run_command(
        ["git", "clone", repo_url, str(cache_dir)],
        dry_run=dry_run,
    )
    if rc != 0:
        return False

    if revision:
        logger.info("Checking out BitNet revision %s", revision)
        rc = run_command(
            ["git", "checkout", revision],
            cwd=cache_dir,
            dry_run=dry_run,
        )
        if rc != 0:
            return False

    return True


def build_bitnet_cpp(cache_dir: Path, dry_run: bool) -> bool:
    """Build bitnet.cpp from the cloned repo."""
    build_dir = cache_dir / "build"

    if (build_dir / "bin" / "llama-quantize").exists():
        logger.info("bitnet.cpp already built at %s", build_dir)
        return True

    logger.info("Building bitnet.cpp...")
    if not dry_run:
        build_dir.mkdir(parents=True, exist_ok=True)

    rc = run_command(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
        dry_run=dry_run,
    )
    if rc != 0:
        return False

    rc = run_command(
        ["cmake", "--build", ".", "--config", "Release", "-j"],
        cwd=build_dir,
        dry_run=dry_run,
    )
    return rc == 0


def quantize_bitnet(config: dict, dry_run: bool) -> dict:
    """Run the BitNet ternary quantization pipeline.

    Steps:
      1. Clone microsoft/BitNet repo
      2. Build bitnet.cpp (cmake)
      3. Run preprocess-huggingface-bitnet.py to quantize weights to ternary
      4. Run convert-hf-to-gguf-bitnet.py to convert to GGUF
      5. Run llama-quantize with i2_s
      6. Output to models/denver-sprinkler-3b-1bit/

    Returns a dict with status and artifact paths.
    """
    bitnet_cfg = config["bitnet"]
    merged_dir = PROJECT_ROOT / config["merged_model_dir"]
    cache_dir = PROJECT_ROOT / bitnet_cfg["cache_dir"]
    output_dir = PROJECT_ROOT / bitnet_cfg["output_dir"]
    quant_type = bitnet_cfg["quant_type"]

    result = {
        "method": "bitnet",
        "status": "pending",
        "output_dir": str(output_dir),
        "artifacts": [],
    }

    # Step 1: Clone repo
    logger.info("=" * 60)
    logger.info("BitNet Step 1: Clone repository")
    logger.info("=" * 60)
    if not clone_bitnet_repo(
        cache_dir, bitnet_cfg["repo_url"], bitnet_cfg.get("revision"), dry_run
    ):
        result["status"] = "failed"
        result["error"] = "Failed to clone BitNet repo"
        return result

    # Step 2: Build bitnet.cpp
    logger.info("=" * 60)
    logger.info("BitNet Step 2: Build bitnet.cpp")
    logger.info("=" * 60)
    if not build_bitnet_cpp(cache_dir, dry_run):
        result["status"] = "failed"
        result["error"] = "Failed to build bitnet.cpp"
        return result

    # Step 3: Preprocess — quantize weights to ternary {-1, 0, +1}
    logger.info("=" * 60)
    logger.info("BitNet Step 3: Preprocess weights to ternary")
    logger.info("=" * 60)
    preprocess_script = cache_dir / bitnet_cfg["preprocess_script"]
    preprocessed_dir = output_dir / "preprocessed"
    if not dry_run:
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Copy merged model to preprocessed dir for in-place modification
    if not dry_run:
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
        shutil.copytree(str(merged_dir), str(preprocessed_dir))

    rc = run_command(
        [
            sys.executable,
            str(preprocess_script),
            "--model", str(preprocessed_dir),
        ],
        cwd=cache_dir,
        dry_run=dry_run,
    )
    if rc != 0:
        result["status"] = "failed"
        result["error"] = "Preprocessing (ternary quantization) failed"
        return result

    # Step 4: Convert to GGUF format
    logger.info("=" * 60)
    logger.info("BitNet Step 4: Convert to GGUF format")
    logger.info("=" * 60)
    convert_script = cache_dir / bitnet_cfg["convert_script"]
    gguf_f32_path = output_dir / "ggml-model-f32.gguf"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    rc = run_command(
        [
            sys.executable,
            str(convert_script),
            "--model", str(preprocessed_dir),
            "--outfile", str(gguf_f32_path),
        ],
        cwd=cache_dir,
        dry_run=dry_run,
    )
    if rc != 0:
        result["status"] = "failed"
        result["error"] = "GGUF conversion failed"
        return result

    # Step 5: Run llama-quantize with i2_s
    logger.info("=" * 60)
    logger.info("BitNet Step 5: Quantize with i2_s")
    logger.info("=" * 60)
    llama_quantize = cache_dir / "build" / "bin" / "llama-quantize"
    final_gguf_path = output_dir / f"ggml-model-{quant_type}.gguf"

    rc = run_command(
        [
            str(llama_quantize),
            str(gguf_f32_path),
            str(final_gguf_path),
            quant_type,
        ],
        dry_run=dry_run,
    )
    if rc != 0:
        result["status"] = "failed"
        result["error"] = f"llama-quantize ({quant_type}) failed"
        return result

    # Clean up intermediate files
    if not dry_run:
        if gguf_f32_path.exists():
            gguf_f32_path.unlink()
            logger.info("Removed intermediate GGUF: %s", gguf_f32_path)
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
            logger.info("Removed preprocessed dir: %s", preprocessed_dir)

    result["status"] = "success"
    result["artifacts"] = [str(final_gguf_path)]
    if not dry_run and final_gguf_path.exists():
        result["size_bytes"] = final_gguf_path.stat().st_size
        result["size_human"] = format_size(final_gguf_path.stat().st_size)

    logger.info("BitNet quantization complete: %s", final_gguf_path)
    return result


# ---------------------------------------------------------------------------
# GGUF quantization (llama.cpp)
# ---------------------------------------------------------------------------


def clone_llamacpp_repo(
    cache_dir: Path, repo_url: str, revision: str | None, dry_run: bool
) -> bool:
    """Clone the llama.cpp repo if not already cached."""
    if cache_dir.exists():
        logger.info("llama.cpp repo already cached at %s", cache_dir)
        return True

    logger.info("Cloning llama.cpp repo to %s", cache_dir)
    if not dry_run:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
    rc = run_command(
        ["git", "clone", repo_url, str(cache_dir)],
        dry_run=dry_run,
    )
    if rc != 0:
        return False

    if revision:
        logger.info("Checking out llama.cpp revision %s", revision)
        rc = run_command(
            ["git", "checkout", revision],
            cwd=cache_dir,
            dry_run=dry_run,
        )
        if rc != 0:
            return False

    return True


def build_llamacpp(cache_dir: Path, dry_run: bool) -> bool:
    """Build llama.cpp from the cloned repo."""
    build_dir = cache_dir / "build"

    if (build_dir / "bin" / "llama-quantize").exists():
        logger.info("llama.cpp already built at %s", build_dir)
        return True

    logger.info("Building llama.cpp...")
    if not dry_run:
        build_dir.mkdir(parents=True, exist_ok=True)

    rc = run_command(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
        dry_run=dry_run,
    )
    if rc != 0:
        return False

    rc = run_command(
        ["cmake", "--build", ".", "--config", "Release", "-j"],
        cwd=build_dir,
        dry_run=dry_run,
    )
    return rc == 0


def quantize_gguf(config: dict, quant_types: list[str] | None, dry_run: bool) -> dict:
    """Run the GGUF quantization pipeline via llama.cpp.

    Steps:
      1. Clone llama.cpp repo
      2. Build llama.cpp (cmake)
      3. Run convert_hf_to_gguf.py to convert merged HF model to GGUF F16
      4. Run llama-quantize for each quant type
      5. Output to models/denver-sprinkler-3b-gguf/

    Returns a dict with status and artifact paths.
    """
    gguf_cfg = config["gguf"]
    merged_dir = PROJECT_ROOT / config["merged_model_dir"]
    cache_dir = PROJECT_ROOT / gguf_cfg["cache_dir"]
    output_dir = PROJECT_ROOT / gguf_cfg["output_dir"]

    if quant_types is None:
        quant_types = gguf_cfg["quant_types"]

    result = {
        "method": "gguf",
        "status": "pending",
        "output_dir": str(output_dir),
        "artifacts": [],
        "variants": {},
    }

    # Step 1: Clone repo
    logger.info("=" * 60)
    logger.info("GGUF Step 1: Clone llama.cpp repository")
    logger.info("=" * 60)
    if not clone_llamacpp_repo(
        cache_dir, gguf_cfg["repo_url"], gguf_cfg.get("revision"), dry_run
    ):
        result["status"] = "failed"
        result["error"] = "Failed to clone llama.cpp repo"
        return result

    # Step 2: Build llama.cpp
    logger.info("=" * 60)
    logger.info("GGUF Step 2: Build llama.cpp")
    logger.info("=" * 60)
    if not build_llamacpp(cache_dir, dry_run):
        result["status"] = "failed"
        result["error"] = "Failed to build llama.cpp"
        return result

    # Step 3: Convert HF model to GGUF F16
    logger.info("=" * 60)
    logger.info("GGUF Step 3: Convert HF model to GGUF F16")
    logger.info("=" * 60)
    convert_script = cache_dir / gguf_cfg["convert_script"]
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "ggml-model-f16.gguf"

    rc = run_command(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ],
        cwd=cache_dir,
        dry_run=dry_run,
    )
    if rc != 0:
        result["status"] = "failed"
        result["error"] = "HF-to-GGUF conversion failed"
        return result

    # Step 4: Quantize each variant
    logger.info("=" * 60)
    logger.info("GGUF Step 4: Quantize variants")
    logger.info("=" * 60)
    llama_quantize = cache_dir / "build" / "bin" / "llama-quantize"
    any_success = False

    for qt in quant_types:
        logger.info("Quantizing with %s...", qt)
        quant_path = output_dir / f"ggml-model-{qt}.gguf"

        rc = run_command(
            [
                str(llama_quantize),
                str(f16_path),
                str(quant_path),
                qt,
            ],
            dry_run=dry_run,
        )
        if rc != 0:
            logger.warning("Quantization with %s failed, skipping", qt)
            result["variants"][qt] = {"status": "failed"}
            continue

        any_success = True
        variant_info = {"status": "success", "path": str(quant_path)}
        if not dry_run and quant_path.exists():
            size = quant_path.stat().st_size
            variant_info["size_bytes"] = size
            variant_info["size_human"] = format_size(size)
        result["variants"][qt] = variant_info
        result["artifacts"].append(str(quant_path))

    # Clean up F16 intermediate (large file)
    if not dry_run and f16_path.exists():
        f16_path.unlink()
        logger.info("Removed intermediate F16 GGUF: %s", f16_path)

    result["status"] = "success" if any_success else "failed"
    if not any_success:
        result["error"] = "All quantization variants failed"

    logger.info("GGUF quantization complete: %d variants", len(result["artifacts"]))
    return result


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------


def check_response_quality(response: str, config: dict) -> dict:
    """Run basic quality checks on a single model response.

    Checks:
      - Minimum response length
      - Repetition detection (repeated n-grams)
      - Business detail accuracy (when relevant)

    Returns a dict with pass/fail and details.
    """
    quality_cfg = config["quality"]
    checks = {"passed": True, "issues": []}

    # Length check
    if len(response.strip()) < quality_cfg["min_response_length"]:
        checks["passed"] = False
        checks["issues"].append(
            f"Response too short ({len(response.strip())} chars, "
            f"minimum {quality_cfg['min_response_length']})"
        )

    # Repetition check — look for repeated 4-grams
    words = response.lower().split()
    if len(words) >= 8:
        ngram_size = 4
        ngrams = [
            tuple(words[i : i + ngram_size])
            for i in range(len(words) - ngram_size + 1)
        ]
        if ngrams:
            unique_ngrams = set(ngrams)
            repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
            if repetition_ratio > quality_cfg["max_repetition_ratio"]:
                checks["passed"] = False
                checks["issues"].append(
                    f"High repetition ratio ({repetition_ratio:.2f}, "
                    f"max {quality_cfg['max_repetition_ratio']})"
                )

    # Business details check — only flag if the response seems to
    # attempt providing contact info but gets it wrong
    details = quality_cfg.get("required_business_details", {})
    response_lower = response.lower()

    for field, expected in details.items():
        # Only check if the response appears to reference this field
        triggers = {
            "phone": ["phone", "call", "number", "(303)"],
            "email": ["email", "mail", "@"],
            "address": ["address", "location", "located", "decatur"],
        }
        field_triggers = triggers.get(field, [])
        mentions_field = any(t in response_lower for t in field_triggers)

        if mentions_field and expected.lower() not in response_lower:
            checks["passed"] = False
            checks["issues"].append(
                f"Incorrect {field}: expected '{expected}' but not found in response"
            )

    return checks


def evaluate_quantized_model(
    model_path: str,
    inference_binary: str,
    config: dict,
    dry_run: bool,
) -> dict:
    """Evaluate a quantized GGUF model against held-out questions.

    Uses llama-cli (or bitnet run_inference.py) to generate responses,
    then checks quality.

    Returns eval results dict.
    """
    eval_cfg = config["eval"]
    held_out_path = PROJECT_ROOT / eval_cfg["held_out_questions_path"]
    reference_path = PROJECT_ROOT / eval_cfg["reference_eval_path"]

    eval_result = {
        "model_path": model_path,
        "questions_evaluated": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "details": [],
    }

    if dry_run:
        logger.info("[DRY RUN] Would evaluate %s against held-out questions", model_path)
        eval_result["dry_run"] = True
        return eval_result

    # Load held-out questions
    if not held_out_path.exists():
        logger.warning(
            "Held-out questions not found at %s — skipping eval", held_out_path
        )
        eval_result["skipped"] = True
        eval_result["skip_reason"] = "held_out_questions.json not found"
        return eval_result

    with open(held_out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both bare list [{...}] and wrapper object {"questions": [{...}]}
    if isinstance(data, list):
        held_out = data
    else:
        held_out = data.get("questions", data.get("held_out", []))

    # Load reference eval for comparison (if available)
    reference_answers = {}
    if reference_path.exists():
        with open(reference_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
            for item in ref_data.get("results", []):
                reference_answers[item["question"]] = item.get(
                    "finetuned_model_answer", ""
                )

    system_prompt = eval_cfg["system_prompt"]
    max_tokens = eval_cfg["max_new_tokens"]

    for i, pair in enumerate(held_out):
        question = pair["instruction"]
        logger.info("Evaluating question %d/%d: %s", i + 1, len(held_out), question[:80])

        # Build the prompt in Llama chat format
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # Run inference via CLI
        try:
            proc = subprocess.run(
                [
                    inference_binary,
                    "-m", model_path,
                    "-p", prompt,
                    "-n", str(max_tokens),
                    "--temp", str(eval_cfg["temperature"]),
                    "--top-p", "0.9",
                    "--no-display-prompt",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            response = proc.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Inference failed for question %d: %s", i + 1, e)
            response = ""

        # Quality check
        quality = check_response_quality(response, config)

        detail = {
            "question_index": i,
            "question": question,
            "quantized_answer": response[:500],
            "reference_answer": pair.get("response", "")[:500],
            "pre_quant_answer": reference_answers.get(question, "")[:500],
            "quality_check": quality,
        }
        eval_result["details"].append(detail)
        eval_result["questions_evaluated"] += 1

        if quality["passed"]:
            eval_result["passed"] += 1
        else:
            eval_result["failed"] += 1
            logger.warning(
                "Quality check FAILED for question %d: %s",
                i + 1,
                "; ".join(quality["issues"]),
            )

    if eval_result["questions_evaluated"] > 0:
        eval_result["pass_rate"] = (
            eval_result["passed"] / eval_result["questions_evaluated"]
        )

    return eval_result


# ---------------------------------------------------------------------------
# Size comparison
# ---------------------------------------------------------------------------


def log_size_comparison(
    merged_dir: Path,
    bitnet_result: dict | None,
    gguf_result: dict | None,
    dry_run: bool,
) -> dict:
    """Log and return size comparison between original and quantized models."""
    comparison = {"original": {}, "variants": {}}

    if dry_run:
        # Use placeholder sizes for dry run
        original_size = 6_400_000_000  # ~6.4 GB estimated for 3B params FP16
        logger.info("[DRY RUN] Using estimated FP16 size: %s", format_size(original_size))
    else:
        original_size = get_dir_size_bytes(merged_dir)

    comparison["original"] = {
        "path": str(merged_dir),
        "size_bytes": original_size,
        "size_human": format_size(original_size),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("SIZE COMPARISON")
    logger.info("=" * 60)
    logger.info("Original (FP16): %s", format_size(original_size))

    if bitnet_result and bitnet_result.get("status") == "success":
        size = bitnet_result.get("size_bytes", 0)
        ratio = (size / original_size * 100) if original_size > 0 and size > 0 else 0
        comparison["variants"]["bitnet_i2_s"] = {
            "size_bytes": size,
            "size_human": format_size(size),
            "compression_ratio": f"{ratio:.1f}%",
        }
        logger.info(
            "BitNet i2_s:     %s (%.1f%% of original)",
            format_size(size),
            ratio,
        )

    if gguf_result and gguf_result.get("status") == "success":
        for qt, info in gguf_result.get("variants", {}).items():
            if info.get("status") != "success":
                continue
            size = info.get("size_bytes", 0)
            ratio = (size / original_size * 100) if original_size > 0 and size > 0 else 0
            comparison["variants"][f"gguf_{qt}"] = {
                "size_bytes": size,
                "size_human": format_size(size),
                "compression_ratio": f"{ratio:.1f}%",
            }
            logger.info(
                "GGUF %-10s  %s (%.1f%% of original)",
                qt + ":",
                format_size(size),
                ratio,
            )

    logger.info("=" * 60)
    return comparison


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def upload_to_s3(config: dict, results: dict, dry_run: bool):
    """Upload quantized model artifacts to S3."""
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed. Skipping S3 upload.")
        return

    s3_bucket = os.environ.get("S3_BUCKET_NAME")
    if not s3_bucket:
        logger.warning("S3_BUCKET_NAME not set. Skipping S3 upload.")
        return

    s3_prefix = config["s3"]["prefix"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    s3_prefix = f"{s3_prefix}/{timestamp}"

    if dry_run:
        logger.info("[DRY RUN] Would upload to s3://%s/%s/", s3_bucket, s3_prefix)
        return

    s3 = boto3.client("s3")
    total_uploaded = 0

    # Collect all artifact paths
    artifact_paths = []
    for method_result in [results.get("bitnet"), results.get("gguf")]:
        if method_result and method_result.get("status") == "success":
            for artifact in method_result.get("artifacts", []):
                artifact_paths.append(Path(artifact))

    # Upload quantized models
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            s3_key = f"{s3_prefix}/{artifact_path.name}"
            logger.info(
                "Uploading %s -> s3://%s/%s", artifact_path, s3_bucket, s3_key
            )
            s3.upload_file(str(artifact_path), s3_bucket, s3_key)
            total_uploaded += 1

    # Upload eval results if they exist
    eval_path = PROJECT_ROOT / config["eval"]["output_path"]
    if eval_path.exists():
        s3_key = f"{s3_prefix}/quantized_eval_results.json"
        logger.info("Uploading %s -> s3://%s/%s", eval_path, s3_bucket, s3_key)
        s3.upload_file(str(eval_path), s3_bucket, s3_key)
        total_uploaded += 1

    # Upload model card
    model_card_path = PROJECT_ROOT / "models" / "MODEL_CARD.md"
    if model_card_path.exists():
        s3_key = f"{s3_prefix}/MODEL_CARD.md"
        logger.info("Uploading %s -> s3://%s/%s", model_card_path, s3_bucket, s3_key)
        s3.upload_file(str(model_card_path), s3_bucket, s3_key)
        total_uploaded += 1

    # Upload quantization config
    config_path = PROJECT_ROOT / "configs" / "quantization.yaml"
    if config_path.exists():
        s3_key = f"{s3_prefix}/quantization_config.yaml"
        s3.upload_file(str(config_path), s3_bucket, s3_key)
        total_uploaded += 1

    logger.info(
        "S3 upload complete: %d files to s3://%s/%s/",
        total_uploaded,
        s3_bucket,
        s3_prefix,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: dict):
    """Print a final summary of all quantization results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 60)

    for method in ["bitnet", "gguf"]:
        method_result = results.get(method)
        if not method_result:
            continue

        status = method_result["status"]
        logger.info("")
        logger.info("  %s: %s", method.upper(), status.upper())

        if status == "success":
            for artifact in method_result.get("artifacts", []):
                logger.info("    - %s", artifact)
            if method == "gguf":
                for qt, info in method_result.get("variants", {}).items():
                    size = info.get("size_human", "unknown")
                    logger.info("    %s: %s (%s)", qt, info["status"], size)
        elif status == "failed":
            logger.info("    Error: %s", method_result.get("error", "unknown"))

    # Eval summary
    eval_results = results.get("eval")
    if eval_results:
        logger.info("")
        logger.info("  EVALUATION:")
        for model_path, eval_data in eval_results.items():
            pass_rate = eval_data.get("pass_rate", 0)
            total = eval_data.get("questions_evaluated", 0)
            logger.info(
                "    %s: %.0f%% pass rate (%d questions)",
                Path(model_path).name,
                pass_rate * 100,
                total,
            )

    # Recommendation
    bitnet_result = results.get("bitnet")
    gguf_result = results.get("gguf")

    logger.info("")
    logger.info("  RECOMMENDATION:")
    if bitnet_result and bitnet_result.get("status") == "success":
        # Check if eval passed
        bitnet_eval = (results.get("eval") or {}).get(
            str(bitnet_result.get("artifacts", [""])[0]), {}
        )
        bitnet_pass_rate = bitnet_eval.get("pass_rate", 0)
        min_pass = results.get("_quality_threshold", 0.7)

        if bitnet_pass_rate >= min_pass:
            logger.info(
                "    BitNet i2_s passed quality checks (%.0f%% pass rate).",
                bitnet_pass_rate * 100,
            )
            logger.info("    Use for bitnet.cpp inference if latency is critical.")
        else:
            logger.warning(
                "    BitNet i2_s FAILED quality checks (%.0f%% pass rate, need %.0f%%).",
                bitnet_pass_rate * 100,
                min_pass * 100,
            )
            logger.warning("    This is expected for post-training ternary quantization.")

    if gguf_result and gguf_result.get("status") == "success":
        logger.info(
            "    GGUF Q4_K_M is recommended for production (best quality/size tradeoff)."
        )
        logger.info("    Serve via llama.cpp server for production inference.")

    logger.info("")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Denver Sprinkler fine-tuned model (BitNet and/or GGUF)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bitnet", "gguf", "all"],
        default="all",
        help="Quantization method: bitnet, gguf, or all (default: all)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default=None,
        help="Specific GGUF quant type (e.g., Q4_K_M). Overrides config list.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to quantization config YAML",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip quality evaluation after quantization",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Skip S3 upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate structure without running actual quantization (no model artifacts needed)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)

    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)
    validate_config(config)

    merged_dir = PROJECT_ROOT / config["merged_model_dir"]

    # Validate merged model
    logger.info("=" * 60)
    logger.info("Validating merged model")
    logger.info("=" * 60)
    if not validate_merged_model(merged_dir, args.dry_run):
        sys.exit(1)

    results = {"eval": {}}
    run_bitnet = args.method in ("bitnet", "all")
    run_gguf = args.method in ("gguf", "all")

    # Determine GGUF quant types
    gguf_quant_types = None
    if args.quant_type:
        gguf_quant_types = [args.quant_type]

    # -----------------------------------------------------------------------
    # BitNet quantization
    # -----------------------------------------------------------------------
    if run_bitnet:
        logger.info("")
        logger.info("=" * 60)
        logger.info("BITNET QUANTIZATION")
        logger.info("=" * 60)
        results["bitnet"] = quantize_bitnet(config, args.dry_run)

        if results["bitnet"]["status"] == "failed":
            logger.warning(
                "BitNet quantization failed: %s",
                results["bitnet"].get("error", "unknown"),
            )
            if not run_gguf:
                logger.error("No fallback — GGUF not requested. Exiting.")
                sys.exit(1)
            logger.info("Falling back to GGUF quantization...")

    # -----------------------------------------------------------------------
    # GGUF quantization
    # -----------------------------------------------------------------------
    if run_gguf:
        logger.info("")
        logger.info("=" * 60)
        logger.info("GGUF QUANTIZATION")
        logger.info("=" * 60)
        results["gguf"] = quantize_gguf(config, gguf_quant_types, args.dry_run)

    # -----------------------------------------------------------------------
    # Size comparison
    # -----------------------------------------------------------------------
    logger.info("")
    results["size_comparison"] = log_size_comparison(
        merged_dir,
        results.get("bitnet"),
        results.get("gguf"),
        args.dry_run,
    )

    # -----------------------------------------------------------------------
    # Quality evaluation
    # -----------------------------------------------------------------------
    if not args.skip_eval:
        logger.info("")
        logger.info("=" * 60)
        logger.info("QUALITY EVALUATION")
        logger.info("=" * 60)

        # Evaluate BitNet model
        if run_bitnet and results.get("bitnet", {}).get("status") == "success":
            bitnet_artifacts = results["bitnet"].get("artifacts", [])
            if bitnet_artifacts:
                bitnet_cache = PROJECT_ROOT / config["bitnet"]["cache_dir"]
                # bitnet.cpp uses llama-cli in its build
                bitnet_cli = str(bitnet_cache / "build" / "bin" / "llama-cli")
                results["eval"][bitnet_artifacts[0]] = evaluate_quantized_model(
                    bitnet_artifacts[0], bitnet_cli, config, args.dry_run
                )

        # Evaluate GGUF models
        if run_gguf and results.get("gguf", {}).get("status") == "success":
            llamacpp_cache = PROJECT_ROOT / config["gguf"]["cache_dir"]
            llama_cli = str(llamacpp_cache / "build" / "bin" / "llama-cli")
            for artifact in results["gguf"].get("artifacts", []):
                results["eval"][artifact] = evaluate_quantized_model(
                    artifact, llama_cli, config, args.dry_run
                )

        # Save eval results
        results["_quality_threshold"] = config["quality"]["min_coherence_pass_rate"]
        eval_output_path = PROJECT_ROOT / config["eval"]["output_path"]
        if not args.dry_run:
            eval_output_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            with open(eval_output_path, "w", encoding="utf-8") as f:
                json.dump(results["eval"], f, indent=2, ensure_ascii=False)
            logger.info("Eval results saved to %s", eval_output_path)
        else:
            logger.info(
                "[DRY RUN] Would save eval results to %s", eval_output_path
            )

        # Check for BitNet quality failure -> automatic GGUF fallback
        if run_bitnet and results.get("bitnet", {}).get("status") == "success":
            bitnet_artifacts = results["bitnet"].get("artifacts", [])
            if bitnet_artifacts:
                bitnet_eval = results["eval"].get(bitnet_artifacts[0], {})
                pass_rate = bitnet_eval.get("pass_rate", 0)
                min_rate = config["quality"]["min_coherence_pass_rate"]
                if pass_rate < min_rate and not bitnet_eval.get("dry_run"):
                    logger.warning("")
                    logger.warning("=" * 60)
                    logger.warning("BITNET QUALITY BELOW THRESHOLD")
                    logger.warning("=" * 60)
                    logger.warning(
                        "BitNet pass rate: %.0f%% (threshold: %.0f%%)",
                        pass_rate * 100,
                        min_rate * 100,
                    )
                    logger.warning(
                        "This is expected for post-training ternary quantization."
                    )

                    # Mark BitNet output as experimental
                    results["bitnet"]["experimental"] = True
                    logger.warning(
                        "BitNet output marked as experimental due to quality failure."
                    )

                    # Automatically trigger GGUF fallback if not already run
                    if not run_gguf:
                        logger.warning(
                            "Automatically triggering GGUF quantization as fallback..."
                        )
                        results["gguf"] = quantize_gguf(
                            config, gguf_quant_types, args.dry_run
                        )
                        # Evaluate the GGUF fallback models
                        if results["gguf"].get("status") == "success":
                            llamacpp_cache = (
                                PROJECT_ROOT / config["gguf"]["cache_dir"]
                            )
                            llama_cli = str(
                                llamacpp_cache / "build" / "bin" / "llama-cli"
                            )
                            for artifact in results["gguf"].get("artifacts", []):
                                results["eval"][artifact] = (
                                    evaluate_quantized_model(
                                        artifact, llama_cli, config, args.dry_run
                                    )
                                )

                    logger.warning(
                        "Recommendation: Use GGUF Q4_K_M for production."
                    )
                    logger.warning(
                        "For true 1-bit performance, fine-tune from a "
                        "BitNet-native base model."
                    )
    else:
        logger.info("Skipping quality evaluation (--skip-eval)")

    # -----------------------------------------------------------------------
    # S3 upload
    # -----------------------------------------------------------------------
    if not args.skip_s3:
        logger.info("")
        logger.info("=" * 60)
        logger.info("S3 UPLOAD")
        logger.info("=" * 60)
        upload_to_s3(config, results, args.dry_run)
    else:
        logger.info("Skipping S3 upload (--skip-s3)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("Total elapsed time: %.1f seconds", elapsed)
    print_summary(results)


if __name__ == "__main__":
    main()
