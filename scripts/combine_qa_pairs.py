#!/usr/bin/env python3
"""
Combine, deduplicate, and validate Q&A training pair batches.

Reads all JSONL batch files from data/qa-batches/, removes near-duplicate
questions, validates business details, and writes the final training corpus.

Reads:
  - data/qa-batches/*.jsonl  (batch files from parallel generation)

Writes:
  - data/training-pairs.jsonl
"""

import glob
import json
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
BATCH_DIR = BASE_DIR / "data" / "qa-batches"
OUTPUT_FILE = BASE_DIR / "data" / "training-pairs.jsonl"

REQUIRED_FIELDS = {"instruction", "response", "source", "source_ref"}
VALID_SOURCES = {"site", "rag"}

# Business details that must be exact
CANONICAL_PHONE = "(303) 993-8717"
CANONICAL_EMAIL = "info@denversprinklerservices.com"
CANONICAL_ADDRESS = "3971 S Decatur St Unit A"

# Dedup threshold
DEDUP_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def load_batches() -> list[dict]:
    """Load all JSONL batch files from the qa-batches directory."""
    pairs = []
    errors = 0
    for fpath in sorted(BATCH_DIR.glob("*.jsonl")):
        for line_num, line in enumerate(open(fpath, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not REQUIRED_FIELDS.issubset(rec.keys()):
                    missing = REQUIRED_FIELDS - rec.keys()
                    print(f"  WARN: {fpath.name}:{line_num} missing fields: {missing}")
                    errors += 1
                    continue
                pairs.append(rec)
            except json.JSONDecodeError as e:
                print(f"  WARN: {fpath.name}:{line_num} invalid JSON: {e}")
                errors += 1
    return pairs, errors


# ---------------------------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Normalize a string for comparison."""
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()


def deduplicate(pairs: list[dict]) -> tuple[list[dict], int]:
    """Remove near-duplicate questions using SequenceMatcher."""
    normalized = [(normalize(p["instruction"]), i) for i, p in enumerate(pairs)]
    to_remove = set()

    # Sort by normalized instruction for efficient nearby comparison
    normalized_sorted = sorted(normalized, key=lambda x: x[0])

    for i in range(len(normalized_sorted)):
        if normalized_sorted[i][1] in to_remove:
            continue
        for j in range(i + 1, min(i + 50, len(normalized_sorted))):
            if normalized_sorted[j][1] in to_remove:
                continue
            ratio = SequenceMatcher(
                None, normalized_sorted[i][0], normalized_sorted[j][0]
            ).ratio()
            if ratio > DEDUP_THRESHOLD:
                idx_i = normalized_sorted[i][1]
                idx_j = normalized_sorted[j][1]
                # Keep the one with the longer response
                if len(pairs[idx_i]["response"]) >= len(pairs[idx_j]["response"]):
                    to_remove.add(idx_j)
                else:
                    to_remove.add(idx_i)

    deduped = [p for i, p in enumerate(pairs) if i not in to_remove]
    return deduped, len(to_remove)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(pairs: list[dict]) -> bool:
    """Run validation checks on the final corpus."""
    all_passed = True
    total_checks = 0
    passed_checks = 0

    # Check 1: Minimum count
    total_checks += 1
    if len(pairs) >= 1000:
        print(f"  PASS: Pair count ({len(pairs)}) >= 1000")
        passed_checks += 1
    else:
        print(f"  FAIL: Pair count ({len(pairs)}) < 1000 target")
        all_passed = False

    # Check 2: Valid sources
    total_checks += 1
    invalid_sources = [p for p in pairs if p["source"] not in VALID_SOURCES]
    if not invalid_sources:
        source_counts = Counter(p["source"] for p in pairs)
        print(f"  PASS: All sources valid ({dict(source_counts)})")
        passed_checks += 1
    else:
        print(f"  FAIL: {len(invalid_sources)} invalid source values")
        all_passed = False

    # Check 3: No empty fields
    total_checks += 1
    empty = sum(1 for p in pairs if not p["instruction"].strip() or not p["response"].strip())
    if empty == 0:
        print("  PASS: No empty instruction/response fields")
        passed_checks += 1
    else:
        print(f"  FAIL: {empty} pairs with empty fields")
        all_passed = False

    # Check 4: Source attribution
    total_checks += 1
    no_ref = sum(1 for p in pairs if not p.get("source_ref", "").strip())
    if no_ref == 0:
        print("  PASS: All pairs have source_ref")
        passed_checks += 1
    else:
        print(f"  FAIL: {no_ref} pairs missing source_ref")
        all_passed = False

    # Check 5: Business detail accuracy
    total_checks += 1
    phone_pattern = re.compile(r"\(\d{3}\)\s*\d{3}[-.]\d{4}")
    wrong_phones = 0
    for p in pairs:
        for ph in phone_pattern.findall(p["response"]):
            if ph != CANONICAL_PHONE:
                wrong_phones += 1
    if wrong_phones == 0:
        print("  PASS: No incorrect phone numbers")
        passed_checks += 1
    else:
        print(f"  FAIL: {wrong_phones} incorrect phone numbers found")
        all_passed = False

    # Check 6: Response length
    total_checks += 1
    short = sum(1 for p in pairs if len(p["response"]) < 20)
    if short == 0:
        print("  PASS: No extremely short responses (<20 chars)")
        passed_checks += 1
    else:
        print(f"  WARN: {short} responses under 20 chars")
        all_passed = False

    # Check 7: Question diversity
    total_checks += 1
    starters = Counter()
    for p in pairs:
        words = p["instruction"].split()[:3]
        starters[" ".join(words)] += 1
    most_common_start, most_common_count = starters.most_common(1)[0]
    pct = most_common_count / len(pairs) * 100
    if pct < 30:
        print(f"  PASS: Question diversity OK (most common start: '{most_common_start}' at {pct:.1f}%)")
        passed_checks += 1
    else:
        print(f"  WARN: Low question diversity ('{most_common_start}' at {pct:.1f}%)")
        all_passed = False

    # Summary
    print(f"\nResults: {passed_checks}/{total_checks} checks passed")

    # Quality metrics
    resp_lens = [len(p["response"]) for p in pairs]
    instr_lens = [len(p["instruction"]) for p in pairs]
    print(f"\nContent metrics:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Avg response length: {sum(resp_lens) / len(resp_lens):.0f} chars")
    print(f"  Avg question length: {sum(instr_lens) / len(instr_lens):.0f} chars")
    print(f"  Source breakdown: {dict(Counter(p['source'] for p in pairs))}")

    # Spot-check sample
    print(f"\n{'='*60}")
    print("SPOT CHECK — 5 random samples:")
    print(f"{'='*60}")
    for p in random.sample(pairs, min(5, len(pairs))):
        print(f"\n  Q: {p['instruction']}")
        print(f"  A: {p['response'][:200]}{'...' if len(p['response']) > 200 else ''}")
        print(f"  [{p['source']}] {p['source_ref']}")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Q&A Training Pairs — Combine, Dedup, Validate")
    print("=" * 60)

    # Load batches
    print(f"\nLoading batches from {BATCH_DIR}/...")
    pairs, load_errors = load_batches()
    print(f"  Loaded {len(pairs)} pairs ({load_errors} errors)")

    # Deduplicate
    print("\nDeduplicating...")
    deduped, removed = deduplicate(pairs)
    print(f"  Removed {removed} near-duplicates")
    print(f"  Remaining: {len(deduped)} pairs")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(deduped)} pairs to {OUTPUT_FILE}")

    # Validate
    print("\nValidation:")
    passed = validate(deduped)

    if passed:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION FAILED")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
