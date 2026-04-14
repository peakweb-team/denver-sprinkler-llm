#!/usr/bin/env python3
"""
Validate the RAG sources JSONL file for quality and completeness.

Checks:
- Minimum 200 document chunks
- Required fields present in every record
- No empty content
- Valid categories
- Source attribution on every chunk
- Category distribution (all 4 categories represented)
- No obvious HTML/JSX artifacts in content
- Content quality metrics

Usage:
    python scripts/validate_rag_sources.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "rag-sources.jsonl"
REQUIRED_FIELDS = {"source_url", "source_name", "title", "content", "category"}
VALID_CATEGORIES = {"irrigation", "landscaping", "hardscaping", "climate"}
MIN_CHUNKS = 200
MIN_CONTENT_WORDS = 20

# HTML/JSX artifacts that should not appear in clean text
HTML_ARTIFACT_PATTERNS = [
    re.compile(r"<(?:div|span|a|p|img|br|hr|ul|ol|li|table|tr|td|th|form|input|button|script|style|link|meta)\b", re.IGNORECASE),
    re.compile(r"</(?:div|span|a|p|img|br|hr|ul|ol|li|table|tr|td|th|form|input|button|script|style|link|meta)>", re.IGNORECASE),
    re.compile(r"className="),
    re.compile(r"onClick="),
    re.compile(r"<\?php"),
    re.compile(r"\{%\s"),  # Jinja/Django template tags
    re.compile(r"<%[=\s]"),  # ERB/ASP tags
]


def validate() -> bool:
    """Run all validation checks. Returns True if all pass."""
    if not INPUT_FILE.exists():
        print(f"FAIL: Input file not found: {INPUT_FILE}")
        return False

    # Load records
    records: list[dict] = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"FAIL: Invalid JSON on line {line_num}: {e}")
                return False

    print(f"Loaded {len(records)} records from {INPUT_FILE}")
    print()

    all_passed = True

    # --- Check 1: Minimum chunk count ---
    if len(records) >= MIN_CHUNKS:
        print(f"PASS: Chunk count ({len(records)}) >= {MIN_CHUNKS}")
    else:
        print(f"FAIL: Chunk count ({len(records)}) < {MIN_CHUNKS}")
        all_passed = False

    # --- Check 2: Required fields present ---
    missing_fields_count = 0
    for i, rec in enumerate(records):
        missing = REQUIRED_FIELDS - set(rec.keys())
        if missing:
            if missing_fields_count < 5:
                print(f"FAIL: Record {i} missing fields: {missing}")
            missing_fields_count += 1
    if missing_fields_count == 0:
        print("PASS: All records have required fields")
    else:
        print(f"FAIL: {missing_fields_count} records missing required fields")
        all_passed = False

    # --- Check 3: No empty content ---
    empty_count = sum(1 for r in records if not r.get("content", "").strip())
    if empty_count == 0:
        print("PASS: No empty content fields")
    else:
        print(f"FAIL: {empty_count} records have empty content")
        all_passed = False

    # --- Check 4: Valid categories ---
    invalid_cats = []
    for i, rec in enumerate(records):
        cat = rec.get("category", "")
        if cat not in VALID_CATEGORIES:
            invalid_cats.append((i, cat))
    if not invalid_cats:
        print("PASS: All categories are valid")
    else:
        print(f"FAIL: {len(invalid_cats)} records have invalid categories")
        for idx, cat in invalid_cats[:5]:
            print(f"  Record {idx}: '{cat}'")
        all_passed = False

    # --- Check 5: Source attribution ---
    no_url = sum(1 for r in records if not r.get("source_url", "").strip())
    no_name = sum(1 for r in records if not r.get("source_name", "").strip())
    if no_url == 0 and no_name == 0:
        print("PASS: All records have source attribution (source_url and source_name)")
    else:
        if no_url:
            print(f"FAIL: {no_url} records missing source_url")
        if no_name:
            print(f"FAIL: {no_name} records missing source_name")
        all_passed = False

    # --- Check 6: Category distribution ---
    cat_counter = Counter(r.get("category", "") for r in records)
    represented = VALID_CATEGORIES & set(cat_counter.keys())
    missing_cats = VALID_CATEGORIES - represented
    if not missing_cats:
        print(f"PASS: All 4 categories represented: {dict(cat_counter)}")
    else:
        print(f"FAIL: Missing categories: {missing_cats}")
        print(f"  Found: {dict(cat_counter)}")
        all_passed = False

    # --- Check 7: No HTML/JSX artifacts ---
    artifact_count = 0
    artifact_examples: list[tuple[int, str]] = []
    for i, rec in enumerate(records):
        content = rec.get("content", "")
        for pattern in HTML_ARTIFACT_PATTERNS:
            match = pattern.search(content)
            if match:
                artifact_count += 1
                if len(artifact_examples) < 5:
                    context = content[max(0, match.start()-20):match.end()+20]
                    artifact_examples.append((i, context))
                break
    if artifact_count == 0:
        print("PASS: No HTML/JSX artifacts detected")
    else:
        print(f"WARN: {artifact_count} records contain potential HTML/JSX artifacts")
        for idx, ctx in artifact_examples:
            print(f"  Record {idx}: ...{ctx}...")

    # --- Check 8: Content quality metrics ---
    word_counts = [len(r.get("content", "").split()) for r in records]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    short_count = sum(1 for wc in word_counts if wc < MIN_CONTENT_WORDS)

    print(f"\nContent Quality Metrics:")
    print(f"  Average words per chunk: {avg_words:.0f}")
    print(f"  Min words: {min_words}")
    print(f"  Max words: {max_words}")
    print(f"  Chunks < {MIN_CONTENT_WORDS} words: {short_count}")

    if short_count > 0:
        print(f"WARN: {short_count} chunks have fewer than {MIN_CONTENT_WORDS} words")

    # --- Source distribution ---
    source_counter = Counter(r.get("source_name", "") for r in records)
    print(f"\nSource Distribution:")
    for src, count in source_counter.most_common():
        print(f"  {src}: {count}")

    print(f"\nCategory Distribution:")
    for cat, count in cat_counter.most_common():
        print(f"  {cat}: {count}")

    # --- Final verdict ---
    print(f"\n{'='*60}")
    if all_passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print(f"{'='*60}")

    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
