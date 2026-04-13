#!/usr/bin/env python3
"""
Validate the extracted site corpus JSONL file.

Checks:
  - Expected page count (~62 pages)
  - No empty content fields
  - No JSX/HTML artifacts in content
  - Business details accuracy (phone, email, address)
  - Valid category values
  - Content quality metrics

Usage:
    python scripts/validate_corpus.py                          # default path
    python scripts/validate_corpus.py data/site-corpus.jsonl   # explicit path
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "site-corpus.jsonl"

EXPECTED_MIN_PAGES = 56
EXPECTED_MAX_PAGES = 65
VALID_CATEGORIES = {"service", "city", "info"}

# JSX / HTML artifact patterns that should NOT appear in clean text
ARTIFACT_PATTERNS = [
    (r"<div[\s>]", "<div>"),
    (r"<span[\s>]", "<span>"),
    (r"<p[\s>]", "<p>"),
    (r"<img[\s>]", "<img>"),
    (r"<br\s*/?>", "<br>"),
    (r'className\s*=', "className="),
    (r'\{props\.', "{props."),
    (r'<[A-Z]\w+[\s/>]', "React component tag"),
    (r'dangerouslySetInnerHTML', "dangerouslySetInnerHTML"),
    (r'__html\s*:', "__html:"),
]

# Business details that must appear somewhere in the corpus
BUSINESS_DETAILS = {
    "phone": "(303) 993-8717",
    "email": "info@denversprinklerservices.com",
    "address_street": "3971 S Decatur St Unit A",
    "address_city": "Englewood, CO 80110",
    "company_name": "Denver Sprinkler and Landscape",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> tuple[list[dict], int]:
    """Load JSONL corpus file. Returns (records, parse_error_count)."""
    records = []
    parse_errors = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  FAIL: Invalid JSON on line {line_num}: {e}")
                parse_errors += 1
    return records, parse_errors


def validate(corpus_path: Path) -> bool:
    """Run all validation checks. Returns True if all pass."""
    print(f"Validating: {corpus_path}\n")

    if not corpus_path.exists():
        print(f"  FAIL: Corpus file not found at {corpus_path}")
        return False

    records, parse_errors = load_corpus(corpus_path)
    all_passed = True
    total_checks = 0
    passed_checks = 0

    # --- Check 0: Valid JSON ---
    total_checks += 1
    if parse_errors == 0:
        print("  PASS: All lines are valid JSON")
        passed_checks += 1
    else:
        print(f"  FAIL: {parse_errors} invalid JSON line(s)")
        all_passed = False

    # --- Check 1: Page count ---
    total_checks += 1
    n = len(records)
    if EXPECTED_MIN_PAGES <= n <= EXPECTED_MAX_PAGES:
        print(f"  PASS: Page count = {n} (expected {EXPECTED_MIN_PAGES}-{EXPECTED_MAX_PAGES})")
        passed_checks += 1
    else:
        print(f"  FAIL: Page count = {n} (expected {EXPECTED_MIN_PAGES}-{EXPECTED_MAX_PAGES})")
        all_passed = False

    # --- Check 2: Required fields ---
    total_checks += 1
    missing_fields = []
    non_dict_count = sum(1 for rec in records if not isinstance(rec, dict))
    if non_dict_count:
        print(f"  FAIL: {non_dict_count} record(s) are not JSON objects")
        all_passed = False
        print(f"\nResults: {passed_checks}/{total_checks} checks passed\nVALIDATION FAILED (non-object records — cannot continue)")
        return False
    for rec in records:
        for field in ("page", "title", "content", "category"):
            if field not in rec:
                missing_fields.append((rec.get("page", "?"), field))
    if not missing_fields:
        print("  PASS: All records have required fields (page, title, content, category)")
        passed_checks += 1
    else:
        print(f"  FAIL: Missing fields in {len(missing_fields)} records:")
        for page, field in missing_fields[:5]:
            print(f"         {page}: missing '{field}'")
        all_passed = False
        print(f"\nResults: {passed_checks}/{total_checks} checks passed\nVALIDATION FAILED (schema errors — cannot continue)")
        return False

    # --- Check 3: No empty content ---
    total_checks += 1
    empty = [r["page"] for r in records if not r.get("content", "").strip()]
    if not empty:
        print("  PASS: No empty content fields")
        passed_checks += 1
    else:
        print(f"  FAIL: {len(empty)} pages with empty content:")
        for p in empty:
            print(f"         {p}")
        all_passed = False

    # --- Check 4: Valid categories ---
    total_checks += 1
    invalid_cats = [(r["page"], r["category"]) for r in records
                    if r.get("category") not in VALID_CATEGORIES]
    if not invalid_cats:
        cats = {}
        for r in records:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(f"  PASS: All categories valid ({cats})")
        passed_checks += 1
    else:
        print("  FAIL: Invalid categories:")
        for page, cat in invalid_cats:
            print(f"         {page}: '{cat}'")
        all_passed = False

    # --- Check 5: No JSX/HTML artifacts ---
    total_checks += 1
    artifact_hits = []
    for rec in records:
        content = rec.get("content", "")
        for pattern, label in ARTIFACT_PATTERNS:
            if re.search(pattern, content):
                artifact_hits.append((rec["page"], label))
                break  # one per page is enough
    if not artifact_hits:
        print("  PASS: No JSX/HTML artifacts detected")
        passed_checks += 1
    else:
        print(f"  FAIL: JSX/HTML artifacts found in {len(artifact_hits)} pages:")
        for page, label in artifact_hits[:10]:
            print(f"         {page}: found '{label}'")
        all_passed = False

    # --- Check 6: Business details present in corpus ---
    total_checks += 1
    all_content = " ".join(r.get("content", "") for r in records)
    missing_biz = []
    for label, value in BUSINESS_DETAILS.items():
        if value.lower() not in all_content.lower():
            missing_biz.append((label, value))
    if not missing_biz:
        print("  PASS: All business details found in corpus")
        passed_checks += 1
    else:
        print("  FAIL: Missing business details:")
        for label, value in missing_biz:
            print(f"         {label}: '{value}' not found")
        all_passed = False

    # --- Check 7: No duplicate pages ---
    total_checks += 1
    page_set = set()
    dupes = set()
    for r in records:
        p = r["page"]
        if p in page_set:
            dupes.add(p)
        page_set.add(p)
    if not dupes:
        print("  PASS: No duplicate page paths")
        passed_checks += 1
    else:
        print(f"  FAIL: Duplicate page paths: {dupes}")
        all_passed = False

    # --- Check 8: No nav menu text leaks ---
    total_checks += 1
    nav_pattern = re.compile(r"HOME\s+SERVICES\s+LANDSCAPING", re.IGNORECASE)
    nav_leaks = [r["page"] for r in records if nav_pattern.search(r.get("content", ""))]
    if not nav_leaks:
        print("  PASS: No navigation menu text detected in content")
        passed_checks += 1
    else:
        print(f"  FAIL: Navigation menu text found in {len(nav_leaks)} pages:")
        for p in nav_leaks[:5]:
            print(f"         {p}")
        all_passed = False

    # --- Check 9: No copyright footer text leaks ---
    total_checks += 1
    copyright_pattern = re.compile(
        r"Denver Sprinkler and Landscape Inc\s*\||[\u00a9]", re.IGNORECASE
    )
    copyright_leaks = [r["page"] for r in records if copyright_pattern.search(r.get("content", ""))]
    if not copyright_leaks:
        print("  PASS: No copyright footer text detected in content")
        passed_checks += 1
    else:
        print(f"  FAIL: Copyright footer text found in {len(copyright_leaks)} pages:")
        for p in copyright_leaks[:5]:
            print(f"         {p}")
        all_passed = False

    # --- Check 10: No raw footer address blocks ---
    total_checks += 1
    # Match footer address blocks that start directly with "Address" (not
    # legitimate content like the contact page's "Get In Touch Address...")
    footer_addr_pattern = re.compile(
        r"(?:^|\n)Address\s*3971 S Decatur",
        re.IGNORECASE | re.MULTILINE,
    )
    addr_leaks = [r["page"] for r in records if footer_addr_pattern.search(r.get("content", ""))]
    if not addr_leaks:
        print("  PASS: No raw footer address blocks detected in content")
        passed_checks += 1
    else:
        print(f"  FAIL: Footer address blocks found in {len(addr_leaks)} pages:")
        for p in addr_leaks[:5]:
            print(f"         {p}")
        all_passed = False

    # --- Check 11: No shared CTA "Get In Touch With Us" block ---
    total_checks += 1
    cta_pattern = re.compile(
        r"Get In Touch With Us.*3971 S Decatur",
        re.IGNORECASE | re.DOTALL,
    )
    cta_leaks = [r["page"] for r in records if cta_pattern.search(r.get("content", ""))]
    if not cta_leaks:
        print("  PASS: No shared 'Get In Touch With Us' CTA block detected in content")
        passed_checks += 1
    else:
        print(f"  FAIL: Shared CTA block found in {len(cta_leaks)} pages:")
        for p in cta_leaks[:10]:
            print(f"         {p}")
        all_passed = False

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Results: {passed_checks}/{total_checks} checks passed")

    # Content quality metrics
    if records:
        lengths = [len(r["content"]) for r in records]
        print("\nContent metrics:")
        print(f"  Total pages: {len(records)}")
        print(f"  Average content length: {sum(lengths)/len(lengths):.0f} chars")
        print(f"  Min content length: {min(lengths)} chars ({[r['page'] for r in records if len(r['content'])==min(lengths)][0]})")
        print(f"  Max content length: {max(lengths)} chars")
        short = [r["page"] for r in records if len(r["content"]) < 50]
        if short:
            print(f"  WARNING: {len(short)} pages with very short content (<50 chars):")
            for p in short:
                print(f"           {p}")

    if all_passed:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION FAILED")

    return all_passed


def main() -> None:
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    success = validate(corpus_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
