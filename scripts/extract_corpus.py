#!/usr/bin/env python3
"""
Extract site content from crawl-raw.json into a structured JSONL training corpus.

Fetches docs/crawl-raw.json from peakweb-team/denver-sprinkler via GitHub API,
or accepts a local file path as a fallback. Filters, deduplicates, removes shared
navigation/footer content, and outputs structured JSONL for LLM fine-tuning.

Usage:
    python scripts/extract_corpus.py                      # fetch via gh api
    python scripts/extract_corpus.py /path/to/crawl.json  # use local file
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO = "peakweb-team/denver-sprinkler"
CRAWL_PATH = "docs/crawl-raw.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "site-corpus.jsonl"

# Shared-content threshold: sections appearing on more than this fraction of
# pages are considered shared (nav, header, footer, copyright).
SHARED_THRESHOLD = 0.50

# City names that appear in city landing page URLs
CITY_TOKENS = {
    "littleton", "lakewood", "arvada", "thornton", "aurora", "englewood",
}

# Blog / article path pattern: /category/slug/  (two-level deep paths under
# known content categories)
BLOG_CATEGORIES = {
    "uncategorized", "landscaping", "concrete", "wooden-fencing",
    "brick-pavers", "retaining-walls", "sprinkler-repair",
}

# Blocklist patterns: sections matching any of these are considered shared
# chrome (nav, footer, copyright) and should be excluded regardless of
# frequency threshold.
BLOCKLIST_PATTERNS = [
    # Nav menu text (contains the full site navigation)
    re.compile(r"HOME\s+SERVICES\s+LANDSCAPING", re.IGNORECASE),
    # Copyright footer
    re.compile(r"Denver Sprinkler and Landscape Inc\s*\|", re.IGNORECASE),
    re.compile(r"\u00a9"),  # copyright symbol ©
    # Footer address block — starts directly with "Address" (not "Get In Touch")
    # The contact page's "Get In Touch Address..." section is legitimate content
    re.compile(r"^Address\s*3971 S Decatur", re.IGNORECASE),
    # Shared CTA block: "Get In Touch With Us ... 3971 S Decatur"
    # Note: /contact-us/ uses "Get In Touch Today!" / "Get In Touch Address..."
    # which will NOT match this pattern (requires "With Us" + the address).
    re.compile(r"Get In Touch With Us.*3971 S Decatur", re.IGNORECASE | re.DOTALL),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_crawl_json(local_path: str | None = None) -> dict:
    """Return parsed crawl JSON, either from a local file or via gh api."""
    if local_path:
        print(f"Loading crawl data from local file: {local_path}")
        with open(local_path) as f:
            return json.load(f)

    print(f"Fetching crawl data from {REPO}/{CRAWL_PATH} via gh api ...")
    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{REPO}/contents/{CRAWL_PATH}",
                "-H", "Accept: application/vnd.github.raw+json",
            ],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "The `gh` CLI is not installed. Install/authenticate `gh`, or pass a local crawl.json path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or str(exc)
        raise SystemExit(
            f"Failed to fetch {REPO}/{CRAWL_PATH} via `gh api`: {stderr}\n"
            "Run `gh auth status` or pass a local crawl.json path."
        ) from exc
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Filtering & deduplication
# ---------------------------------------------------------------------------

def filter_pages(pages: list[dict]) -> list[dict]:
    """Remove PDFs (status 0) and deduplicate trailing-slash variants."""
    # Keep only HTTP 200 pages
    html_pages = [p for p in pages if p.get("status") == 200]

    # Deduplicate: prefer the variant WITH trailing slash
    seen: dict[str, dict] = {}
    for page in html_pages:
        canon = page["canonicalPath"].rstrip("/") or "/"
        existing = seen.get(canon)
        if existing is None:
            seen[canon] = page
        else:
            # Prefer the trailing-slash version
            if page["canonicalPath"].endswith("/"):
                seen[canon] = page

    return list(seen.values())


# ---------------------------------------------------------------------------
# Shared content detection
# ---------------------------------------------------------------------------

def build_shared_set(pages: list[dict], threshold_frac: float) -> set[str]:
    """Identify section texts that appear on more than threshold_frac of pages."""
    text_counter: Counter[str] = Counter()
    for page in pages:
        # Count each unique text only once per page
        page_texts = set()
        for section in page.get("sections", []):
            text = section.get("text", "").strip()
            if text:
                # Use first 200 chars as the dedup key to handle minor
                # trailing variations in the same block
                key = text[:200]
                page_texts.add(key)
        for key in page_texts:
            text_counter[key] += 1

    threshold = len(pages) * threshold_frac
    shared = {key for key, count in text_counter.items() if count > threshold}
    return shared


# ---------------------------------------------------------------------------
# Page categorization
# ---------------------------------------------------------------------------

def categorize_page(path: str) -> str:
    """Assign a category based on URL pattern: service, city, or info."""
    # Normalize
    clean = path.strip("/")

    # Blog / article posts: /category/slug/ (two segments, first is a blog cat)
    parts = clean.split("/")
    if len(parts) >= 2 and parts[0] in BLOG_CATEGORIES:
        return "info"

    # Category listing pages (e.g., /concrete/, /landscaping/) that are just
    # parent paths for blog posts -- treat as info
    if len(parts) == 1 and parts[0] in BLOG_CATEGORIES:
        # But some of these are actual service pages. Check if the slug
        # matches a known blog category that doubles as a service.
        # /retaining-walls/, /concrete/, /sprinkler-repair/, /landscaping/
        # are blog category listing pages, not primary service pages.
        # The primary service pages have "-denver" or are top-level services.
        # Actually, these category index pages behave like info/blog pages.
        return "info"

    # Info pages: about, testimonials, contact, join, home
    info_slugs = {"", "about-us", "testimonials", "contact-us", "join-our-team"}
    if clean in info_slugs:
        return "info"

    # City landing pages: any URL containing a city token
    lower = clean.lower()
    for city in CITY_TOKENS:
        if city in lower:
            return "city"

    # Everything else is a service page
    return "service"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def is_blocklisted(text: str) -> bool:
    """Return True if text matches any blocklist pattern (nav/footer/copyright)."""
    for pattern in BLOCKLIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


def extract_page_content(page: dict, shared_keys: set[str]) -> str:
    """Extract unique page content, filtering out shared sections."""
    parts: list[str] = []
    seen_texts: set[str] = set()  # for intra-page deduplication

    # Include h1 as the top heading
    h1_list = page.get("h1", [])
    if h1_list:
        parts.append(f"# {h1_list[0]}")

    for section in page.get("sections", []):
        text = section.get("text", "").strip()
        if not text:
            continue
        key = text[:200]
        if key in shared_keys:
            continue
        # Skip nav elements (they contain menu items)
        if section.get("tag") == "nav":
            continue
        # Skip sections matching blocklist patterns
        if is_blocklisted(text):
            continue
        # Intra-page deduplication: skip if we already saw this text block
        if key in seen_texts:
            continue
        seen_texts.add(key)
        parts.append(text)

    return "\n\n".join(parts)


def clean_title(raw_title: str) -> str:
    """Clean up the page title, removing site name suffix."""
    # Common patterns: "Page Title - Site Name" or "Page Title | Site Name"
    for sep in [" | ", " – ", " - "]:
        if sep in raw_title:
            raw_title = raw_title.split(sep)[0].strip()
            break
    return raw_title


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    local_path = sys.argv[1] if len(sys.argv) > 1 else None
    crawl_data = fetch_crawl_json(local_path)

    all_pages = crawl_data["pages"]
    print(f"Total pages in crawl: {len(all_pages)}")

    pages = filter_pages(all_pages)
    print(f"After filtering (HTML 200 + dedup): {len(pages)}")

    shared_keys = build_shared_set(pages, SHARED_THRESHOLD)
    print(f"Shared sections identified (>{SHARED_THRESHOLD*100:.0f}% threshold): {len(shared_keys)}")

    records = []
    skipped_empty = 0
    for page in pages:
        path = page["canonicalPath"]
        title = clean_title(page.get("title", "") or "")
        content = extract_page_content(page, shared_keys)
        category = categorize_page(path)

        # Skip pages that have no content after shared/blocklist filtering
        # (e.g., blog category index pages that only contain nav chrome)
        if not content.strip():
            skipped_empty += 1
            print(f"  Skipping {path} (no content after filtering)")
            continue

        records.append({
            "page": path,
            "title": title,
            "content": content,
            "category": category,
        })

    # Sort by path for deterministic output
    records.sort(key=lambda r: r["page"])

    # Write JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    cats = Counter(r["category"] for r in records)
    content_lengths = [len(r["content"]) for r in records]
    avg_len = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    empty = sum(1 for r in records if not r["content"].strip())

    print(f"\nExtraction complete!")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Pages: {len(records)}")
    print(f"  Categories: {cats['service']} service, {cats['city']} city, {cats['info']} info")
    print(f"  Shared sections removed: {len(shared_keys)}")
    print(f"  Average content length: {avg_len:.0f} chars")
    if empty:
        print(f"  WARNING: {empty} pages with empty content")


if __name__ == "__main__":
    main()
