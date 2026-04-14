#!/usr/bin/env python3
"""
Generate synthetic Q&A training pairs from site corpus and RAG sources
using the Claude API (Haiku 3.5).

Reads:
  - data/site-corpus.jsonl   (58 pages)
  - data/rag-sources.jsonl   (906 chunks)

Writes:
  - data/training-pairs.jsonl
  - data/.qa-progress.json   (checkpoint, removed on completion)
"""

import json
import os
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SITE_CORPUS = DATA_DIR / "site-corpus.jsonl"
RAG_SOURCES = DATA_DIR / "rag-sources.jsonl"
OUTPUT_FILE = DATA_DIR / "training-pairs.jsonl"
PARTIAL_FILE = DATA_DIR / ".training-pairs-partial.jsonl"
PROGRESS_FILE = DATA_DIR / ".qa-progress.json"

MODEL = "claude-3-5-haiku-latest"
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 5
DEDUP_THRESHOLD = 0.85
RAG_BATCH_SIZE = 5

# ---------------------------------------------------------------------------
# Business details (canonical -- must be exact)
# ---------------------------------------------------------------------------

BUSINESS_DETAILS = """
- Name: Denver Sprinkler and Landscape
- Phone: (303) 993-8717
- Email: info@denversprinklerservices.com
- Address: 3971 S Decatur St Unit A, Englewood, CO 80110
- Hours: Mon-Fri 7am-5pm, Sat 8am-2pm, Sun Closed
- Emergency: Available 24/7
""".strip()

# ---------------------------------------------------------------------------
# System prompt (will be cached via prompt caching)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a training data generator for Denver Sprinkler and Landscape's customer chat assistant.

Your job is to create realistic customer question/answer pairs from the provided source content. These pairs will be used to fine-tune a language model that powers a customer chat widget on the company website.

BUSINESS DETAILS (use these exact values whenever relevant):
{BUSINESS_DETAILS}

OUTPUT FORMAT:
Return a JSON array of objects, each with exactly two keys:
- "instruction": A natural customer question someone might ask via the chat widget.
- "response": A helpful, conversational answer grounded entirely in the provided source text.

RULES:
1. Questions must sound like real customer inquiries with varied phrasing. Use diverse question starters: "What...", "How much...", "Can you...", "I need...", "Do you offer...", "Tell me about...", "Is it possible...", "Where...", "When...", etc. Do NOT start every question with the same phrase.
2. Answers must be grounded ENTIRELY in the provided source text. Do NOT invent services, prices, capabilities, or claims not present in the source.
3. Where relevant, naturally include business details (phone number for contact questions, address for location questions, hours for scheduling questions, emergency line for urgent matters).
4. Answers should be conversational, helpful, and professional -- 2-5 sentences long.
5. Do NOT include any text outside the JSON array. Return ONLY the JSON array.
6. Each Q&A pair should cover a distinct topic or angle from the source material.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_progress(progress: dict) -> None:
    """Save checkpoint to disk."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def load_progress() -> dict:
    """Load checkpoint from disk if it exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"site_done": [], "rag_done": [], "total_input_tokens": 0,
            "total_output_tokens": 0, "cache_read_tokens": 0,
            "cache_creation_tokens": 0}


def append_pairs(pairs: list[dict], path: Path) -> None:
    """Append Q&A pairs to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def call_claude(client: anthropic.Anthropic, user_prompt: str,
                progress: dict) -> list[dict]:
    """Call Claude API with retry logic and return parsed Q&A pairs."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Track token usage
            usage = response.usage
            progress["total_input_tokens"] += usage.input_tokens
            progress["total_output_tokens"] += usage.output_tokens
            if hasattr(usage, "cache_read_input_tokens"):
                progress["cache_read_tokens"] += (
                    usage.cache_read_input_tokens or 0
                )
            if hasattr(usage, "cache_creation_input_tokens"):
                progress["cache_creation_tokens"] += (
                    usage.cache_creation_input_tokens or 0
                )

            # Parse JSON from response
            text = response.content[0].text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            pairs = json.loads(text)
            if not isinstance(pairs, list):
                print(f"  WARNING: Response is not a list, wrapping")
                pairs = [pairs]
            return pairs

        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIStatusError) as e:
            status = getattr(e, "status_code", 0)
            if status in (429, 529, 500) or isinstance(
                e, (anthropic.RateLimitError, anthropic.InternalServerError)
            ):
                wait = min(2 ** attempt * 2, 60)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} "
                      f"(status {status}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except json.JSONDecodeError as e:
            print(f"  WARNING: JSON parse error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"  FAILED: Could not parse response after retries")
                return []

    print(f"  FAILED: Max retries exceeded")
    return []


def normalize_text(text: str) -> str:
    """Normalize text for dedup comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate_pairs(pairs: list[dict]) -> tuple[list[dict], int]:
    """
    Remove near-duplicate instructions using SequenceMatcher.
    Uses category-blocked comparison to keep O(n^2) tractable.
    Returns (deduplicated pairs, number removed).
    """
    # Group by a rough category key
    groups: dict[str, list[int]] = {}
    for i, pair in enumerate(pairs):
        cat = pair.get("source", "unknown")
        ref = pair.get("source_ref", "")
        # For site pages, group by category extracted from ref
        # For RAG, group by source category
        key = cat
        if cat == "site":
            # Extract rough grouping from URL path
            parts = ref.strip("/").split("/")
            if parts:
                key = f"site-{parts[0]}" if parts[0] else "site-home"
        groups.setdefault(key, []).append(i)

    to_remove: set[int] = set()
    normalized_cache: dict[int, str] = {}

    def get_normalized(idx: int) -> str:
        if idx not in normalized_cache:
            normalized_cache[idx] = normalize_text(
                pairs[idx]["instruction"]
            )
        return normalized_cache[idx]

    for group_key, indices in groups.items():
        for i in range(len(indices)):
            if indices[i] in to_remove:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] in to_remove:
                    continue
                norm_a = get_normalized(indices[i])
                norm_b = get_normalized(indices[j])
                ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
                if ratio > DEDUP_THRESHOLD:
                    # Keep the one with the longer response
                    resp_a = len(pairs[indices[i]].get("response", ""))
                    resp_b = len(pairs[indices[j]].get("response", ""))
                    if resp_a >= resp_b:
                        to_remove.add(indices[j])
                    else:
                        to_remove.add(indices[i])

    # Also do cross-group dedup for very high similarity (> 0.95)
    all_indices = [i for i in range(len(pairs)) if i not in to_remove]
    # Only feasible for reasonable sizes
    if len(all_indices) < 5000:
        for i_idx in range(len(all_indices)):
            if all_indices[i_idx] in to_remove:
                continue
            for j_idx in range(i_idx + 1, len(all_indices)):
                if all_indices[j_idx] in to_remove:
                    continue
                norm_a = get_normalized(all_indices[i_idx])
                norm_b = get_normalized(all_indices[j_idx])
                # Quick length check to skip obviously different pairs
                if abs(len(norm_a) - len(norm_b)) > max(len(norm_a),
                                                         len(norm_b)) * 0.3:
                    continue
                ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
                if ratio > 0.95:
                    resp_a = len(
                        pairs[all_indices[i_idx]].get("response", ""))
                    resp_b = len(
                        pairs[all_indices[j_idx]].get("response", ""))
                    if resp_a >= resp_b:
                        to_remove.add(all_indices[j_idx])
                    else:
                        to_remove.add(all_indices[i_idx])

    deduped = [p for i, p in enumerate(pairs) if i not in to_remove]
    return deduped, len(to_remove)


def validate_pairs(pairs: list[dict]) -> None:
    """Run validation checks and print results."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    # Sample 50 random pairs
    sample_size = min(50, len(pairs))
    sample = random.sample(pairs, sample_size)

    print(f"\n--- Spot-check: {sample_size} random Q&A pairs ---\n")
    for i, pair in enumerate(sample, 1):
        print(f"[{i}] SOURCE: {pair.get('source', '?')} | "
              f"REF: {pair.get('source_ref', '?')}")
        print(f"  Q: {pair['instruction']}")
        print(f"  A: {pair['response'][:200]}"
              f"{'...' if len(pair['response']) > 200 else ''}")
        print()

    # Business detail accuracy checks
    print("--- Business Detail Accuracy ---")
    canonical = {
        "phone": "(303) 993-8717",
        "email": "info@denversprinklerservices.com",
        "address_partial": "3971 S Decatur St",
    }
    # Patterns that indicate wrong business details
    phone_pattern = re.compile(r"\(\d{3}\)\s*\d{3}[- ]\d{4}")
    email_pattern = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

    issues = []
    for i, pair in enumerate(pairs):
        resp = pair["response"]
        # Check phone numbers
        phones = phone_pattern.findall(resp)
        for p in phones:
            if p != canonical["phone"]:
                issues.append(
                    f"  Pair {i}: wrong phone '{p}' in response")
        # Check emails
        emails = email_pattern.findall(resp)
        for e in emails:
            if ("denver" in e.lower() or "sprinkler" in e.lower()) and \
               e != canonical["email"]:
                issues.append(
                    f"  Pair {i}: wrong email '{e}' in response")

    if issues:
        print(f"  ISSUES FOUND ({len(issues)}):")
        for issue in issues[:20]:
            print(issue)
    else:
        print("  All business details verified correct.")

    # Response length check
    print("\n--- Response Length ---")
    short = [i for i, p in enumerate(pairs) if len(p["response"]) < 20]
    long_ = [i for i, p in enumerate(pairs) if len(p["response"]) > 500]
    avg_len = sum(len(p["response"]) for p in pairs) / len(pairs)
    print(f"  Average response length: {avg_len:.0f} chars")
    print(f"  Too short (<20 chars): {len(short)} pairs")
    print(f"  Too long (>500 chars): {len(long_)} pairs")

    # Instruction diversity check
    print("\n--- Instruction Diversity ---")
    starters: dict[str, int] = {}
    for pair in sample:
        words = pair["instruction"].split()[:3]
        starter = " ".join(words).lower()
        starters[starter] = starters.get(starter, 0) + 1
    max_starter = max(starters.values()) if starters else 0
    max_pct = max_starter / sample_size * 100
    print(f"  Most common 3-word starter appears {max_starter}/{sample_size}"
          f" times ({max_pct:.0f}%)")
    if max_pct > 30:
        most_common = max(starters, key=starters.get)
        print(f"  WARNING: Low diversity. Most common: '{most_common}'")
    else:
        print("  Diversity looks good (no starter > 30%).")

    # Summary stats
    print("\n--- Summary ---")
    print(f"  Total pairs: {len(pairs)}")
    site_count = sum(1 for p in pairs if p["source"] == "site")
    rag_count = sum(1 for p in pairs if p["source"] == "rag")
    print(f"  Site pairs: {site_count}")
    print(f"  RAG pairs: {rag_count}")

    # Category breakdown
    categories: dict[str, int] = {}
    for pair in pairs:
        cat = pair.get("source_ref", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  Unique source refs: {len(categories)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Denver Sprinkler Q&A Pair Generator")
    print("=" * 70)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set. Cannot proceed.")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Load source data
    print(f"\nLoading source data...")
    site_pages = load_jsonl(SITE_CORPUS)
    rag_chunks = load_jsonl(RAG_SOURCES)
    print(f"  Site corpus: {len(site_pages)} pages")
    print(f"  RAG sources: {len(rag_chunks)} chunks")

    # Load or initialize progress
    progress = load_progress()
    if progress["site_done"] or progress["rag_done"]:
        print(f"\nResuming from checkpoint:")
        print(f"  Site pages done: {len(progress['site_done'])}")
        print(f"  RAG batches done: {len(progress['rag_done'])}")

    all_pairs: list[dict] = []

    # Load any existing partial results
    if PARTIAL_FILE.exists():
        all_pairs = load_jsonl(PARTIAL_FILE)
        print(f"  Loaded {len(all_pairs)} existing partial pairs")

    # -----------------------------------------------------------------------
    # Process site corpus
    # -----------------------------------------------------------------------
    print(f"\n--- Processing Site Corpus ({len(site_pages)} pages) ---")
    for idx, page in enumerate(site_pages):
        if idx in progress["site_done"]:
            continue

        page_url = page.get("page", "unknown")
        title = page.get("title", "Untitled")
        content = page.get("content", "")

        if not content.strip():
            print(f"  [{idx + 1}/{len(site_pages)}] Skipping empty: "
                  f"{page_url}")
            progress["site_done"].append(idx)
            save_progress(progress)
            continue

        # Determine pair count based on content length
        num_pairs = 5 if len(content) < 500 else 8 if len(content) > 2000 \
            else 6

        user_prompt = (
            f"Generate {num_pairs} Q&A pairs from this website page.\n\n"
            f"Page URL: {page_url}\n"
            f"Title: {title}\n"
            f"Category: {page.get('category', 'general')}\n\n"
            f"Content:\n{content}"
        )

        print(f"  [{idx + 1}/{len(site_pages)}] {page_url} "
              f"(requesting {num_pairs} pairs)...", end="", flush=True)

        pairs = call_claude(client, user_prompt, progress)

        # Add source metadata
        for pair in pairs:
            pair["source"] = "site"
            pair["source_ref"] = page_url

        all_pairs.extend(pairs)
        append_pairs(pairs, PARTIAL_FILE)
        progress["site_done"].append(idx)
        save_progress(progress)

        print(f" got {len(pairs)}")
        time.sleep(RATE_LIMIT_DELAY)

    # -----------------------------------------------------------------------
    # Process RAG sources (batched)
    # -----------------------------------------------------------------------
    rag_batches = [
        rag_chunks[i:i + RAG_BATCH_SIZE]
        for i in range(0, len(rag_chunks), RAG_BATCH_SIZE)
    ]
    print(f"\n--- Processing RAG Sources ({len(rag_batches)} batches "
          f"of {RAG_BATCH_SIZE}) ---")

    for batch_idx, batch in enumerate(rag_batches):
        if batch_idx in progress["rag_done"]:
            continue

        # Build batch content
        chunks_text = []
        source_refs = []
        for i, chunk in enumerate(batch, 1):
            url = chunk.get("source_url", "unknown")
            source_refs.append(url)
            chunks_text.append(
                f"--- Chunk {i} ---\n"
                f"Source: {chunk.get('source_name', 'Unknown')}\n"
                f"URL: {url}\n"
                f"Title: {chunk.get('title', 'Untitled')}\n"
                f"Category: {chunk.get('category', 'general')}\n"
                f"Content:\n{chunk.get('content', '')}\n"
            )

        user_prompt = (
            f"Generate 1-2 Q&A pairs per chunk from these "
            f"{len(batch)} source chunks. Return a flat JSON array "
            f"with all pairs combined.\n\n"
            + "\n".join(chunks_text)
        )

        print(f"  [Batch {batch_idx + 1}/{len(rag_batches)}] "
              f"({len(batch)} chunks)...", end="", flush=True)

        pairs = call_claude(client, user_prompt, progress)

        # Add source metadata
        # Try to match pairs to their source chunks based on content
        primary_ref = source_refs[0] if source_refs else "unknown"
        for pair in pairs:
            pair["source"] = "rag"
            # Use the primary ref for the batch; individual matching
            # is unreliable without explicit chunk IDs in the response
            pair["source_ref"] = primary_ref

        # Try to distribute source_refs more accurately
        if len(pairs) <= len(batch) * 2:
            # Rough assignment: distribute pairs across chunks
            for pi, pair in enumerate(pairs):
                chunk_idx = min(pi // 2, len(source_refs) - 1)
                pair["source_ref"] = source_refs[chunk_idx]

        all_pairs.extend(pairs)
        append_pairs(pairs, PARTIAL_FILE)
        progress["rag_done"].append(batch_idx)
        save_progress(progress)

        print(f" got {len(pairs)}")
        time.sleep(RATE_LIMIT_DELAY)

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------
    print(f"\n--- Deduplication ---")
    print(f"  Pairs before dedup: {len(all_pairs)}")
    all_pairs, removed = deduplicate_pairs(all_pairs)
    print(f"  Pairs removed: {removed}")
    print(f"  Pairs after dedup: {len(all_pairs)}")

    # -----------------------------------------------------------------------
    # Write final output
    # -----------------------------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_pairs)} pairs to {OUTPUT_FILE}")

    # Clean up checkpoint and partial files
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    if PARTIAL_FILE.exists():
        PARTIAL_FILE.unlink()

    # -----------------------------------------------------------------------
    # Cost estimate
    # -----------------------------------------------------------------------
    print(f"\n--- Cost Estimate ---")
    input_tokens = progress["total_input_tokens"]
    output_tokens = progress["total_output_tokens"]
    cache_read = progress.get("cache_read_tokens", 0)
    cache_create = progress.get("cache_creation_tokens", 0)

    # Haiku 3.5 pricing (per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * 0.80
    output_cost = (output_tokens / 1_000_000) * 4.00
    cache_read_cost = (cache_read / 1_000_000) * 0.08
    cache_create_cost = (cache_create / 1_000_000) * 1.00
    total_cost = input_cost + output_cost + cache_read_cost + cache_create_cost

    print(f"  Input tokens: {input_tokens:,}")
    print(f"  Output tokens: {output_tokens:,}")
    print(f"  Cache read tokens: {cache_read:,}")
    print(f"  Cache creation tokens: {cache_create:,}")
    print(f"  Estimated cost: ${total_cost:.4f}")
    print(f"    Input:        ${input_cost:.4f}")
    print(f"    Output:       ${output_cost:.4f}")
    print(f"    Cache read:   ${cache_read_cost:.4f}")
    print(f"    Cache create: ${cache_create_cost:.4f}")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    validate_pairs(all_pairs)

    print("\nDone!")
    return all_pairs


if __name__ == "__main__":
    main()
