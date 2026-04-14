#!/usr/bin/env python3
"""
Crawl authoritative landscaping/irrigation sources into a RAG training corpus.

Fetches content from CSU Extension, EPA WaterSense, Rain Bird, and Denver Water.
Respects robots.txt, rate limits at 2 seconds per domain, and outputs structured
JSONL for fine-tuning augmentation.

Usage:
    python scripts/crawl_rag_sources.py
"""

import json
import re
import time
import defusedxml.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "rag-sources.jsonl"

USER_AGENT = (
    "DenverSprinklerBot/1.0 "
    "(training data collection; contact: info@denversprinklerservices.com)"
)
RATE_LIMIT_SECONDS = 2.0
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
CHUNK_TARGET_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

# Post-crawl content filter: reject pages whose content is primarily agricultural/
# non-residential even when the URL matched a positive keyword (e.g. "compost"
# or "irrigation" in a farming context).  A page is rejected when it contains
# >= CONTENT_NEGATIVE_THRESHOLD of these terms.
CONTENT_NEGATIVE_TERMS = [
    "livestock", "cattle", "rangeland", "pasture", "grazing",
    "cropland", "commodity", "agricultural producer",
    "milkweed", "knapweed",
]
CONTENT_NEGATIVE_THRESHOLD = 3  # >= 3 total occurrences => reject

# ---------------------------------------------------------------------------
# Category keyword maps
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "irrigation": [
        "irrigation", "irrigate", "sprinkler", "drip", "spray head",
        "rotor", "nozzle", "valve", "controller", "timer", "emitter",
        "backflow", "winterize", "blowout", "water pressure", "gpm",
        "gallons per minute", "zone", "head-to-head", "precipitation rate",
        "smart controller", "rain sensor", "flow sensor", "lateral line",
        "mainline", "pipe", "fitting", "swing joint",
    ],
    "landscaping": [
        "landscaping", "landscape", "lawn", "garden", "plant", "tree",
        "shrub", "flower", "turf", "grass", "sod", "seed", "mulch",
        "compost", "fertilizer", "pruning", "mowing", "aeration",
        "overseeding", "xeriscaping", "xeriscape", "native plant",
        "perennial", "annual", "groundcover", "shade", "ornamental",
        "planting", "soil amendment", "topsoil",
    ],
    "hardscaping": [
        "hardscaping", "hardscape", "retaining wall", "paver", "patio",
        "concrete", "flagstone", "brick", "stone", "gravel", "walkway",
        "pathway", "edging", "border", "wall block", "drainage",
        "french drain", "dry creek", "fire pit", "outdoor kitchen",
        "masonry", "stepping stone", "landscape block", "pavement",
        "cobble", "slate", "sandstone", "crushed rock", "decomposed granite",
        "landscape fabric", "weed barrier", "dry stack", "mortar",
        "landscape wall", "garden wall", "seat wall", "raised bed",
    ],
    "climate": [
        "drought", "climate", "colorado", "denver", "front range",
        "water restriction", "water conservation", "water-wise",
        "water wise", "semi-arid", "high altitude", "freeze", "frost",
        "winter", "snow", "seasonal", "temperature", "usda zone",
        "hardiness zone", "growing season", "evapotranspiration",
        "water budget", "watering schedule", "watering restriction",
    ],
}

# ---------------------------------------------------------------------------
# Source Registry
# ---------------------------------------------------------------------------
SOURCE_REGISTRY: list[dict] = [
    {
        "name": "Colorado State University Extension",
        "domain": "extension.colostate.edu",
        "strategy": "sitemap",
        "sitemap_urls": [
            "https://extension.colostate.edu/resource-cpt-sitemap1.xml",
            "https://extension.colostate.edu/resource-cpt-sitemap2.xml",
            "https://extension.colostate.edu/resource-cpt-sitemap3.xml",
            "https://extension.colostate.edu/resource-cpt-sitemap4.xml",
        ],
        "url_keywords": [
            "lawn", "grass", "turf", "irrigation", "sprinkler",
            "xeriscape", "xeriscaping", "landscape", "landscaping",
            "garden", "compost", "mulch", "tree", "shrub",
            "flower", "drought", "native", "perennial",
            "aerat", "mow", "prun", "weed",
            "groundcover", "annual", "bulb", "vine",
            "rose", "evergreen",
            "deciduous", "ornamental", "wildflower", "pollinator",
            "drip", "clay-soil", "alkaline",
            "sod", "overseed", "dethatch",
            "retaining", "paver", "patio", "flagstone",
            "hardscape", "concrete", "gravel", "edging",
            "walkway", "stone", "brick", "masonry",
        ],
        "url_negative_keywords": [
            "livestock", "cattle", "crop", "corn", "wheat", "dryland",
            "rangeland", "wolf", "wildlife", "rabbit", "deer-resistant",
            "4-h", "youth", "commodity", "insurance", "rabies",
            "decreed-water", "water-right", "water-law",
            "poultry", "chicken", "horse", "sheep", "goat", "swine",
            "pig", "forage", "hay", "alfalfa", "barley", "oat",
            "soybean", "sunflower", "sugar-beet", "potato",
            "beef", "dairy", "meat", "carcass", "slaughter",
            "agricultural-producer", "irrigated-pasture",
            "small-acreage-pasture", "grazing", "knapweed", "milkweed",
        ],
        "content_selectors": ["article", "main", ".entry-content", "#content"],
    },
    {
        "name": "EPA WaterSense",
        "domain": "www.epa.gov",
        "strategy": "seed",
        "seed_urls": [
            "https://www.epa.gov/watersense/landscaping-tips",
            "https://www.epa.gov/watersense/watering-tips",
            "https://www.epa.gov/watersense/what-plant",
            "https://www.epa.gov/watersense/irrigation-pro",
            "https://www.epa.gov/watersense/watersense-labeled-controllers",
            "https://www.epa.gov/watersense/sprinkler-spruce-up",
            "https://www.epa.gov/watersense/when-its-hot",
            "https://www.epa.gov/watersense/spray-sprinkler-bodies",
        ],
        "content_selectors": ["article", "main", ".main-content", "#main-content"],
    },
    {
        "name": "Rain Bird",
        "domain": "www.rainbird.com",
        "strategy": "sitemap",
        "sitemap_urls": [
            "https://www.rainbird.com/sitemap.xml",
        ],
        "url_patterns": [
            r"/homeowners/blog/",
            r"/blog/",
            r"/education/",
        ],
        "content_selectors": ["article", "main", ".node__content", ".field--name-body"],
    },
    {
        "name": "Denver Water",
        "domain": "www.denverwater.org",
        "strategy": "seed",
        "seed_urls": [
            # Conservation hub & watering rules
            "https://www.denverwater.org/residential/rebates-and-conservation-tips",
            "https://www.denverwater.org/residential/rebates-and-conservation-tips/summer-watering-rules",
            "https://www.denverwater.org/residential/rebates-and-conservation-tips/remodel-your-yard",
            # Efficiency tips — lawn, garden, trees, landscape
            "https://www.denverwater.org/residential/efficiency-tip/watering-your-lawn",
            "https://www.denverwater.org/residential/efficiency-tip/vegetable-and-flower-gardens",
            "https://www.denverwater.org/residential/efficiency-tip/tree-care-guide",
            "https://www.denverwater.org/residential/efficiency-tip/tips-using-compost",
            "https://www.denverwater.org/residential/efficiency-tip/have-healthy-lawn-less-water",
            "https://www.denverwater.org/residential/efficiency-tip/watering-weather-mind",
            "https://www.denverwater.org/residential/efficiency-tip/winter-watering-tips-trees-shrubs",
            "https://www.denverwater.org/residential/efficiency-tip/summer-watering-rules-are-here",
            "https://www.denverwater.org/residential/efficiency-tip/resources-water-smart-landscaping",
            "https://www.denverwater.org/residential/efficiency-tip/sprinkler-check-time",
            "https://www.denverwater.org/residential/efficiency-tip/learning-about-coloradoscaping",
            "https://www.denverwater.org/residential/efficiency-tip/beautiful-xeric-landscape",
            "https://www.denverwater.org/residential/efficiency-tip/discounts-turfgrass-removal",
            "https://www.denverwater.org/residential/efficiency-tip/fall-planting",
            "https://www.denverwater.org/residential/efficiency-tip/make-summer-switch-water-wise-look",
            "https://www.denverwater.org/residential/efficiency-tip/plan-ahead-now-change-your-lawn-and-landscape-next-spring-and-summer",
            "https://www.denverwater.org/residential/efficiency-tip/wait-dont-irrigate",
            "https://www.denverwater.org/residential/efficiency-tip/water-waste-adds",
            # Smart irrigation
            "https://www.denverwater.org/residential/conservation-tip/smart-irrigation-month",
        ],
        "content_selectors": ["article", "main", ".node__content", ".field--name-body"],
    },
    # Irrigation Association removed: JS-rendered pages yield too little extractable content.
    # Toro removed: residential irrigation pages return 404; only professional/commercial content exists.
]

# ---------------------------------------------------------------------------
# Robots.txt Cache
# ---------------------------------------------------------------------------
_robots_cache: dict[str, RobotFileParser] = {}


def check_robots(url: str, session: Optional[requests.Session] = None) -> bool:
    """Return True if our User-Agent is allowed to fetch this URL.

    Uses the rate limiter when fetching robots.txt to be polite.
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain not in _robots_cache:
        rp = RobotFileParser()
        robots_url = f"{parsed.scheme}://{domain}/robots.txt"
        # Respect rate limit for robots.txt fetch
        now = time.time()
        last = _last_fetch_time.get(domain, 0.0)
        wait = RATE_LIMIT_SECONDS - (now - last)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_fetch_time[domain] = time.time()
            fetch_session = session or requests.Session()
            resp = fetch_session.get(
                robots_url, headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                # No robots.txt or error — assume allowed
                rp.parse([])
        except requests.RequestException:
            rp.parse([])
        _robots_cache[domain] = rp
    return _robots_cache[domain].can_fetch(USER_AGENT, url)


# ---------------------------------------------------------------------------
# Polite Fetcher
# ---------------------------------------------------------------------------
_last_fetch_time: dict[str, float] = {}


def polite_fetch(url: str, session: requests.Session) -> Optional[str]:
    """Fetch a URL politely: respect rate limits, retry on errors."""
    domain = urlparse(url).netloc

    # Rate limiting per domain
    now = time.time()
    last = _last_fetch_time.get(domain, 0.0)
    wait = RATE_LIMIT_SECONDS - (now - last)
    if wait > 0:
        time.sleep(wait)

    for attempt in range(MAX_RETRIES):
        try:
            _last_fetch_time[domain] = time.time()
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff = (2 ** attempt) * 2
                print(f"  HTTP {resp.status_code} for {url}, retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            # 4xx other than 429 — don't retry
            print(f"  HTTP {resp.status_code} for {url}, skipping")
            return None
        except requests.RequestException as e:
            backoff = (2 ** attempt) * 2
            print(f"  Error fetching {url}: {e}, retrying in {backoff}s...")
            time.sleep(backoff)

    print(f"  Failed to fetch {url} after {MAX_RETRIES} retries")
    return None


# ---------------------------------------------------------------------------
# URL Discovery
# ---------------------------------------------------------------------------

def discover_sitemap_urls(
    sitemap_urls: list[str],
    session: requests.Session,
    url_keywords: Optional[list[str]] = None,
    url_patterns: Optional[list[str]] = None,
    url_negative_keywords: Optional[list[str]] = None,
) -> list[str]:
    """Parse sitemaps and filter URLs by keywords or patterns."""
    all_urls: list[str] = []

    for sitemap_url in sitemap_urls:
        html = polite_fetch(sitemap_url, session)
        if not html:
            continue

        # Handle sitemap index (contains other sitemaps)
        try:
            root = ET.fromstring(html)
        except ET.ParseError:
            print(f"  Failed to parse sitemap XML: {sitemap_url}")
            continue

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        # Check if this is a sitemap index
        sitemap_entries = root.findall(".//sm:sitemap/sm:loc", ns)
        if sitemap_entries:
            child_urls = [e.text.strip() for e in sitemap_entries if e.text]
            child_discovered = discover_sitemap_urls(
                child_urls, session, url_keywords, url_patterns, url_negative_keywords
            )
            all_urls.extend(child_discovered)
            continue

        # Regular sitemap — extract URLs
        loc_entries = root.findall(".//sm:url/sm:loc", ns)
        for loc in loc_entries:
            if loc.text:
                all_urls.append(loc.text.strip())

    # Filter by keywords (any keyword appears in the URL path)
    if url_keywords:
        filtered = []
        for url in all_urls:
            path_lower = urlparse(url).path.lower()
            if any(kw in path_lower for kw in url_keywords):
                filtered.append(url)
        all_urls = filtered

    # Filter OUT negative keywords (any negative keyword in URL path => exclude)
    if url_negative_keywords:
        filtered = []
        for url in all_urls:
            path_lower = urlparse(url).path.lower()
            if not any(nkw in path_lower for nkw in url_negative_keywords):
                filtered.append(url)
        all_urls = filtered

    # Filter by regex patterns (any pattern matches)
    if url_patterns:
        filtered = []
        for url in all_urls:
            if any(re.search(pat, url) for pat in url_patterns):
                filtered.append(url)
        all_urls = filtered

    return list(dict.fromkeys(all_urls))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Content Extraction
# ---------------------------------------------------------------------------

def extract_content(html: str, selectors: list[str]) -> tuple[str, str]:
    """Extract title and main text content from HTML.

    Returns (title, content_text).
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()
        # Remove common suffixes
        for sep in [" | ", " – ", " - ", " :: "]:
            if sep in title:
                title = title.split(sep)[0].strip()

    # Also try h1
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Remove unwanted elements
    for tag_name in ["nav", "header", "footer", "aside", "script", "style",
                     "noscript", "iframe", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements with common nav/footer classes/ids
    for selector in [
        ".nav", ".navbar", ".navigation", ".menu", ".sidebar",
        ".footer", ".header", ".breadcrumb", ".pagination",
        "#nav", "#navbar", "#navigation", "#menu", "#sidebar",
        "#footer", "#header", ".cookie-banner", ".social-share",
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Try content selectors in order
    content_el = None
    for sel in selectors:
        content_el = soup.select_one(sel)
        if content_el:
            break

    if not content_el:
        # Fallback: use body
        content_el = soup.find("body")

    if not content_el:
        return title, ""

    # Get text, preserving paragraph structure
    paragraphs = []
    for el in content_el.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "blockquote"]):
        text = el.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text and len(text) > 10:
            paragraphs.append(text)

    # If we got very little from structured elements, try raw text
    if len(paragraphs) < 3:
        raw_text = content_el.get_text(separator="\n", strip=True)
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.split("\n")]
        paragraphs = [line for line in lines if len(line) > 10]

    content = "\n\n".join(paragraphs)

    # Clean up any remaining HTML artifacts
    content = re.sub(r"<[^>]+>", "", content)
    content = re.sub(r"&[a-zA-Z]+;", " ", content)
    content = re.sub(r"&#\d+;", " ", content)
    content = re.sub(r"\s{3,}", "\n\n", content)

    return title, content.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """Split text into ~500-word chunks with ~50-word overlap.

    Preserves paragraph boundaries where possible.
    """
    if not text.strip():
        return []

    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []
    current_paragraphs: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > CHUNK_TARGET_WORDS and current_words:
            # Flush current chunk
            chunks.append("\n\n".join(current_paragraphs))

            # Create overlap from the tail of the current chunk
            overlap_words = current_words[-CHUNK_OVERLAP_WORDS:]
            overlap_text = " ".join(overlap_words)
            current_paragraphs = [overlap_text]
            current_words = list(overlap_words)

        current_paragraphs.append(para)
        current_words.extend(para.split())

    # Flush last chunk
    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    # Filter out tiny chunks (less than 30 words)
    chunks = [c for c in chunks if len(c.split()) >= 30]

    return chunks


# ---------------------------------------------------------------------------
# Category Classification
# ---------------------------------------------------------------------------

def classify_category(text: str) -> str:
    """Classify text into a category based on keyword frequency.

    Uses occurrence count (not just unique presence) so that a text
    mentioning "patio" 5 times scores higher for hardscaping than one
    that mentions it once.  Hardscaping keywords get a 3x boost because
    the category is narrow and easily drowned out by broader terms.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}
    # Boost factor for underrepresented categories
    category_boost = {"hardscaping": 3.0}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        boost = category_boost.get(cat, 1.0)
        score = 0.0
        for kw in keywords:
            score += text_lower.count(kw) * boost
        scores[cat] = score

    if not any(scores.values()):
        return "irrigation"  # default

    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def crawl_source(source: dict, session: requests.Session) -> list[dict]:
    """Crawl a single source and return document chunk records."""
    name = source["name"]
    print(f"\n{'='*60}")
    print(f"Crawling: {name}")
    print(f"{'='*60}")

    # Discover URLs
    if source["strategy"] == "sitemap":
        urls = discover_sitemap_urls(
            source["sitemap_urls"],
            session,
            url_keywords=source.get("url_keywords"),
            url_patterns=source.get("url_patterns"),
            url_negative_keywords=source.get("url_negative_keywords"),
        )
        print(f"  Discovered {len(urls)} URLs from sitemaps")
    else:
        urls = list(source["seed_urls"])
        print(f"  Using {len(urls)} seed URLs")

    records: list[dict] = []
    pages_fetched = 0
    pages_skipped = 0

    for url in urls:
        # Check robots.txt
        if not check_robots(url, session):
            print(f"  Blocked by robots.txt: {url}")
            pages_skipped += 1
            continue

        html = polite_fetch(url, session)
        if not html:
            pages_skipped += 1
            continue

        pages_fetched += 1
        title, content = extract_content(html, source["content_selectors"])

        if not content or len(content.split()) < 30:
            print(f"  Skipping (too little content): {url}")
            continue

        # Post-crawl content relevance filter
        content_lower = content.lower()
        neg_hits = sum(content_lower.count(t) for t in CONTENT_NEGATIVE_TERMS)
        if neg_hits >= CONTENT_NEGATIVE_THRESHOLD:
            print(f"  Skipping (agricultural content, {neg_hits} neg hits): {url}")
            pages_skipped += 1
            continue

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_title = title
            if len(chunks) > 1:
                chunk_title = f"{title} (Part {i+1}/{len(chunks)})"

            category = classify_category(chunk)
            records.append({
                "source_url": url,
                "source_name": name,
                "title": chunk_title,
                "content": chunk,
                "category": category,
            })

    print(f"  Pages fetched: {pages_fetched}")
    print(f"  Pages skipped: {pages_skipped}")
    print(f"  Chunks produced: {len(records)}")

    return records


def main() -> None:
    """Run the full crawl pipeline."""
    print("RAG Source Crawler")
    print(f"User-Agent: {USER_AGENT}")
    print(f"Rate limit: {RATE_LIMIT_SECONDS}s per domain")
    print(f"Output: {OUTPUT_FILE}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })

    all_records: list[dict] = []

    for source in SOURCE_REGISTRY:
        records = crawl_source(source, session)
        all_records.extend(records)

    # Compute stats
    source_counter = Counter(r["source_name"] for r in all_records)
    category_counter = Counter(r["category"] for r in all_records)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"Total chunks: {len(all_records)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nChunks by source:")
    for src, count in source_counter.most_common():
        print(f"  {src}: {count}")
    print("\nChunks by category:")
    for cat, count in category_counter.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
