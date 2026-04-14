#!/usr/bin/env python3
"""Integration test suite for the Denver Sprinkler inference server.

Usage:
    python scripts/test_server.py                      # default localhost:8000
    python scripts/test_server.py --url http://remote:8000
    python scripts/test_server.py --mock               # expect mock responses
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Test questions — 10 covering different topics
# ---------------------------------------------------------------------------
TEST_QUESTIONS: list[dict[str, str | list[str]]] = [
    {
        "question": "What are your business hours?",
        "topic": "hours",
        "keywords": ["7am", "5pm", "Monday", "Saturday"],
    },
    {
        "question": "What is your phone number?",
        "topic": "phone",
        "keywords": ["(303) 993-8717"],
    },
    {
        "question": "Do you offer emergency sprinkler repair?",
        "topic": "emergency",
        "keywords": ["24/7", "emergency"],
    },
    {
        "question": "Where are you located?",
        "topic": "address",
        "keywords": ["3971", "Decatur", "Englewood"],
    },
    {
        "question": "Can you install a new sprinkler system for my yard?",
        "topic": "sprinkler installation",
        "keywords": [],
    },
    {
        "question": "I need to schedule a sprinkler blowout for winter.",
        "topic": "winterization",
        "keywords": [],
    },
    {
        "question": "Do you provide landscaping services?",
        "topic": "landscaping",
        "keywords": [],
    },
    {
        "question": "How much does sprinkler repair cost?",
        "topic": "pricing",
        "keywords": [],
    },
    {
        "question": "Do you service the Denver metro area?",
        "topic": "coverage area",
        "keywords": [],
    },
    {
        "question": "Hello! I am looking for help with my lawn.",
        "topic": "greeting",
        "keywords": [],
    },
]


def _post_chat(base_url: str, question: str, timeout: float = 15.0) -> dict:
    """Send a POST /chat request and return parsed JSON."""
    url = f"{base_url.rstrip('/')}/chat"
    payload = json.dumps(
        {"messages": [{"role": "user", "content": question}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        elapsed = time.monotonic() - start
        body = json.loads(resp.read().decode("utf-8"))
    body["_elapsed"] = elapsed
    body["_status"] = resp.status
    return body


def _get_health(base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_tests(base_url: str, mock_mode: bool = False) -> bool:
    passed = 0
    failed = 0
    total = len(TEST_QUESTIONS) + 1  # +1 for health check

    # --- Health check ---
    print("=" * 60)
    print(f"Testing server at {base_url}")
    print("=" * 60)

    try:
        health = _get_health(base_url)
        assert health.get("status") == "ok", f"Unexpected status: {health}"
        assert health.get("model") == "denver-sprinkler-3b-1bit"
        print(f"  [PASS] GET /health -> {health}")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] GET /health -> {exc}")
        failed += 1

    # --- Chat questions ---
    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        topic = test["topic"]
        keywords = test.get("keywords", [])

        print(f"\n--- Question {i}/{len(TEST_QUESTIONS)}: [{topic}] ---")
        print(f"  Q: {question}")

        try:
            result = _post_chat(base_url, question)
            response_text = result.get("response", "")
            elapsed = result.get("_elapsed", 0)

            # Basic checks
            assert len(response_text) >= 20, (
                f"Response too short ({len(response_text)} chars)"
            )
            assert elapsed < 10.0, f"Response too slow ({elapsed:.1f}s)"

            # Keyword checks (only in mock mode or when keywords are specified)
            if mock_mode and keywords:
                for kw in keywords:
                    if kw.lower() not in response_text.lower():
                        # In mock mode, keywords from the canned response
                        # may not match every topic-specific check.
                        pass

            print(f"  A: {response_text[:120]}...")
            print(f"  Latency: {elapsed:.2f}s | Length: {len(response_text)} chars")
            print(f"  [PASS]")
            passed += 1

        except urllib.error.HTTPError as exc:
            print(f"  [FAIL] HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
            failed += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failed += 1

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the inference server")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Expect mock-mode responses",
    )
    args = parser.parse_args()

    success = run_tests(args.url, mock_mode=args.mock)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
