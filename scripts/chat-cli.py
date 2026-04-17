#!/usr/bin/env python3
"""Interactive CLI tool for testing the Denver Sprinkler inference API.

Usage examples:
    # Interactive REPL (default)
    python scripts/chat-cli.py

    # Connect to a specific server
    python scripts/chat-cli.py --url http://my-server:8000

    # Batch mode — run prompts from a file
    python scripts/chat-cli.py --batch data/test-prompts.txt

    # Save conversation transcript
    python scripts/chat-cli.py --transcript

    # Batch mode with transcript
    python scripts/chat-cli.py --batch data/test-prompts.txt --transcript
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"

WELCOME = """
Denver Sprinkler Chat CLI
=========================
Commands:
  /reset    Clear conversation history
  /history  Show conversation history
  /quit     Exit (or /exit, or Ctrl+C)
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def health_check(base_url: str, timeout: int) -> dict[str, Any] | None:
    """Call GET /health and return the parsed JSON, or None on failure."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        print(f"Cannot connect to {base_url}. Is the server running?")
        return None
    except requests.Timeout:
        print(f"Health check timed out after {timeout}s.")
        return None
    except Exception as exc:
        print(f"Health check failed: {exc}")
        return None


def send_chat(
    base_url: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> tuple[str | None, float, str | None]:
    """Send POST /chat and return (response_text, latency_ms, error_string).

    On success error_string is None; on failure response_text is None.
    """
    payload = {"messages": messages}
    start = time.monotonic()
    try:
        resp = requests.post(
            f"{base_url}/chat",
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            return None, latency_ms, "Rate limited. Waiting and retrying..."
        if resp.status_code >= 500:
            return None, latency_ms, f"Server error ({resp.status_code}). Try again or check server logs."

        try:
            data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            return None, latency_ms, "Unexpected response format from server."

        if resp.status_code != 200:
            detail = data.get("detail", resp.text)
            return None, latency_ms, f"Error ({resp.status_code}): {detail}"

        return data.get("response", ""), latency_ms, None

    except requests.ConnectionError:
        latency_ms = (time.monotonic() - start) * 1000
        return None, latency_ms, f"Cannot connect to {base_url}. Is the server running?"
    except requests.Timeout:
        latency_ms = (time.monotonic() - start) * 1000
        return None, latency_ms, f"Request timed out after {timeout}s. The model may be under heavy load."
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return None, latency_ms, f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def save_transcript(
    base_url: str,
    messages: list[dict[str, str]],
    latencies: list[float],
) -> Path:
    """Write a JSON transcript and return the file path."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"chat-{now.strftime('%Y-%m-%d-%H%M%S')}.json"
    path = TRANSCRIPT_DIR / filename

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    transcript = {
        "timestamp": now.isoformat(),
        "endpoint": base_url,
        "messages": messages,
        "metadata": {
            "total_turns": len([m for m in messages if m["role"] == "user"]),
            "avg_latency_ms": round(avg_latency, 1),
        },
    }

    path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------

def run_repl(base_url: str, timeout: int, save_tx: bool) -> None:
    """Run the interactive REPL loop."""
    # Health check
    info = health_check(base_url, timeout)
    if info is None:
        sys.exit(1)

    print(f"Connected to {base_url}")
    print(f"Model: {info.get('model', 'unknown')}  Status: {info.get('status', '?')}  Version: {info.get('version', '?')}")
    print(WELCOME)

    messages: list[dict[str, str]] = []
    latencies: list[float] = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.lower() in ("/quit", "/exit"):
                break
            if user_input.lower() == "/reset":
                messages.clear()
                latencies.clear()
                print("Conversation history cleared.\n")
                continue
            if user_input.lower() == "/history":
                if not messages:
                    print("(no messages yet)\n")
                else:
                    for msg in messages:
                        role = "You" if msg["role"] == "user" else "Bot"
                        print(f"  {role}: {msg['content']}")
                    print()
                continue

            # Build request
            messages.append({"role": "user", "content": user_input})
            text, latency_ms, error = send_chat(base_url, messages, timeout)

            if error:
                # Remove the failed user message so history stays clean
                messages.pop()
                print(f"Error: {error}\n")
                continue

            messages.append({"role": "assistant", "content": text})
            latencies.append(latency_ms)
            print(f"Bot: {text}  [{latency_ms:.0f}ms]\n")

    except KeyboardInterrupt:
        print("\n")

    # Transcript
    if save_tx and messages:
        path = save_transcript(base_url, messages, latencies)
        print(f"Transcript saved to {path}")

    print("Goodbye!")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(base_url: str, batch_file: str, timeout: int, save_tx: bool) -> None:
    """Run prompts from a file in single-turn mode."""
    # Health check
    info = health_check(base_url, timeout)
    if info is None:
        sys.exit(1)

    print(f"Connected to {base_url}")
    print(f"Model: {info.get('model', 'unknown')}  Status: {info.get('status', '?')}")
    print(f"Running batch from: {batch_file}\n")

    path = Path(batch_file)
    if not path.exists():
        print(f"File not found: {batch_file}")
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    if not prompts:
        print("No prompts found in file.")
        sys.exit(1)

    total = len(prompts)
    success = 0
    failed = 0
    latencies: list[float] = []
    all_messages: list[dict[str, str]] = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{total}] You: {prompt}")
        single_msg = [{"role": "user", "content": prompt}]
        text, latency_ms, error = send_chat(base_url, single_msg, timeout)

        if error:
            failed += 1
            print(f"       Error: {error}\n")
        else:
            success += 1
            latencies.append(latency_ms)
            print(f"       Bot: {text}  [{latency_ms:.0f}ms]\n")
            all_messages.append({"role": "user", "content": prompt})
            all_messages.append({"role": "assistant", "content": text})

    # Summary
    print("=" * 50)
    print(f"Batch complete: {success}/{total} succeeded, {failed} failed")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"Latency — avg: {avg:.0f}ms  min: {min(latencies):.0f}ms  max: {max(latencies):.0f}ms")
    print("=" * 50)

    # Transcript
    if save_tx and all_messages:
        tx_path = save_transcript(base_url, all_messages, latencies)
        print(f"Transcript saved to {tx_path}")

    sys.exit(1 if failed > 0 else 0)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat-cli",
        description="Interactive CLI for the Denver Sprinkler inference API.",
        epilog="""Examples:
  python scripts/chat-cli.py
  python scripts/chat-cli.py --url http://my-server:8000
  python scripts/chat-cli.py --batch data/test-prompts.txt
  python scripts/chat-cli.py --transcript
  python scripts/chat-cli.py --batch data/test-prompts.txt --transcript --timeout 60
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"API endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--batch",
        metavar="FILE",
        help="Run prompts from a text file (one per line) instead of interactive REPL",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Save conversation to a timestamped JSON file in data/transcripts/",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    url = args.url.rstrip("/")

    if args.batch:
        run_batch(url, args.batch, args.timeout, args.transcript)
    else:
        run_repl(url, args.timeout, args.transcript)


if __name__ == "__main__":
    main()
