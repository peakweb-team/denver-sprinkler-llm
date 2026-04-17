"""Inference engine — wraps llama-cli as an async subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List

from server.config import (
    INFERENCE_TIMEOUT,
    LLAMA_CLI_PATH,
    MAX_CONTEXT_CHARS,
    MAX_TOKENS,
    MOCK_MODE,
    MODEL_PATH,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from server.models import ChatMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock responses for testing without llama-cli / model binary
# ---------------------------------------------------------------------------
MOCK_RESPONSE = (
    "Thank you for reaching out to Denver Sprinkler and Landscape! "
    "We would be happy to help you. Please give us a call at (303) 993-8717 "
    "or email info@denversprinklerservices.com and we can discuss your needs. "
    "Our office is open Monday through Friday from 7am to 5pm and Saturday "
    "from 8am to 2pm. We are located at 3971 S Decatur St Unit A, "
    "Englewood, CO 80110. For emergencies, we are available 24/7."
)


# ---------------------------------------------------------------------------
# Prompt formatting — Llama 3.2 Instruct chat template
# ---------------------------------------------------------------------------

def format_prompt(
    messages: List[ChatMessage],
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Build a BitNet-b1.58-2B-4T chat-template prompt.

    Uses the model's native format: System:/User:/Assistant: with <|eot_id|> delimiters.
    Truncates older messages (keeping the system prompt and latest turns)
    if the formatted result would exceed *MAX_CONTEXT_CHARS*.
    """
    header = f"<|begin_of_text|>System: {system_prompt}<|eot_id|>"
    suffix = "Assistant:"
    budget = MAX_CONTEXT_CHARS - len(header) - len(suffix)

    # Map roles to BitNet format
    role_map = {"user": "User", "assistant": "Assistant"}

    # Build per-message fragments newest-first so we can drop the oldest.
    fragments: list[str] = []
    for msg in reversed(messages):
        role = role_map.get(msg.role, msg.role.capitalize())
        frag = f"{role}: {msg.content}<|eot_id|>"
        if budget - len(frag) < 0:
            break
        fragments.append(frag)
        budget -= len(frag)

    fragments.reverse()
    return header + "".join(fragments) + suffix


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Async wrapper around the llama-cli binary.

    A ``Semaphore(1)`` ensures only one inference process runs at a time
    (appropriate for a single-instance deployment on constrained hardware).
    """

    def __init__(self) -> None:
        self._mock = MOCK_MODE

        if not self._mock:
            if not os.path.isfile(LLAMA_CLI_PATH):
                raise FileNotFoundError(
                    f"llama-cli binary not found at {LLAMA_CLI_PATH}"
                )
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model file not found at {MODEL_PATH}"
                )

        self._semaphore = asyncio.Semaphore(1)
        logger.info(
            "InferenceEngine initialised (mock=%s, model=%s)",
            self._mock,
            MODEL_PATH,
        )

    # ---- public API -------------------------------------------------------

    async def generate(self, messages: List[ChatMessage]) -> str:
        """Run inference and return the assistant response text."""
        if self._mock:
            return MOCK_RESPONSE

        prompt = format_prompt(messages)

        async with self._semaphore:
            return await self._run_llama_cli(prompt)

    # ---- private helpers --------------------------------------------------

    async def _run_llama_cli(self, prompt: str) -> str:
        import tempfile
        # Write prompt to temp file to avoid shell escaping issues with special tokens
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            # Use shell wrapper to set unlimited stack size — bitnet.cpp's
            # optimized kernels use large stack allocations that exceed Python's
            # default subprocess stack limit.
            cmd = (
                f"ulimit -s unlimited && "
                f"{LLAMA_CLI_PATH} -m {MODEL_PATH} "
                f"--file {prompt_file} "
                f"-n {MAX_TOKENS} --ctx-size 2048 "
                f"--temp {TEMPERATURE} --top-p 0.9"
            )
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception:
            import os
            os.unlink(prompt_file)
            raise

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=b""),
                timeout=INFERENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"Inference timed out after {INFERENCE_TIMEOUT}s"
            )

        import os
        os.unlink(prompt_file)

        if proc.returncode != 0:
            err = stderr.decode(encoding="utf-8", errors="replace")
            logger.error("llama-cli exited %d: %s", proc.returncode, err)
            raise RuntimeError(f"llama-cli failed (exit {proc.returncode})")

        raw = stdout.decode(encoding="utf-8", errors="replace")
        return self._clean_output(raw)

    @staticmethod
    def _clean_output(raw: str) -> str:
        """Extract the assistant's response from llama-cli output."""
        # Find the last "Assistant:" in the output and take everything after it
        marker = "Assistant:"
        if marker in raw:
            text = raw.rsplit(marker, 1)[1]
        else:
            text = raw

        # Remove [end of text] marker
        text = text.replace("[end of text]", "")
        # Remove performance stats
        import re
        text = re.sub(r'\[.*?Prompt:.*?\]', '', text)
        text = re.sub(r'Exiting\.\.\.', '', text)
        return text.strip()
