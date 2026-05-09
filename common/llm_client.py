"""
Claude LLM client.

WHY A WRAPPER (instead of using anthropic SDK directly everywhere):
  1. Single place to add observability, retries, cost tracking
  2. Easy to swap models per-agent (Haiku for cheap classification,
     Sonnet for complex reasoning, Opus for highest quality)
  3. Centralized prompt-template loading
  4. Test seam — agents take a `LLMClient` interface, easy to mock

Reads ANTHROPIC_API_KEY from env. The user said they'll configure this
separately; we never hardcode it.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from common.logging_utils import get_logger
from common.observability import trace_span

logger = get_logger(__name__)

# Model tiers — pick by task complexity vs cost.
# Verify current model strings against Anthropic docs at deploy time.
MODEL_FAST = "claude-haiku-4-5"        # cheap, fast — classification, extraction
MODEL_BALANCED = "claude-sonnet-4-5"   # default — most agent reasoning
MODEL_DEEP = "claude-opus-4-5"         # highest quality — complex multi-step reasoning


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_BALANCED,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_model = default_model
        self.max_retries = max_retries
        self._client: Any = None
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — LLM calls will fail until configured")

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic  # lazy import
                self._client = Anthropic(api_key=self.api_key, max_retries=self.max_retries)
            except ImportError as e:
                raise RuntimeError("Install `anthropic` package: pip install anthropic") from e
        return self._client

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Single-shot completion. Returns dict with 'text', 'tool_uses', 'usage'.

        We deliberately do NOT use streaming here — the gateway response is
        synchronous JSON-RPC. Streaming belongs in the agent's `message/stream`
        method (we'd add that as a future extension).
        """
        client = self._ensure_client()
        chosen_model = model or self.default_model

        with trace_span(
            "llm.complete",
            input={"model": chosen_model, "system_preview": system[:120]},
            metadata={"temperature": temperature, "max_tokens": max_tokens},
        ) as span:
            kwargs: dict[str, Any] = {
                "model": chosen_model,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            resp = client.messages.create(**kwargs)

            # Extract text + any tool_use blocks
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id, "name": block.name, "input": block.input,
                    })

            result = {
                "text": "".join(text_parts),
                "tool_uses": tool_uses,
                "stop_reason": resp.stop_reason,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            }
            span["output"] = {
                "stop_reason": result["stop_reason"],
                "usage": result["usage"],
                "text_preview": result["text"][:160],
            }
            return result
