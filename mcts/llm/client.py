"""LLM client abstraction for synchronous trade generation.

All clients are **synchronous** (httpx sync). This is intentional:
``Agent.choose_action()`` is sync and there is no benefit to async
in a turn-based game where we make one LLM call at a time.

Supported providers:
- ``claude``: Anthropic Messages API
- ``openai``: OpenAI Chat Completions API
- ``ollama``: Local Ollama REST API
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for an LLM client."""

    provider: str = "claude"
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.3
    max_tokens: int = 512
    api_key_env_var: str = ""
    base_url: str = ""
    timeout_seconds: float = 10.0
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement :meth:`complete` for raw text completion.
    The :meth:`complete_json` convenience method parses JSON from the
    response and retries once on parse failure.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send a prompt and return the raw response text."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a prompt and parse JSON from the response.

        On JSON parse failure, retries once with a corrective follow-up.
        """
        raw = self.complete(system_prompt, user_prompt)
        result = _try_parse_json(raw)
        if result is not None:
            return result

        # Retry: ask the LLM to fix its JSON
        logger.warning("JSON parse failed, retrying with correction prompt")
        retry_prompt = (
            f"Your previous response was not valid JSON. "
            f"The raw response was:\n{raw}\n\n"
            f"Please respond with ONLY valid JSON, no other text."
        )
        raw_retry = self.complete(system_prompt, retry_prompt)
        result = _try_parse_json(raw_retry)
        if result is not None:
            return result

        # Give up -- return a minimal error dict
        logger.error("JSON parse failed after retry: %s", raw_retry[:200])
        return {"error": "Failed to parse JSON", "raw": raw_retry[:500]}

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (e.g., close HTTP session)."""


# ---------------------------------------------------------------------------
# Claude (Anthropic Messages API)
# ---------------------------------------------------------------------------

class ClaudeClient(LLMClient):
    """Anthropic Claude API client using httpx sync."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        api_key = os.environ.get(
            config.api_key_env_var or "ANTHROPIC_API_KEY", ""
        )
        self._client = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=config.timeout_seconds,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post("/v1/messages", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return str(data["content"][0]["text"])
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                logger.warning(
                    "Claude API attempt %d failed: %s", attempt + 1, exc,
                )
                if attempt == self.config.max_retries:
                    raise
        return ""  # pragma: no cover

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# OpenAI (Chat Completions API)
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """OpenAI Chat Completions API client using httpx sync."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        api_key = os.environ.get(
            config.api_key_env_var or "OPENAI_API_KEY", ""
        )
        base = config.base_url or "https://api.openai.com"
        self._client = httpx.Client(
            base_url=base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post(
                    "/v1/chat/completions", json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                logger.warning(
                    "OpenAI API attempt %d failed: %s", attempt + 1, exc,
                )
                if attempt == self.config.max_retries:
                    raise
        return ""  # pragma: no cover

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Ollama (local REST API)
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    """Local Ollama REST API client using httpx sync."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        base = (
            config.base_url
            or os.environ.get("OLLAMA_BASE_URL", "")
            or "http://localhost:11434"
        )
        self._client = httpx.Client(
            base_url=base,
            timeout=config.timeout_seconds,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return str(data["message"]["content"])
            except (httpx.HTTPError, KeyError) as exc:
                logger.warning(
                    "Ollama API attempt %d failed: %s", attempt + 1, exc,
                )
                if attempt == self.config.max_retries:
                    raise
        return ""  # pragma: no cover

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMClient]] = {
    "claude": ClaudeClient,
    "openai": OpenAIClient,
    "ollama": OllamaClient,
}


def create_client(config: LLMConfig) -> LLMClient:
    """Factory function to create the appropriate LLM client.

    The provider is resolved from ``config.provider`` or the
    ``MONOPOLY_LLM_PROVIDER`` environment variable.
    """
    provider = (
        os.environ.get("MONOPOLY_LLM_PROVIDER", "") or config.provider
    ).lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Choose from: {', '.join(_PROVIDERS)}"
        )
    return cls(config)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse JSON from LLM output.

    Handles common LLM quirks: markdown code fences, leading text.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
