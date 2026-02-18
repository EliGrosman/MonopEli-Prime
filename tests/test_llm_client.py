"""Tests for LLM client abstraction (all mocked, no real API calls)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mcts.llm.client import (
    ClaudeClient,
    LLMClient,
    LLMConfig,
    OllamaClient,
    OpenAIClient,
    _try_parse_json,
    create_client,
)

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

class TestTryParseJson:
    def test_valid_json(self) -> None:
        result = _try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fence(self) -> None:
        result = _try_parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_bare_code_fence(self) -> None:
        result = _try_parse_json('```\n{"key": 1}\n```')
        assert result == {"key": 1}

    def test_json_embedded_in_text(self) -> None:
        result = _try_parse_json('Here is the result: {"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self) -> None:
        assert _try_parse_json("not json at all") is None

    def test_empty_string_returns_none(self) -> None:
        assert _try_parse_json("") is None

    def test_array_returns_none(self) -> None:
        # We only accept dicts, not arrays
        assert _try_parse_json("[1, 2, 3]") is None

    def test_nested_json(self) -> None:
        text = '{"decision": "counter", "counter_proposal": {"give_money": 50}}'
        result = _try_parse_json(text)
        assert result is not None
        assert result["decision"] == "counter"

    def test_whitespace_handling(self) -> None:
        result = _try_parse_json('  \n  {"key": 1}  \n  ')
        assert result == {"key": 1}


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "claude"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 512
        assert cfg.max_retries == 2

    def test_custom(self) -> None:
        cfg = LLMConfig(provider="ollama", model="llama3.1", temperature=0.7)
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3.1"
        assert cfg.temperature == 0.7


# ---------------------------------------------------------------------------
# complete_json (on base class via a concrete mock subclass)
# ---------------------------------------------------------------------------

class _MockLLMClient(LLMClient):
    """Concrete LLMClient that returns predetermined responses."""

    def __init__(
        self, responses: list[str], config: LLMConfig | None = None,
    ) -> None:
        super().__init__(config or LLMConfig())
        self._responses = responses
        self._call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp

    def close(self) -> None:
        pass


class TestCompleteJson:
    def test_valid_json_first_try(self) -> None:
        client = _MockLLMClient(['{"to_player": 1, "give_money": 50}'])
        result = client.complete_json("sys", "user")
        assert result["to_player"] == 1
        assert client._call_count == 1

    def test_retries_on_invalid_json(self) -> None:
        client = _MockLLMClient([
            "I think the best trade is...",  # First: invalid
            '{"to_player": 1}',              # Retry: valid
        ])
        result = client.complete_json("sys", "user")
        assert result["to_player"] == 1
        assert client._call_count == 2

    def test_returns_error_dict_on_double_failure(self) -> None:
        client = _MockLLMClient(["not json", "still not json"])
        result = client.complete_json("sys", "user")
        assert "error" in result
        assert client._call_count == 2

    def test_handles_code_fenced_json(self) -> None:
        client = _MockLLMClient(['```json\n{"decision": "accept"}\n```'])
        result = client.complete_json("sys", "user")
        assert result["decision"] == "accept"


# ---------------------------------------------------------------------------
# create_client factory
# ---------------------------------------------------------------------------

class TestCreateClient:
    def test_creates_claude_client(self) -> None:
        cfg = LLMConfig(provider="claude")
        client = create_client(cfg)
        assert isinstance(client, ClaudeClient)
        client.close()

    def test_creates_openai_client(self) -> None:
        cfg = LLMConfig(provider="openai")
        client = create_client(cfg)
        assert isinstance(client, OpenAIClient)
        client.close()

    def test_creates_ollama_client(self) -> None:
        cfg = LLMConfig(provider="ollama")
        client = create_client(cfg)
        assert isinstance(client, OllamaClient)
        client.close()

    def test_unknown_provider_raises(self) -> None:
        cfg = LLMConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_client(cfg)

    def test_env_var_override(self) -> None:
        cfg = LLMConfig(provider="claude")
        with patch.dict("os.environ", {"MONOPOLY_LLM_PROVIDER": "ollama"}):
            client = create_client(cfg)
            assert isinstance(client, OllamaClient)
            client.close()

    def test_case_insensitive(self) -> None:
        cfg = LLMConfig(provider="Claude")
        client = create_client(cfg)
        assert isinstance(client, ClaudeClient)
        client.close()


# ---------------------------------------------------------------------------
# Provider-specific clients (mocked HTTP)
# ---------------------------------------------------------------------------

def _mock_httpx_response(data: dict[str, Any]) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "http://test"),
    )
    return resp


class TestClaudeClient:
    def test_complete_success(self) -> None:
        cfg = LLMConfig(provider="claude", model="test-model")
        client = ClaudeClient(cfg)

        mock_resp = _mock_httpx_response({
            "content": [{"text": '{"result": "ok"}'}],
        })
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.complete("system", "user")
        assert result == '{"result": "ok"}'

    def test_complete_retries_on_error(self) -> None:
        cfg = LLMConfig(provider="claude", max_retries=1)
        client = ClaudeClient(cfg)

        mock_fail = MagicMock(side_effect=httpx.HTTPError("timeout"))
        mock_ok = _mock_httpx_response({"content": [{"text": "ok"}]})
        client._client = MagicMock()
        client._client.post.side_effect = [mock_fail.side_effect, mock_ok]

        # Should fail once then succeed
        # Actually side_effect with mixed exceptions and values:
        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPError("timeout")
            return mock_ok

        client._client.post.side_effect = side_effect
        result = client.complete("sys", "user")
        assert result == "ok"
        assert call_count == 2


class TestOpenAIClient:
    def test_complete_success(self) -> None:
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
        client = OpenAIClient(cfg)

        mock_resp = _mock_httpx_response({
            "choices": [{"message": {"content": '{"trade": true}'}}],
        })
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.complete("system", "user")
        assert result == '{"trade": true}'


class TestOllamaClient:
    def test_complete_success(self) -> None:
        cfg = LLMConfig(provider="ollama", model="llama3.1")
        client = OllamaClient(cfg)

        mock_resp = _mock_httpx_response({
            "message": {"content": '{"decision": "accept"}'},
        })
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.complete("system", "user")
        assert result == '{"decision": "accept"}'

    def test_default_base_url(self) -> None:
        cfg = LLMConfig(provider="ollama")
        with patch.dict("os.environ", {}, clear=False):
            # Remove OLLAMA_BASE_URL if set
            import os
            old = os.environ.pop("OLLAMA_BASE_URL", None)
            try:
                client = OllamaClient(cfg)
                assert "localhost:11434" in str(client._client.base_url)
                client.close()
            finally:
                if old is not None:
                    os.environ["OLLAMA_BASE_URL"] = old
