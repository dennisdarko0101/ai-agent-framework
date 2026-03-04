"""Tests for LLM provider implementations (mocked — no real API calls)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import LLMProviderType, Settings
from src.llm.provider import (
    ClaudeProvider,
    LLMResponse,
    OpenAIProvider,
    ProviderFactory,
    TokenUsage,
    ToolCall,
)

# ───────────── Helpers for building mock responses ───────────────


def _mock_anthropic_response(
    text: str = "Hello",
    tool_use: list[dict[str, Any]] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
    model: str = "claude-sonnet-4-20250514",
    stop_reason: str = "end_turn",
):
    """Build a mock Anthropic Messages response."""
    blocks = []
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    text_block.model_dump = MagicMock(return_value={"type": "text", "text": text})
    blocks.append(text_block)

    if tool_use:
        for tu in tool_use:
            tb = MagicMock()
            tb.type = "tool_use"
            tb.id = tu.get("id", "toolu_mock")
            tb.name = tu["name"]
            tb.input = tu.get("input", {})
            tb.model_dump = MagicMock(return_value={
                "type": "tool_use",
                "id": tb.id,
                "name": tb.name,
                "input": tb.input,
            })
            blocks.append(tb)

    resp = MagicMock()
    resp.content = blocks
    resp.model = model
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


def _mock_openai_response(
    content: str = "Hello",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
):
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content

    if tool_calls:
        mock_tcs = []
        for tc in tool_calls:
            mtc = MagicMock()
            mtc.id = tc.get("id", "call_mock")
            mtc.type = "function"
            mtc.function = MagicMock()
            mtc.function.name = tc["name"]
            mtc.function.arguments = tc.get("arguments", "{}")
            mock_tcs.append(mtc)
        message.tool_calls = mock_tcs
    else:
        message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


# ────────────────── Claude provider tests ────────────────────────


class TestClaudeProvider:
    @pytest.mark.asyncio
    async def test_generate_text_response(self):
        provider = ClaudeProvider(api_key="test-key")
        mock_resp = _mock_anthropic_response(text="Paris is the capital of France.")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        assert isinstance(result, LLMResponse)
        assert "Paris" in result.content
        assert result.tool_calls == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self, weather_schema):
        provider = ClaudeProvider(api_key="test-key")
        mock_resp = _mock_anthropic_response(
            text="Let me check the weather.",
            tool_use=[{
                "id": "toolu_weather",
                "name": "get_weather",
                "input": {"location": "London"},
            }],
            stop_reason="tool_use",
        )
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Weather in London?"}],
            tools=[weather_schema],
        )

        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "London"}
        assert result.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_token_tracking(self):
        provider = ClaudeProvider(api_key="test-key")
        mock_resp = _mock_anthropic_response(input_tokens=50, output_tokens=100)
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        await provider.generate(messages=[{"role": "user", "content": "hi"}])
        assert provider.total_input_tokens == 50
        assert provider.total_output_tokens == 100

        # Second call accumulates
        await provider.generate(messages=[{"role": "user", "content": "hi"}])
        assert provider.total_input_tokens == 100
        assert provider.total_output_tokens == 200

    @pytest.mark.asyncio
    async def test_system_message_passed(self):
        provider = ClaudeProvider(api_key="test-key")
        mock_resp = _mock_anthropic_response()
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system="You are a pirate.",
        )

        call_kwargs = provider.client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "You are a pirate."

    @pytest.mark.asyncio
    async def test_generate_with_tools_delegates(self, weather_schema):
        provider = ClaudeProvider(api_key="test-key")
        mock_resp = _mock_anthropic_response()
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(return_value=mock_resp)

        result = await provider.generate_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[weather_schema],
        )
        assert isinstance(result, LLMResponse)


# ────────────────── OpenAI provider tests ────────────────────────


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_generate_text_response(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_resp = _mock_openai_response(content="Paris is the capital.")
        provider.client = MagicMock()
        provider.client.chat = MagicMock()
        provider.client.chat.completions = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Capital of France?"}],
        )

        assert "Paris" in result.content
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self, weather_schema):
        provider = OpenAIProvider(api_key="test-key")
        mock_resp = _mock_openai_response(
            content="",
            tool_calls=[{
                "id": "call_weather",
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            }],
            finish_reason="tool_calls",
        )
        provider.client = MagicMock()
        provider.client.chat = MagicMock()
        provider.client.chat.completions = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=[weather_schema],
        )

        assert result.has_tool_calls
        assert result.tool_calls[0].tool_name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "Tokyo"}

    @pytest.mark.asyncio
    async def test_system_message_prepended(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_resp = _mock_openai_response()
        provider.client = MagicMock()
        provider.client.chat = MagicMock()
        provider.client.chat.completions = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system="You are a pirate.",
        )

        call_kwargs = provider.client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs.get("messages", [])
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a pirate."

    @pytest.mark.asyncio
    async def test_token_tracking(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_resp = _mock_openai_response(prompt_tokens=25, completion_tokens=75)
        provider.client = MagicMock()
        provider.client.chat = MagicMock()
        provider.client.chat.completions = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await provider.generate(messages=[{"role": "user", "content": "hi"}])
        assert provider.total_input_tokens == 25
        assert provider.total_output_tokens == 75


# ────────────────── Data class tests ─────────────────────────────


class TestDataClasses:
    def test_tool_call_default_id(self):
        tc = ToolCall(tool_name="foo", arguments={"x": 1})
        assert tc.call_id.startswith("call_")

    def test_token_usage_total(self):
        u = TokenUsage(input_tokens=10, output_tokens=20)
        assert u.total_tokens == 30

    def test_llm_response_has_tool_calls(self):
        r = LLMResponse(content="hi")
        assert not r.has_tool_calls

        r2 = LLMResponse(
            content="",
            tool_calls=[ToolCall(tool_name="x", arguments={})],
        )
        assert r2.has_tool_calls


# ────────────────── Factory tests ────────────────────────────────


class TestProviderFactory:
    def test_create_claude(self):
        provider = ProviderFactory.create(
            provider_type=LLMProviderType.CLAUDE,
            api_key="test-key",
        )
        assert isinstance(provider, ClaudeProvider)

    def test_create_openai(self):
        provider = ProviderFactory.create(
            provider_type=LLMProviderType.OPENAI,
            api_key="test-key",
        )
        assert isinstance(provider, OpenAIProvider)

    def test_create_from_settings(self):
        settings = Settings(
            anthropic_api_key="key-a",
            default_llm_provider=LLMProviderType.CLAUDE,
            default_model="claude-sonnet-4-20250514",
        )
        provider = ProviderFactory.create(settings=settings)
        assert isinstance(provider, ClaudeProvider)
        assert provider.model == "claude-sonnet-4-20250514"
