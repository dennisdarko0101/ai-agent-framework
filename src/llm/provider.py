"""LLM provider abstraction with Claude and OpenAI implementations."""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

from src.config.settings import LLMProviderType, Settings, get_settings
from src.llm.function_calling import (
    ToolSchema,
    convert_to_anthropic_tools,
    convert_to_openai_tools,
    parse_anthropic_tool_calls,
    parse_openai_tool_calls,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool/function call requested by the LLM."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class TokenUsage:
    """Token consumption for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    stop_reason: str = ""
    raw: Any = None  # Original provider response for debugging

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract LLM provider — one implementation per backend."""

    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a response, optionally with tool definitions."""
        ...

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Convenience wrapper that always passes tools."""
        return await self.generate(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

    def _track_usage(self, usage: TokenUsage) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens

    async def _retry_with_backoff(self, coro_factory, *, retries: int | None = None):  # type: ignore[no-untyped-def]
        """Execute an async callable with exponential backoff on transient errors."""
        retries = retries or self.max_retries
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return await coro_factory()
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                openai.RateLimitError,
                openai.APIConnectionError,
            ) as exc:
                last_exc = exc
                wait = 2 ** attempt
                log.warning(
                    "llm_retry",
                    attempt=attempt + 1,
                    wait=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)
            except (
                anthropic.APIStatusError,
                openai.APIStatusError,
            ):
                # Non-transient API error — do not retry
                raise
        raise RuntimeError(f"All {retries} retries exhausted") from last_exc

    # ── Multi-turn tool-use message building ──────────────────────────

    def build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        """Build the assistant message dict for multi-turn tool-use conversations.

        Subclasses override to use their native tool-calling format.
        """
        return {"role": "assistant", "content": response.content}

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCall],
        results: list[str],
        errors: list[bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Build message(s) containing tool execution results.

        Args:
            tool_calls: The tool calls that were executed.
            results: The text output from each tool call.
            errors: Optional flags indicating which results are errors.
        """
        parts: list[str] = []
        for i, (tc, result) in enumerate(zip(tool_calls, results, strict=False)):
            prefix = "ERROR: " if (errors and errors[i]) else ""
            parts.append(f"[{tc.tool_name}]: {prefix}{result}")
        return [{"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------------------
# Claude (Anthropic) provider
# ---------------------------------------------------------------------------

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude with native tool use."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        max_retries: int = 3,
    ):
        super().__init__(model=model, api_key=api_key, max_retries=max_retries)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = convert_to_anthropic_tools(tools)

        response = await self._retry_with_backoff(
            lambda: self.client.messages.create(**kwargs)
        )

        # Parse content blocks
        text_parts: list[str] = []
        raw_blocks: list[dict[str, Any]] = []
        for block in response.content:
            block_dict = block.model_dump() if hasattr(block, "model_dump") else dict(block)
            raw_blocks.append(block_dict)
            if block.type == "text":
                text_parts.append(block.text)

        # Parse tool calls
        parsed_calls = parse_anthropic_tool_calls(raw_blocks)
        tool_calls = [
            ToolCall(
                tool_name=tc["tool_name"],
                arguments=tc["arguments"],
                call_id=tc["call_id"],
            )
            for tc in parsed_calls
        ]

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        self._track_usage(usage)

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            stop_reason=response.stop_reason or "",
            raw=response,
        )

    def build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if response.content:
            content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.tool_name,
                "input": tc.arguments,
            })
        return {"role": "assistant", "content": content}

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCall],
        results: list[str],
        errors: list[bool] | None = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for i, (tc, result) in enumerate(zip(tool_calls, results, strict=False)):
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tc.call_id,
                "content": result,
            }
            if errors and errors[i]:
                block["is_error"] = True
            content.append(block)
        return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT with function calling."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        max_retries: int = 3,
    ):
        super().__init__(model=model, api_key=api_key, max_retries=max_retries)
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        # Prepend system message if provided
        all_messages = list(messages)
        if system:
            all_messages.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = convert_to_openai_tools(tools)

        response = await self._retry_with_backoff(
            lambda: self.client.chat.completions.create(**kwargs)
        )

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        raw_tool_calls = None
        if message.tool_calls:
            raw_tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        parsed_calls = parse_openai_tool_calls(raw_tool_calls)
        tool_calls = [
            ToolCall(
                tool_name=tc["tool_name"],
                arguments=tc["arguments"],
                call_id=tc["call_id"],
            )
            for tc in parsed_calls
        ]

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        self._track_usage(usage)

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            stop_reason=choice.finish_reason or "",
            raw=response,
        )

    def build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": response.content or None}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCall],
        results: list[str],
        errors: list[bool] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": tc.call_id, "content": result}
            for tc, result in zip(tool_calls, results, strict=False)
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ProviderFactory:
    """Create an LLM provider from settings or explicit parameters."""

    @staticmethod
    def create(
        provider_type: LLMProviderType | None = None,
        model: str | None = None,
        api_key: str | None = None,
        settings: Settings | None = None,
    ) -> BaseLLMProvider:
        settings = settings or get_settings()
        provider_type = provider_type or settings.default_llm_provider
        model = model or settings.default_model
        api_key = api_key or settings.get_api_key(provider_type)

        if provider_type == LLMProviderType.CLAUDE:
            return ClaudeProvider(model=model, api_key=api_key)
        elif provider_type == LLMProviderType.OPENAI:
            return OpenAIProvider(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
