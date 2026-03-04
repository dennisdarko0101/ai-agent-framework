"""Tests for BaseAgent data classes and ReActAgent — 25+ tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentAction, AgentResponse, AgentStep, BaseAgent
from src.agents.react import ReActAgent
from src.llm.provider import LLMResponse, TokenUsage, ToolCall
from src.memory.conversation import ConversationMemory
from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry

# ───────── Helpers ───────────────────────────────────────────────────────

class _AddTool(BaseTool):
    name = "add"
    description = "Add two numbers"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=str(arguments["a"] + arguments["b"]))


def _make_llm(**overrides: Any) -> AsyncMock:
    """Create a mock LLM provider with sensible defaults."""
    llm = AsyncMock()
    llm.build_assistant_message = MagicMock(
        return_value={"role": "assistant", "content": "thinking..."},
    )
    llm.build_tool_result_messages = MagicMock(
        return_value=[{"role": "user", "content": "result"}],
    )
    for k, v in overrides.items():
        setattr(llm, k, v)
    return llm


def _response(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ───────── AgentAction ──────────────────────────────────────────────────

class TestAgentAction:
    def test_defaults(self) -> None:
        action = AgentAction(tool_name="calc", arguments={"x": 1})
        assert action.tool_name == "calc"
        assert action.success is True
        assert action.result_output == ""

    def test_with_error(self) -> None:
        action = AgentAction(
            tool_name="calc", arguments={},
            result_error="boom", success=False,
        )
        assert action.success is False
        assert action.result_error == "boom"


# ───────── AgentStep ────────────────────────────────────────────────────

class TestAgentStep:
    def test_defaults(self) -> None:
        step = AgentStep(step_number=1)
        assert step.reasoning == ""
        assert step.actions == []
        assert step.observation == ""
        assert isinstance(step.timestamp, float)

    def test_with_actions(self) -> None:
        action = AgentAction(tool_name="t", arguments={})
        step = AgentStep(step_number=2, reasoning="think", actions=[action])
        assert len(step.actions) == 1


# ───────── AgentResponse ────────────────────────────────────────────────

class TestAgentResponse:
    def test_defaults(self) -> None:
        r = AgentResponse()
        assert r.success is True
        assert r.output == ""
        assert r.error is None
        assert r.total_tokens == 0
        assert r.metadata == {}

    def test_error_sets_success_false(self) -> None:
        r = AgentResponse(error="something broke")
        assert r.success is False

    def test_explicit_success_overridden_by_error(self) -> None:
        r = AgentResponse(success=True, error="bad")
        assert r.success is False

    def test_metadata(self) -> None:
        r = AgentResponse(metadata={"key": "val"})
        assert r.metadata["key"] == "val"


# ───────── BaseAgent (abstract) ─────────────────────────────────────────

class TestBaseAgent:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseAgent(llm=AsyncMock())  # type: ignore[abstract]

    def test_repr(self) -> None:
        llm = AsyncMock()
        agent = ReActAgent(llm=llm, name="test-bot")
        assert "test-bot" in repr(agent)


# ───────── ReActAgent — no tools ────────────────────────────────────────

class TestReActAgentNoTools:
    @pytest.mark.asyncio
    async def test_simple_response(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("The answer is 42."))
        agent = ReActAgent(llm=llm, name="test")
        result = await agent.run("What is the answer?")
        assert result.success
        assert result.output == "The answer is 42."
        assert len(result.steps) == 1
        assert result.steps[0].actions == []

    @pytest.mark.asyncio
    async def test_system_prompt_used(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("ok"))
        agent = ReActAgent(llm=llm, system_prompt="Be concise.")
        await agent.run("hi")
        call_kwargs = llm.generate.call_args[1]
        assert call_kwargs["system"] == "Be concise."

    @pytest.mark.asyncio
    async def test_default_system_prompt(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("ok"))
        agent = ReActAgent(llm=llm)
        await agent.run("hi")
        call_kwargs = llm.generate.call_args[1]
        assert "step by step" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_empty_task(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response(""))
        agent = ReActAgent(llm=llm)
        result = await agent.run("")
        assert result.success  # empty but no error

    @pytest.mark.asyncio
    async def test_token_counting(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("answer"))
        agent = ReActAgent(llm=llm)
        result = await agent.run("q")
        assert result.total_tokens == 15  # 10 + 5

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("ok"))
        agent = ReActAgent(llm=llm)
        result = await agent.run("q")
        assert result.execution_time > 0


# ───────── ReActAgent — with tools ──────────────────────────────────────

class TestReActAgentWithTools:
    @pytest.fixture(autouse=True)
    def _reset_registry(self) -> None:
        ToolRegistry.reset()

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(_AddTool())
        return reg

    @pytest.mark.asyncio
    async def test_single_tool_call_and_answer(self, registry: ToolRegistry) -> None:
        tool_response = _response(
            "Let me add those.",
            [ToolCall(tool_name="add", arguments={"a": 2, "b": 3}, call_id="c1")],
        )
        final_response = _response("The result is 5.")
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=[tool_response, final_response])

        agent = ReActAgent(llm=llm, tools=registry)
        result = await agent.run("Add 2 and 3")

        assert result.success
        assert result.output == "The result is 5."
        assert len(result.steps) == 2
        assert result.steps[0].actions[0].tool_name == "add"
        assert result.steps[0].actions[0].result_output == "5"

    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self, registry: ToolRegistry) -> None:
        responses = [
            _response("Step 1", [ToolCall(tool_name="add", arguments={"a": 1, "b": 2}, call_id="c1")]),
            _response("Step 2", [ToolCall(tool_name="add", arguments={"a": 3, "b": 4}, call_id="c2")]),
            _response("Final: 3 and 7"),
        ]
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=responses)

        agent = ReActAgent(llm=llm, tools=registry)
        result = await agent.run("Compute both")

        assert result.success
        assert len(result.steps) == 3
        assert result.total_tokens == 45  # 15 * 3

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self, registry: ToolRegistry) -> None:
        # LLM always requests tools — should hit max_steps
        tool_resp = _response(
            "again", [ToolCall(tool_name="add", arguments={"a": 0, "b": 0}, call_id="c")],
        )
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=tool_resp)

        agent = ReActAgent(llm=llm, tools=registry, max_steps=3)
        result = await agent.run("loop forever")

        assert not result.success
        assert result.error == "max_steps_exceeded"
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_tool_execution_error_handled(self, registry: ToolRegistry) -> None:
        # Request a non-existent tool
        tool_resp = _response(
            "try this",
            [ToolCall(tool_name="nonexistent", arguments={}, call_id="c1")],
        )
        final_resp = _response("That tool failed, here's my answer.")
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=[tool_resp, final_resp])

        agent = ReActAgent(llm=llm, tools=registry)
        result = await agent.run("do something")

        assert result.success
        assert result.steps[0].actions[0].success is False

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_single_step(self, registry: ToolRegistry) -> None:
        tool_resp = _response(
            "Two additions",
            [
                ToolCall(tool_name="add", arguments={"a": 1, "b": 1}, call_id="c1"),
                ToolCall(tool_name="add", arguments={"a": 2, "b": 2}, call_id="c2"),
            ],
        )
        final_resp = _response("Done: 2 and 4")
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=[tool_resp, final_resp])

        agent = ReActAgent(llm=llm, tools=registry)
        result = await agent.run("two calcs")

        assert result.success
        assert len(result.steps[0].actions) == 2
        assert result.steps[0].actions[0].result_output == "2"
        assert result.steps[0].actions[1].result_output == "4"

    @pytest.mark.asyncio
    async def test_no_tools_configured(self) -> None:
        tool_resp = _response(
            "I'll use a tool",
            [ToolCall(tool_name="anything", arguments={}, call_id="c1")],
        )
        final_resp = _response("No tools available, giving up.")
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=[tool_resp, final_resp])

        agent = ReActAgent(llm=llm, tools=None)
        result = await agent.run("do it")

        assert result.steps[0].actions[0].success is False
        assert "No tool registry" in result.steps[0].actions[0].result_error

    @pytest.mark.asyncio
    async def test_step_tracking(self, registry: ToolRegistry) -> None:
        tool_resp = _response(
            "reasoning text",
            [ToolCall(tool_name="add", arguments={"a": 5, "b": 5}, call_id="c1")],
        )
        final_resp = _response("10")
        llm = _make_llm()
        llm.generate = AsyncMock(side_effect=[tool_resp, final_resp])

        agent = ReActAgent(llm=llm, tools=registry)
        result = await agent.run("5+5")

        step0 = result.steps[0]
        assert step0.step_number == 1
        assert step0.reasoning == "reasoning text"
        assert "[add]: 10" in step0.observation

    @pytest.mark.asyncio
    async def test_custom_max_steps(self) -> None:
        agent = ReActAgent(llm=AsyncMock(), max_steps=25)
        assert agent.max_steps == 25


# ───────── ReActAgent — memory integration ──────────────────────────────

class TestReActAgentMemory:
    @pytest.mark.asyncio
    async def test_memory_receives_messages(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("hello back"))
        memory = ConversationMemory(max_messages=10)

        agent = ReActAgent(llm=llm, memory=memory)
        await agent.run("hello")

        stats = memory.get_stats()
        assert stats["message_count"] == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_memory_context_sent_to_llm(self) -> None:
        llm = _make_llm()
        llm.generate = AsyncMock(return_value=_response("ok"))
        memory = ConversationMemory(max_messages=10)
        await memory.add("user", "prior message")

        agent = ReActAgent(llm=llm, memory=memory)
        await agent.run("new message")

        call_kwargs = llm.generate.call_args[1]
        messages = call_kwargs["messages"]
        # Should contain prior + new
        assert len(messages) >= 2
        assert messages[0]["content"] == "prior message"


# ───────── Provider message building ────────────────────────────────────

class TestProviderMessageBuilding:
    def test_base_build_assistant_message(self) -> None:
        from src.llm.provider import BaseLLMProvider

        class _Dummy(BaseLLMProvider):
            async def generate(self, messages, **kw):  # type: ignore[override]
                return LLMResponse()

        provider = _Dummy(model="test", api_key="k")
        resp = LLMResponse(content="hi", tool_calls=[])
        msg = provider.build_assistant_message(resp)
        assert msg == {"role": "assistant", "content": "hi"}

    def test_base_build_tool_result_messages(self) -> None:
        from src.llm.provider import BaseLLMProvider

        class _Dummy(BaseLLMProvider):
            async def generate(self, messages, **kw):  # type: ignore[override]
                return LLMResponse()

        provider = _Dummy(model="test", api_key="k")
        tc = ToolCall(tool_name="calc", arguments={}, call_id="c1")
        msgs = provider.build_tool_result_messages([tc], ["42"])
        assert len(msgs) == 1
        assert "[calc]: 42" in msgs[0]["content"]

    def test_base_build_tool_result_with_errors(self) -> None:
        from src.llm.provider import BaseLLMProvider

        class _Dummy(BaseLLMProvider):
            async def generate(self, messages, **kw):  # type: ignore[override]
                return LLMResponse()

        provider = _Dummy(model="test", api_key="k")
        tc = ToolCall(tool_name="calc", arguments={}, call_id="c1")
        msgs = provider.build_tool_result_messages([tc], ["fail"], errors=[True])
        assert "ERROR:" in msgs[0]["content"]
