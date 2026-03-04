"""Tests for orchestration patterns — 22+ tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentResponse, AgentStep, BaseAgent
from src.agents.react import ReActAgent
from src.llm.provider import LLMResponse, TokenUsage
from src.orchestration.pipeline import AgentPipeline
from src.orchestration.router import TaskRouter
from src.orchestration.supervisor import AgentSupervisor

# ───────── Helpers ───────────────────────────────────────────────────────

def _make_agent(name: str, output: str, success: bool = True) -> ReActAgent:
    """Create a ReActAgent whose run() returns a canned response."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content=output,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
    )
    llm.build_assistant_message = MagicMock(return_value={"role": "assistant", "content": ""})
    llm.build_tool_result_messages = MagicMock(return_value=[])
    agent = ReActAgent(llm=llm, name=name)
    return agent


def _make_stub_agent(name: str, output: str, success: bool = True, tokens: int = 10) -> BaseAgent:
    """Create a minimal stub agent for orchestration tests."""
    agent = AsyncMock(spec=BaseAgent)
    agent.name = name
    agent.run = AsyncMock(
        return_value=AgentResponse(
            output=output,
            steps=[AgentStep(step_number=1)],
            success=success,
            error=None if success else "agent_error",
            total_tokens=tokens,
        ),
    )
    return agent


def _make_llm(content: str = "") -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content=content,
            usage=TokenUsage(input_tokens=5, output_tokens=3),
        ),
    )
    return llm


# ───────── AgentPipeline ────────────────────────────────────────────────

class TestAgentPipeline:
    def test_empty_pipeline_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AgentPipeline(agents=[])

    @pytest.mark.asyncio
    async def test_single_agent(self) -> None:
        agent = _make_stub_agent("a", "result-A")
        pipe = AgentPipeline(agents=[agent])
        result = await pipe.run("task")
        assert result.success
        assert result.output == "result-A"

    @pytest.mark.asyncio
    async def test_two_agent_chain(self) -> None:
        a1 = _make_stub_agent("first", "intermediate")
        a2 = _make_stub_agent("second", "final")
        pipe = AgentPipeline(agents=[a1, a2])
        result = await pipe.run("start")

        assert result.success
        assert result.output == "final"
        # Second agent should receive first agent's output
        a2.run.assert_awaited_once_with("intermediate")

    @pytest.mark.asyncio
    async def test_three_agent_chain(self) -> None:
        agents = [
            _make_stub_agent("a", "out-a"),
            _make_stub_agent("b", "out-b"),
            _make_stub_agent("c", "out-c"),
        ]
        pipe = AgentPipeline(agents=agents)
        result = await pipe.run("go")
        assert result.success
        assert result.output == "out-c"
        assert result.metadata["stages_completed"] == 3

    @pytest.mark.asyncio
    async def test_failure_stops_pipeline(self) -> None:
        a1 = _make_stub_agent("ok", "fine")
        a2 = _make_stub_agent("broken", "error output", success=False)
        a3 = _make_stub_agent("never", "should not run")
        pipe = AgentPipeline(agents=[a1, a2, a3])
        result = await pipe.run("go")

        assert not result.success
        assert "stage 1" in result.error
        assert "broken" in result.error
        a3.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_token_accumulation(self) -> None:
        a1 = _make_stub_agent("a", "x", tokens=20)
        a2 = _make_stub_agent("b", "y", tokens=30)
        pipe = AgentPipeline(agents=[a1, a2])
        result = await pipe.run("go")
        assert result.total_tokens == 50

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self) -> None:
        agent = _make_stub_agent("a", "ok")
        pipe = AgentPipeline(agents=[agent])
        result = await pipe.run("go")
        assert result.execution_time > 0

    def test_repr(self) -> None:
        agent = _make_stub_agent("alpha", "x")
        pipe = AgentPipeline(agents=[agent])
        assert "alpha" in repr(pipe)

    @pytest.mark.asyncio
    async def test_stage_outputs_in_metadata(self) -> None:
        a1 = _make_stub_agent("a", "out-a")
        a2 = _make_stub_agent("b", "out-b")
        pipe = AgentPipeline(agents=[a1, a2])
        result = await pipe.run("start")
        assert len(result.metadata["stage_outputs"]) == 2


# ───────── TaskRouter ───────────────────────────────────────────────────

class TestTaskRouter:
    def test_empty_routes_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TaskRouter(llm=AsyncMock(), routes={})

    @pytest.mark.asyncio
    async def test_routes_to_correct_agent(self) -> None:
        coder = _make_stub_agent("coder", "code output")
        researcher = _make_stub_agent("researcher", "research output")
        llm = _make_llm("coding")

        router = TaskRouter(llm=llm, routes={"coding": coder, "research": researcher})
        result = await router.route("Write a function")

        assert result.success
        assert result.output == "code output"
        assert result.metadata["routed_to"] == "coding"

    @pytest.mark.asyncio
    async def test_default_agent_fallback(self) -> None:
        coder = _make_stub_agent("coder", "code")
        fallback = _make_stub_agent("general", "fallback result")
        llm = _make_llm("unknown_category")

        router = TaskRouter(
            llm=llm,
            routes={"coding": coder},
            default_agent=fallback,
        )
        result = await router.route("Do something vague")

        assert result.success
        assert result.output == "fallback result"

    @pytest.mark.asyncio
    async def test_no_default_returns_error(self) -> None:
        coder = _make_stub_agent("coder", "code")
        llm = _make_llm("unknown_category")

        router = TaskRouter(llm=llm, routes={"coding": coder})
        result = await router.route("Do something vague")

        assert not result.success
        assert "unroutable_task" in result.error

    @pytest.mark.asyncio
    async def test_classification_uses_low_temperature(self) -> None:
        agent = _make_stub_agent("a", "ok")
        llm = _make_llm("a")
        router = TaskRouter(llm=llm, routes={"a": agent})
        await router.route("task")
        call_kwargs = llm.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_token_accumulation(self) -> None:
        agent = _make_stub_agent("a", "ok", tokens=20)
        llm = _make_llm("a")
        router = TaskRouter(llm=llm, routes={"a": agent})
        result = await router.route("task")
        # Router classification tokens + agent tokens
        assert result.total_tokens == 20 + 8  # 5+3 from router LLM

    def test_repr(self) -> None:
        router = TaskRouter(
            llm=AsyncMock(),
            routes={"code": _make_stub_agent("c", "x")},
        )
        assert "code" in repr(router)


# ───────── AgentSupervisor ──────────────────────────────────────────────

class TestAgentSupervisor:
    def test_empty_agents_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AgentSupervisor(llm=AsyncMock(), agents={})

    @pytest.mark.asyncio
    async def test_single_delegation_then_done(self) -> None:
        worker = _make_stub_agent("worker", "42")
        llm = AsyncMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="DELEGATE worker: compute the answer",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                ),
                LLMResponse(
                    content="DONE: The answer is 42",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                ),
            ],
        )

        sup = AgentSupervisor(llm=llm, agents={"worker": worker})
        result = await sup.run("What is the answer?")

        assert result.success
        assert "42" in result.output
        assert result.metadata["rounds"] == 2
        assert result.metadata["delegations"] == 1

    @pytest.mark.asyncio
    async def test_done_signal_stops_immediately(self) -> None:
        worker = _make_stub_agent("worker", "unused")
        llm = _make_llm("DONE: Already know the answer.")

        sup = AgentSupervisor(llm=llm, agents={"worker": worker})
        result = await sup.run("task")

        assert result.success
        assert "Already know" in result.output
        worker.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_max_rounds_exceeded(self) -> None:
        worker = _make_stub_agent("worker", "partial")
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="DELEGATE worker: keep going",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
        )

        sup = AgentSupervisor(llm=llm, agents={"worker": worker}, max_rounds=2)
        result = await sup.run("big task")

        assert not result.success
        assert result.error == "max_rounds_exceeded"

    @pytest.mark.asyncio
    async def test_unknown_agent_handled(self) -> None:
        worker = _make_stub_agent("worker", "ok")
        llm = AsyncMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="DELEGATE nonexistent: do stuff",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                ),
                LLMResponse(
                    content="DONE: Could not complete",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                ),
            ],
        )

        sup = AgentSupervisor(llm=llm, agents={"worker": worker})
        result = await sup.run("task")

        assert result.success
        worker.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_token_accumulation(self) -> None:
        worker = _make_stub_agent("worker", "ok", tokens=20)
        llm = AsyncMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="DELEGATE worker: subtask",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                ),
                LLMResponse(
                    content="DONE: result",
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                ),
            ],
        )
        sup = AgentSupervisor(llm=llm, agents={"worker": worker})
        result = await sup.run("task")
        # 8 (round1) + 20 (worker) + 8 (round2)
        assert result.total_tokens == 36

    @pytest.mark.asyncio
    async def test_metadata_tracks_rounds(self) -> None:
        worker = _make_stub_agent("w", "ok")
        llm = AsyncMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(content="DELEGATE w: a", usage=TokenUsage()),
                LLMResponse(content="DELEGATE w: b", usage=TokenUsage()),
                LLMResponse(content="DONE: final", usage=TokenUsage()),
            ],
        )
        sup = AgentSupervisor(llm=llm, agents={"w": worker})
        result = await sup.run("task")
        assert result.metadata["rounds"] == 3
        assert result.metadata["delegations"] == 2

    @pytest.mark.asyncio
    async def test_multiple_agents(self) -> None:
        coder = _make_stub_agent("coder", "code done")
        reviewer = _make_stub_agent("reviewer", "looks good")
        llm = AsyncMock()
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(content="DELEGATE coder: write code", usage=TokenUsage()),
                LLMResponse(content="DELEGATE reviewer: review code", usage=TokenUsage()),
                LLMResponse(content="DONE: All done", usage=TokenUsage()),
            ],
        )
        sup = AgentSupervisor(llm=llm, agents={"coder": coder, "reviewer": reviewer})
        result = await sup.run("build and review")
        assert result.success
        coder.run.assert_awaited_once()
        reviewer.run.assert_awaited_once()

    def test_repr(self) -> None:
        sup = AgentSupervisor(
            llm=AsyncMock(),
            agents={"alpha": _make_stub_agent("a", "x")},
        )
        assert "alpha" in repr(sup)
