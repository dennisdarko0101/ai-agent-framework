"""Tests for specialised agents — 16+ tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.coder import CoderAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearchAgent
from src.agents.reviewer import ReviewerAgent
from src.llm.provider import LLMResponse, TokenUsage


def _make_llm(content: str = "done") -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content=content,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
    )
    llm.build_assistant_message = MagicMock(return_value={"role": "assistant", "content": ""})
    llm.build_tool_result_messages = MagicMock(return_value=[])
    return llm


# ───────── PlannerAgent ─────────────────────────────────────────────────

class TestPlannerAgent:
    def test_default_name(self) -> None:
        agent = PlannerAgent(llm=AsyncMock())
        assert agent.name == "planner"

    def test_has_default_system_prompt(self) -> None:
        agent = PlannerAgent(llm=AsyncMock())
        assert "planning" in agent.system_prompt.lower()

    def test_custom_system_prompt(self) -> None:
        agent = PlannerAgent(llm=AsyncMock(), system_prompt="Custom plan prompt")
        assert agent.system_prompt == "Custom plan prompt"

    @pytest.mark.asyncio
    async def test_run(self) -> None:
        llm = _make_llm("Step 1: Do X\nStep 2: Do Y")
        agent = PlannerAgent(llm=llm)
        result = await agent.run("Build a website")
        assert result.success
        assert "Step 1" in result.output

    @pytest.mark.asyncio
    async def test_create_plan(self) -> None:
        llm = _make_llm("1. Research\n2. Design\n3. Implement")
        agent = PlannerAgent(llm=llm)
        result = await agent.create_plan("Build an API")
        assert result.success
        # create_plan wraps the task in a planning prompt
        call_kwargs = llm.generate.call_args[1]
        assert "plan" in call_kwargs["messages"][-1]["content"].lower()


# ───────── ResearchAgent ────────────────────────────────────────────────

class TestResearchAgent:
    def test_default_name(self) -> None:
        agent = ResearchAgent(llm=AsyncMock())
        assert agent.name == "researcher"

    def test_has_default_system_prompt(self) -> None:
        agent = ResearchAgent(llm=AsyncMock())
        assert "research" in agent.system_prompt.lower()

    def test_default_max_steps_higher(self) -> None:
        agent = ResearchAgent(llm=AsyncMock())
        assert agent.max_steps == 15

    @pytest.mark.asyncio
    async def test_run(self) -> None:
        llm = _make_llm("Research findings: ...")
        agent = ResearchAgent(llm=llm)
        result = await agent.run("Tell me about AI")
        assert result.success

    @pytest.mark.asyncio
    async def test_research_method(self) -> None:
        llm = _make_llm("Key findings: X, Y, Z")
        agent = ResearchAgent(llm=llm)
        result = await agent.research("quantum computing", depth="deep")
        assert result.success
        call_kwargs = llm.generate.call_args[1]
        msg_content = call_kwargs["messages"][-1]["content"]
        assert "quantum computing" in msg_content
        assert "deep" in msg_content


# ───────── CoderAgent ───────────────────────────────────────────────────

class TestCoderAgent:
    def test_default_name(self) -> None:
        agent = CoderAgent(llm=AsyncMock())
        assert agent.name == "coder"

    def test_has_default_system_prompt(self) -> None:
        agent = CoderAgent(llm=AsyncMock())
        assert "coding" in agent.system_prompt.lower() or "code" in agent.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_write_code(self) -> None:
        llm = _make_llm("def hello(): return 'world'")
        agent = CoderAgent(llm=llm)
        result = await agent.write_code("A hello function", language="python")
        assert result.success
        call_kwargs = llm.generate.call_args[1]
        msg = call_kwargs["messages"][-1]["content"]
        assert "python" in msg.lower()

    @pytest.mark.asyncio
    async def test_debug_code(self) -> None:
        llm = _make_llm("Fixed: changed == to is")
        agent = CoderAgent(llm=llm)
        result = await agent.debug_code("x == None", "Use 'is' for None checks")
        assert result.success
        call_kwargs = llm.generate.call_args[1]
        msg = call_kwargs["messages"][-1]["content"]
        assert "x == None" in msg


# ───────── ReviewerAgent ────────────────────────────────────────────────

class TestReviewerAgent:
    def test_default_name(self) -> None:
        agent = ReviewerAgent(llm=AsyncMock())
        assert agent.name == "reviewer"

    def test_has_default_system_prompt(self) -> None:
        agent = ReviewerAgent(llm=AsyncMock())
        assert "review" in agent.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_review(self) -> None:
        llm = _make_llm("Quality: Good. Feedback: Consider adding tests.")
        agent = ReviewerAgent(llm=llm)
        result = await agent.review("Here is some code...")
        assert result.success

    @pytest.mark.asyncio
    async def test_review_with_criteria(self) -> None:
        llm = _make_llm("Meets all criteria.")
        agent = ReviewerAgent(llm=llm)
        result = await agent.review("content", criteria="Must be < 100 words")
        assert result.success
        call_kwargs = llm.generate.call_args[1]
        msg = call_kwargs["messages"][-1]["content"]
        assert "100 words" in msg
