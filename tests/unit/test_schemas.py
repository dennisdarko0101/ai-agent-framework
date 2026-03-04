"""Tests for API request/response schemas — 15 tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ActionInfo,
    AgentInfo,
    AgentListResponse,
    CodeRequest,
    HealthResponse,
    PipelineRequest,
    PipelineStage,
    ResearchRequest,
    StepInfo,
    StreamEvent,
    TaskRequest,
    TaskResponse,
    ToolInfo,
    ToolListResponse,
)


class TestTaskRequest:
    def test_minimal(self) -> None:
        req = TaskRequest(task="hello")
        assert req.task == "hello"
        assert req.agent_type == "react"
        assert req.max_steps == 10

    def test_all_fields(self) -> None:
        req = TaskRequest(
            task="do X",
            agent_type="planner",
            tools=["calc"],
            memory_type="summary",
            max_steps=20,
        )
        assert req.agent_type == "planner"
        assert req.tools == ["calc"]

    def test_empty_task_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(task="")

    def test_max_steps_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(task="x", max_steps=0)
        with pytest.raises(ValidationError):
            TaskRequest(task="x", max_steps=100)


class TestPipelineRequest:
    def test_valid(self) -> None:
        req = PipelineRequest(
            task="go",
            stages=[PipelineStage(agent_type="planner"), PipelineStage(agent_type="coder")],
        )
        assert len(req.stages) == 2

    def test_empty_stages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineRequest(task="go", stages=[])


class TestResearchRequest:
    def test_defaults(self) -> None:
        req = ResearchRequest(topic="AI")
        assert req.depth == "standard"


class TestCodeRequest:
    def test_defaults(self) -> None:
        req = CodeRequest(task="sort fn")
        assert req.language == "python"


class TestTaskResponse:
    def test_minimal(self) -> None:
        resp = TaskResponse(result="done")
        assert resp.success is True
        assert resp.steps == []

    def test_with_steps(self) -> None:
        resp = TaskResponse(
            result="ok",
            steps=[StepInfo(step_number=1, reasoning="think")],
        )
        assert len(resp.steps) == 1


class TestStreamEvent:
    def test_defaults(self) -> None:
        event = StreamEvent(event_type="thought", data="thinking...")
        assert event.event_type == "thought"
        assert event.timestamp > 0


class TestListResponses:
    def test_agent_list(self) -> None:
        resp = AgentListResponse(agents=[
            AgentInfo(name="react", description="General agent", agent_type="react"),
        ])
        assert len(resp.agents) == 1

    def test_tool_list(self) -> None:
        resp = ToolListResponse(tools=[
            ToolInfo(name="calc", description="Calculator"),
        ])
        assert len(resp.tools) == 1

    def test_health(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "0.1.0"


class TestActionInfo:
    def test_defaults(self) -> None:
        action = ActionInfo(tool_name="calc")
        assert action.success is True
        assert action.result == ""
