"""Integration tests for REST API endpoints — 15+ tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.llm.provider import LLMResponse, TokenUsage
from src.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    ToolRegistry.reset()


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Test answer",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
    )
    llm.build_assistant_message = MagicMock(return_value={"role": "assistant", "content": ""})
    llm.build_tool_result_messages = MagicMock(return_value=[])
    return llm


@pytest.fixture
def client(mock_llm: AsyncMock) -> TestClient:
    with patch("src.api.dependencies.ProviderFactory") as factory_mock:
        factory_mock.create.return_value = mock_llm
        yield TestClient(app)


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_includes_counts(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "agents" in data
        assert "tools" in data


class TestListEndpoints:
    def test_list_agents(self, client: TestClient) -> None:
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) >= 5
        names = {a["name"] for a in data["agents"]}
        assert "react" in names
        assert "planner" in names

    def test_list_tools(self, client: TestClient) -> None:
        resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["tools"], list)


class TestRunEndpoint:
    def test_run_returns_result(self, client: TestClient) -> None:
        resp = client.post("/api/v1/run", json={"task": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["result"] == "Test answer"

    def test_run_with_agent_type(self, client: TestClient) -> None:
        resp = client.post("/api/v1/run", json={"task": "Plan this", "agent_type": "planner"})
        assert resp.status_code == 200

    def test_run_includes_metadata(self, client: TestClient) -> None:
        resp = client.post("/api/v1/run", json={"task": "Hello"})
        data = resp.json()
        assert "total_tokens" in data["metadata"]
        assert "execution_time" in data["metadata"]

    def test_run_empty_task_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/run", json={"task": ""})
        assert resp.status_code == 422  # validation error

    def test_run_invalid_max_steps(self, client: TestClient) -> None:
        resp = client.post("/api/v1/run", json={"task": "x", "max_steps": 0})
        assert resp.status_code == 422


class TestSpecializedEndpoints:
    def test_plan_endpoint(self, client: TestClient) -> None:
        resp = client.post("/api/v1/plan", json={"task": "Build an API"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_research_endpoint(self, client: TestClient) -> None:
        resp = client.post("/api/v1/research", json={"topic": "AI safety"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_code_endpoint(self, client: TestClient) -> None:
        resp = client.post("/api/v1/code", json={"task": "sort function"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_code_with_language(self, client: TestClient) -> None:
        resp = client.post("/api/v1/code", json={"task": "hello", "language": "javascript"})
        assert resp.status_code == 200


class TestPipelineEndpoint:
    def test_pipeline_runs(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pipeline", json={
            "task": "Do something",
            "stages": [
                {"agent_type": "planner"},
                {"agent_type": "coder"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_pipeline_empty_stages_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/pipeline", json={"task": "x", "stages": []})
        assert resp.status_code == 422
