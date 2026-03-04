"""Integration tests for WebSocket streaming — 8 tests."""

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
            content="WebSocket answer",
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


class TestWebSocket:
    def test_basic_task(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": "Hello"})
            events = []
            # Collect all events until connection closes or we get a result
            try:
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["event_type"] in ("result", "error"):
                        break
            except Exception:
                pass
            assert len(events) >= 1
            result_events = [e for e in events if e["event_type"] == "result"]
            assert len(result_events) >= 1

    def test_event_has_required_fields(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": "test"})
            event = ws.receive_json()
            assert "event_type" in event
            assert "data" in event
            assert "timestamp" in event
            assert "step_number" in event

    def test_empty_task_returns_error(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": ""})
            event = ws.receive_json()
            assert event["event_type"] == "error"

    def test_custom_agent_type(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": "plan this", "agent_type": "planner"})
            events = []
            try:
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["event_type"] in ("result", "error"):
                        break
            except Exception:
                pass
            assert any(e["event_type"] == "result" for e in events)

    def test_event_timestamp_is_positive(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": "hi"})
            event = ws.receive_json()
            assert event["timestamp"] > 0

    def test_result_contains_answer(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": "answer me"})
            events = []
            try:
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["event_type"] in ("result", "error"):
                        break
            except Exception:
                pass
            result_events = [e for e in events if e["event_type"] == "result"]
            assert any("WebSocket answer" in e["data"] for e in result_events)

    def test_max_steps_parameter(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"task": "test", "max_steps": 3})
            event = ws.receive_json()
            assert event["event_type"] in ("thought", "result", "error")

    def test_missing_task_key(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"not_task": "hello"})
            event = ws.receive_json()
            assert event["event_type"] == "error"
