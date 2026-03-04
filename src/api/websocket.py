"""WebSocket endpoint for real-time agent streaming."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.dependencies import create_agent, get_llm_provider, get_memory, get_tool_registry
from src.utils.logger import get_logger

log = get_logger(__name__)

ws_router = APIRouter()


@ws_router.websocket("/ws/run")
async def ws_run(websocket: WebSocket) -> None:
    """Stream agent execution in real-time.

    Client sends a JSON message::

        {"task": "...", "agent_type": "react", "max_steps": 10}

    Server streams JSON ``StreamEvent`` objects::

        {"event_type": "thought", "data": "...", "step_number": 1, "timestamp": ...}
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        task = data.get("task", "")
        if not task:
            await _send_event(websocket, "error", "Empty task", 0)
            return

        agent_type = data.get("agent_type", "react")
        max_steps = data.get("max_steps", 10)
        memory_type = data.get("memory_type", "conversation")

        llm = get_llm_provider()
        tools = get_tool_registry(llm)
        memory = get_memory(memory_type, llm)
        agent = create_agent(agent_type, llm, tools, memory, max_steps)

        async def on_event(event_type: str, event_data: dict[str, Any]) -> None:
            payload = json.dumps(event_data) if isinstance(event_data, dict) else str(event_data)
            step = event_data.get("step", 0) if isinstance(event_data, dict) else 0
            await _send_event(websocket, event_type, payload, step)

        result = await agent.run(task, event_callback=on_event)

        # Send final result event
        await _send_event(
            websocket,
            "result",
            result.output,
            len(result.steps),
        )

    except WebSocketDisconnect:
        log.info("ws_disconnected")
    except Exception as exc:
        log.warning("ws_error", error=str(exc))
        import contextlib
        with contextlib.suppress(Exception):
            await _send_event(websocket, "error", str(exc), 0)


async def _send_event(
    websocket: WebSocket,
    event_type: str,
    data: str,
    step_number: int,
) -> None:
    await websocket.send_json({
        "event_type": event_type,
        "data": data,
        "step_number": step_number,
        "timestamp": time.time(),
    })
