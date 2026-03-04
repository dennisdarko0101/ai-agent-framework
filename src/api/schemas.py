"""Pydantic models for API requests and responses."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

# ───────── Requests ─────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, description="Task description")
    agent_type: str = Field(default="react", description="Agent type: react, planner, researcher, coder, reviewer")
    tools: list[str] | None = Field(default=None, description="Tool names to enable (None = all)")
    memory_type: str = Field(default="conversation", description="Memory backend")
    max_steps: int = Field(default=10, ge=1, le=50, description="Max reasoning steps")


class PipelineStage(BaseModel):
    agent_type: str = Field(default="react", description="Agent type for this stage")


class PipelineRequest(BaseModel):
    task: str = Field(..., min_length=1)
    stages: list[PipelineStage] = Field(..., min_length=1, description="Pipeline stages")
    memory_type: str = Field(default="conversation")


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    depth: str = Field(default="standard", description="Research depth: standard or deep")


class CodeRequest(BaseModel):
    task: str = Field(..., min_length=1)
    language: str = Field(default="python")


# ───────── Responses ────────────────────────────────────────────────────

class ActionInfo(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}
    result: str = ""
    success: bool = True


class StepInfo(BaseModel):
    step_number: int
    reasoning: str = ""
    actions: list[ActionInfo] = []
    observation: str = ""


class TaskResponse(BaseModel):
    result: str
    steps: list[StepInfo] = []
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = {}


class StreamEvent(BaseModel):
    event_type: str = Field(..., description="thought, action, observation, result, error")
    data: str = ""
    step_number: int = 0
    timestamp: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    agents: int = 0
    tools: int = 0


class AgentInfo(BaseModel):
    name: str
    description: str
    agent_type: str


class AgentListResponse(BaseModel):
    agents: list[AgentInfo]


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = {}


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]
