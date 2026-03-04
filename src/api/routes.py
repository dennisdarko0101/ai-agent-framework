"""REST API routes for the agent framework."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.base import AgentResponse
from src.api.dependencies import (
    AGENT_DESCRIPTIONS,
    create_agent,
    get_llm_provider,
    get_memory,
    get_tool_registry,
)
from src.api.schemas import (
    ActionInfo,
    AgentInfo,
    AgentListResponse,
    CodeRequest,
    HealthResponse,
    PipelineRequest,
    ResearchRequest,
    StepInfo,
    TaskRequest,
    TaskResponse,
    ToolInfo,
    ToolListResponse,
)
from src.orchestration.pipeline import AgentPipeline

router = APIRouter(prefix="/api/v1")
health_router = APIRouter()


# ───────── Helpers ───────────────────────────────────────────────────────

def _to_task_response(resp: AgentResponse) -> TaskResponse:
    """Convert an AgentResponse to the API TaskResponse model."""
    steps = []
    for s in resp.steps:
        actions = [
            ActionInfo(
                tool_name=a.tool_name,
                arguments=a.arguments,
                result=a.result_output if a.success else (a.result_error or ""),
                success=a.success,
            )
            for a in s.actions
        ]
        steps.append(StepInfo(
            step_number=s.step_number,
            reasoning=s.reasoning,
            actions=actions,
            observation=s.observation,
        ))
    return TaskResponse(
        result=resp.output,
        steps=steps,
        success=resp.success,
        error=resp.error,
        metadata={
            "total_tokens": resp.total_tokens,
            "execution_time": round(resp.execution_time, 3),
            **resp.metadata,
        },
    )


def _setup_agent(
    agent_type: str,
    memory_type: str = "conversation",
    max_steps: int = 10,
    **kwargs: Any,
) -> Any:
    """Wire up LLM, tools, memory, and create an agent."""
    llm = get_llm_provider()
    tools = get_tool_registry(llm)
    memory = get_memory(memory_type, llm)
    return create_agent(agent_type, llm, tools, memory, max_steps, **kwargs)


# ───────── Health ────────────────────────────────────────────────────────

@health_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry.get_instance()
    return HealthResponse(
        agents=len(AGENT_DESCRIPTIONS),
        tools=len(registry.list_tools()),
    )


# ───────── Task execution ───────────────────────────────────────────────

@router.post("/run", response_model=TaskResponse)
async def run_task(request: TaskRequest) -> TaskResponse:
    """Execute a task with the specified agent."""
    try:
        agent = _setup_agent(request.agent_type, request.memory_type, request.max_steps)
        response = await agent.run(request.task)
        return _to_task_response(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/plan", response_model=TaskResponse)
async def plan_task(request: TaskRequest) -> TaskResponse:
    """Create a plan for a complex task."""
    try:
        agent = _setup_agent("planner", request.memory_type, request.max_steps)
        response = await agent.create_plan(request.task)
        return _to_task_response(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/research", response_model=TaskResponse)
async def research_topic(request: ResearchRequest) -> TaskResponse:
    """Research a topic."""
    try:
        agent = _setup_agent("researcher")
        response = await agent.research(request.topic, depth=request.depth)
        return _to_task_response(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/code", response_model=TaskResponse)
async def generate_code(request: CodeRequest) -> TaskResponse:
    """Generate code from a specification."""
    try:
        agent = _setup_agent("coder")
        response = await agent.write_code(request.task, language=request.language)
        return _to_task_response(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/pipeline", response_model=TaskResponse)
async def run_pipeline(request: PipelineRequest) -> TaskResponse:
    """Run a multi-agent pipeline."""
    try:
        llm = get_llm_provider()
        tools = get_tool_registry(llm)
        memory = get_memory(request.memory_type, llm)
        agents = [
            create_agent(stage.agent_type, llm, tools, memory)
            for stage in request.stages
        ]
        pipeline = AgentPipeline(agents=agents)
        response = await pipeline.run(request.task)
        return _to_task_response(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ───────── Discovery ─────────────────────────────────────────────────────

@router.get("/agents", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    """List available agent types."""
    return AgentListResponse(
        agents=[
            AgentInfo(name=name, description=desc, agent_type=name)
            for name, desc in AGENT_DESCRIPTIONS.items()
        ],
    )


@router.get("/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """List available tools with their schemas."""
    registry = get_tool_registry()
    return ToolListResponse(
        tools=[
            ToolInfo(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in registry.list_tools()
        ],
    )
