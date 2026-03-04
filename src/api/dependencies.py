"""Shared dependencies for API routes — agent creation, tool registration, LLM setup."""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.agents.coder import CoderAgent
from src.agents.planner import PlannerAgent
from src.agents.react import ReActAgent
from src.agents.researcher import ResearchAgent
from src.agents.reviewer import ReviewerAgent
from src.llm.provider import BaseLLMProvider, ProviderFactory
from src.memory.base import BaseMemory
from src.memory.conversation import ConversationMemory
from src.memory.summary import SummaryMemory
from src.tools.registry import ToolRegistry
from src.utils.logger import get_logger

log = get_logger(__name__)

_AGENT_TYPES: dict[str, type[ReActAgent]] = {
    "react": ReActAgent,
    "planner": PlannerAgent,
    "researcher": ResearchAgent,
    "coder": CoderAgent,
    "reviewer": ReviewerAgent,
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "react": "General-purpose ReAct agent with step-by-step reasoning",
    "planner": "Decomposes complex tasks into structured plans",
    "researcher": "Gathers and synthesises information from multiple sources",
    "coder": "Writes, debugs, and verifies code",
    "reviewer": "Reviews and critiques content for quality and correctness",
}


def get_llm_provider() -> BaseLLMProvider:
    """Create the default LLM provider from settings."""
    return ProviderFactory.create()


def get_tool_registry(llm: BaseLLMProvider | None = None) -> ToolRegistry:
    """Return the global tool registry, registering built-in tools on first call."""
    registry = ToolRegistry.get_instance()
    if registry.list_tools():
        return registry

    _register_builtin_tools(registry, llm)
    return registry


def _register_builtin_tools(
    registry: ToolRegistry, llm: BaseLLMProvider | None = None,
) -> None:
    """Register all built-in tools that are available."""
    from src.tools.calculator import CalculatorTool
    from src.tools.code_executor import CodeExecutorTool
    from src.tools.file_ops import FileReadTool, FileWriteTool
    from src.tools.web_search import WebSearchTool

    for tool_cls in [CalculatorTool, FileReadTool, FileWriteTool, CodeExecutorTool, WebSearchTool]:
        try:
            registry.register(tool_cls())
        except (ValueError, Exception) as exc:  # noqa: BLE001
            log.debug("tool_register_skip", tool=tool_cls.__name__, reason=str(exc))

    # Async tools
    try:
        from src.tools.api_caller import APICallerTool
        registry.register(APICallerTool())
    except (ValueError, Exception) as exc:  # noqa: BLE001
        log.debug("tool_register_skip", tool="APICallerTool", reason=str(exc))

    # LLM-dependent tools
    if llm:
        try:
            from src.tools.text_tools import ExtractTool, SummarizeTool
            registry.register(SummarizeTool(llm=llm))
            registry.register(ExtractTool(llm=llm))
        except (ValueError, Exception) as exc:  # noqa: BLE001
            log.debug("tool_register_skip", tool="text_tools", reason=str(exc))


def get_memory(memory_type: str, llm: BaseLLMProvider | None = None) -> BaseMemory:
    """Create a memory backend from its type name."""
    if memory_type == "summary" and llm:
        return SummaryMemory(llm=llm)
    return ConversationMemory()


def create_agent(
    agent_type: str,
    llm: BaseLLMProvider,
    tools: ToolRegistry | None = None,
    memory: BaseMemory | None = None,
    max_steps: int = 10,
    **kwargs: Any,
) -> BaseAgent:
    """Instantiate an agent by type name."""
    cls = _AGENT_TYPES.get(agent_type, ReActAgent)
    return cls(llm=llm, tools=tools, memory=memory, max_steps=max_steps, **kwargs)
