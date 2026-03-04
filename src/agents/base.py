"""Base agent abstractions — extend BaseAgent to build new agent types."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory
from src.tools.registry import ToolRegistry
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """A single tool invocation within a reasoning step."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""
    result_output: str = ""
    result_error: str | None = None
    success: bool = True


@dataclass
class AgentStep:
    """One think → act → observe cycle."""

    step_number: int
    reasoning: str = ""
    actions: list[AgentAction] = field(default_factory=list)
    observation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """Final output from an agent run."""

    output: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    total_tokens: int = 0
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.error and self.success:
            self.success = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract agent — subclass and implement ``run()``."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str = "",
        max_steps: int = 10,
        name: str = "agent",
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.name = name

    @abstractmethod
    async def run(
        self,
        task: str,
        event_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AgentResponse:
        """Execute a task and return the response.

        Args:
            task: The task description.
            event_callback: Optional async callback for streaming events.
                Called with ``(event_type, data)`` where *event_type* is one of
                ``thought``, ``action``, ``observation``, ``result``, ``error``.
        """
        ...

    def _build_system_prompt(self) -> str:
        """Return the system prompt for the LLM call."""
        return self.system_prompt

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
