"""Base tool abstractions — extend BaseTool to add new capabilities to agents."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.llm.function_calling import ToolSchema
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Outcome of a single tool execution."""

    output: str = ""
    error: str | None = None
    execution_time: float = 0.0
    success: bool = True

    def __post_init__(self) -> None:
        if self.error:
            self.success = False


# ---------------------------------------------------------------------------
# Sync base
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """Abstract base for all tools.

    Subclass this, set *name*, *description*, and *parameters*, then implement
    ``execute()``.  The framework handles schema generation and registry
    integration automatically.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}  # noqa: RUF012 — mutable default is fine on ABC

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool and return a result."""
        ...

    def to_schema(self) -> ToolSchema:
        """Convert to a provider-agnostic ToolSchema for LLM function calling."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters or {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    def _timed_execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute with automatic timing — convenience for subclasses."""
        start = time.monotonic()
        try:
            result = self.execute(arguments)
            if result.execution_time == 0.0:
                result.execution_time = time.monotonic() - start
            return result
        except Exception as exc:
            return ToolResult(
                error=str(exc),
                execution_time=time.monotonic() - start,
            )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------

class AsyncTool(BaseTool):
    """Base class for tools that require async I/O (HTTP calls, LLM, etc.).

    Implement ``execute_async`` instead of ``execute``.
    """

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError(
            f"{type(self).__name__} is async — use execute_async() or "
            "the registry's async execute method."
        )

    @abstractmethod
    async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool asynchronously and return a result."""
        ...
