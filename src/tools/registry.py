"""Tool registry — singleton catalogue of available tools with auto-discovery."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from src.llm.function_calling import ToolSchema
from src.tools.base import AsyncTool, BaseTool, ToolResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class ToolRegistry:
    """Central registry of tools available to agents.

    Provides a singleton via ``get_instance()`` but can also be instantiated
    directly for testing or multi-tenant isolation.
    """

    _instance: ToolRegistry | None = None

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── Singleton access ────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """Return the global registry, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the global registry (mainly useful in tests)."""
        cls._instance = None

    # ── Core API ────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry.

        Raises:
            ValueError: on empty name or duplicate registration.
            TypeError: if *tool* is not a BaseTool instance.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, got {type(tool).__name__}")
        if not tool.name:
            raise ValueError("Tool must have a non-empty 'name'")
        if not tool.description:
            raise ValueError(f"Tool {tool.name!r} must have a non-empty 'description'")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")

        self._tools[tool.name] = tool
        log.info("tool_registered", tool=tool.name)

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name.

        Raises:
            KeyError: if the tool is not registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool not found: {name!r}") from None

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[ToolSchema]:
        """Generate ToolSchema objects for every registered tool.

        These can be passed straight to an LLM provider's ``generate()`` call.
        """
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Look up a tool and execute it, handling sync/async transparently.

        Returns a ``ToolResult`` — errors are captured, never raised.
        """
        try:
            tool = self.get(tool_name)
        except KeyError as exc:
            return ToolResult(error=str(exc))

        start = time.monotonic()
        try:
            if isinstance(tool, AsyncTool):
                result = await tool.execute_async(arguments)
            else:
                result = tool.execute(arguments)
            if result.execution_time == 0.0:
                result.execution_time = time.monotonic() - start
            return result
        except Exception as exc:
            return ToolResult(
                error=f"Execution error in {tool_name!r}: {exc}",
                execution_time=time.monotonic() - start,
            )

    # ── Auto-discovery ──────────────────────────────────────────────────

    @staticmethod
    def discover_tools(directory: str | Path) -> list[type[BaseTool]]:
        """Scan *directory* for Python files and return all BaseTool subclasses found.

        Does **not** register them — call ``register(cls())`` on each class you
        want to activate.  Classes that require constructor arguments will need
        to be instantiated manually.

        Raises:
            FileNotFoundError: if *directory* does not exist.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        tool_classes: list[type[BaseTool]] = []
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"_discovered_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            except Exception:
                log.warning("discover_import_failed", path=str(path))
                continue

            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseTool)
                    and obj not in (BaseTool, AsyncTool)
                ):
                    tool_classes.append(obj)

        log.info("discover_tools", directory=str(directory), found=len(tool_classes))
        return tool_classes
