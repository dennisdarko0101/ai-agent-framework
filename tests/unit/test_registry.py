"""Tests for ToolRegistry — 15 tests covering registration, lookup, execution, and discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.tools.base import AsyncTool, BaseTool, ToolResult
from src.tools.registry import ToolRegistry

# ───────── Helpers ────────────────────────────────────────────────────

class _AddTool(BaseTool):
    name = "add"
    description = "Add two numbers"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=str(arguments["a"] + arguments["b"]))


class _AsyncEchoTool(AsyncTool):
    name = "echo"
    description = "Echo input"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=arguments["msg"])


class _NoNameTool(BaseTool):
    name = ""
    description = "Broken"

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult()


class _NoDescTool(BaseTool):
    name = "nodesc"
    description = ""

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult()


# ───────── Tests ──────────────────────────────────────────────────────

class TestToolRegistry:
    def setup_method(self) -> None:
        ToolRegistry.reset()
        self.registry = ToolRegistry()

    # -- Registration --

    def test_register_and_get(self) -> None:
        self.registry.register(_AddTool())
        tool = self.registry.get("add")
        assert tool.name == "add"

    def test_get_nonexistent(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            self.registry.get("nope")

    def test_duplicate_name_rejected(self) -> None:
        self.registry.register(_AddTool())
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(_AddTool())

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'name'"):
            self.registry.register(_NoNameTool())

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'description'"):
            self.registry.register(_NoDescTool())

    def test_non_tool_rejected(self) -> None:
        with pytest.raises(TypeError, match="Expected BaseTool"):
            self.registry.register("not a tool")  # type: ignore[arg-type]

    # -- Listing & schemas --

    def test_list_tools(self) -> None:
        self.registry.register(_AddTool())
        self.registry.register(_AsyncEchoTool())
        tools = self.registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"add", "echo"}

    def test_get_schemas(self) -> None:
        self.registry.register(_AddTool())
        schemas = self.registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "add"
        assert "a" in schemas[0].parameters["properties"]

    # -- Execution --

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self) -> None:
        self.registry.register(_AddTool())
        result = await self.registry.execute("add", {"a": 3, "b": 4})
        assert result.success
        assert result.output == "7"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self) -> None:
        self.registry.register(_AsyncEchoTool())
        result = await self.registry.execute("echo", {"msg": "hello"})
        assert result.success
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_execute_nonexistent(self) -> None:
        result = await self.registry.execute("missing", {})
        assert not result.success
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_execute_captures_exceptions(self) -> None:
        class _Boom(BaseTool):
            name = "boom"
            description = "explodes"

            def execute(self, arguments: dict[str, Any]) -> ToolResult:
                raise RuntimeError("kaboom")

        self.registry.register(_Boom())
        result = await self.registry.execute("boom", {})
        assert not result.success
        assert "kaboom" in (result.error or "")

    # -- Singleton --

    def test_singleton_pattern(self) -> None:
        ToolRegistry.reset()
        a = ToolRegistry.get_instance()
        b = ToolRegistry.get_instance()
        assert a is b

    def test_reset_clears_singleton(self) -> None:
        a = ToolRegistry.get_instance()
        ToolRegistry.reset()
        b = ToolRegistry.get_instance()
        assert a is not b

    # -- Discovery --

    def test_discover_tools(self, tmp_path: Path) -> None:
        tool_file = tmp_path / "my_tool.py"
        tool_file.write_text(
            "from src.tools.base import BaseTool, ToolResult\n"
            "class PingTool(BaseTool):\n"
            "    name = 'ping'\n"
            "    description = 'Ping'\n"
            "    def execute(self, arguments):\n"
            "        return ToolResult(output='pong')\n",
            encoding="utf-8",
        )
        classes = ToolRegistry.discover_tools(tmp_path)
        assert len(classes) == 1
        assert classes[0].__name__ == "PingTool"

    def test_discover_skips_private_files(self, tmp_path: Path) -> None:
        private = tmp_path / "_internal.py"
        private.write_text(
            "from src.tools.base import BaseTool, ToolResult\n"
            "class Hidden(BaseTool):\n"
            "    name = 'hidden'\n"
            "    description = 'Hidden'\n"
            "    def execute(self, arguments):\n"
            "        return ToolResult()\n",
            encoding="utf-8",
        )
        classes = ToolRegistry.discover_tools(tmp_path)
        assert len(classes) == 0

    def test_discover_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError):
            ToolRegistry.discover_tools("/nonexistent/path")
