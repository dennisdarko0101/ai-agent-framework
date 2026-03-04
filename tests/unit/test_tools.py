"""Tests for built-in tools — 35+ tests covering execute, validation, and error handling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.api_caller import APICallerTool
from src.tools.base import AsyncTool, BaseTool, ToolResult
from src.tools.calculator import CalculatorTool
from src.tools.code_executor import CodeExecutorTool
from src.tools.file_ops import FileReadTool, FileWriteTool
from src.tools.text_tools import ExtractTool, SummarizeTool
from src.tools.web_search import WebSearchTool

# ───────── ToolResult ─────────────────────────────────────────────────

class TestToolResult:
    def test_success_by_default(self) -> None:
        r = ToolResult(output="ok")
        assert r.success is True
        assert r.error is None

    def test_error_sets_success_false(self) -> None:
        r = ToolResult(error="boom")
        assert r.success is False

    def test_explicit_success_overridden_by_error(self) -> None:
        r = ToolResult(output="ok", error="but also error", success=True)
        # __post_init__ should force success=False when error is set
        assert r.success is False


# ───────── BaseTool / AsyncTool ───────────────────────────────────────

class TestBaseToolInterface:
    def test_to_schema(self) -> None:
        class DummyTool(BaseTool):
            name = "dummy"
            description = "A dummy tool"
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }

            def execute(self, arguments: dict[str, Any]) -> ToolResult:
                return ToolResult(output="ok")

        schema = DummyTool().to_schema()
        assert schema.name == "dummy"
        assert schema.description == "A dummy tool"
        assert "x" in schema.parameters["properties"]

    def test_async_tool_execute_raises(self) -> None:
        class MyAsync(AsyncTool):
            name = "async_dummy"
            description = "async dummy"

            async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
                return ToolResult(output="ok")

        with pytest.raises(NotImplementedError):
            MyAsync().execute({})

    def test_repr(self) -> None:
        class T(BaseTool):
            name = "r"
            description = "repr test"
            def execute(self, arguments: dict[str, Any]) -> ToolResult:
                return ToolResult()

        assert "name='r'" in repr(T())


# ───────── CalculatorTool ─────────────────────────────────────────────

class TestCalculator:
    def setup_method(self) -> None:
        self.calc = CalculatorTool()

    def test_addition(self) -> None:
        r = self.calc.execute({"expression": "2 + 3"})
        assert r.success and r.output == "5"

    def test_subtraction(self) -> None:
        r = self.calc.execute({"expression": "10 - 4"})
        assert r.output == "6"

    def test_multiplication(self) -> None:
        r = self.calc.execute({"expression": "6 * 7"})
        assert r.output == "42"

    def test_division(self) -> None:
        r = self.calc.execute({"expression": "15 / 4"})
        assert r.output == "3.75"

    def test_exponents(self) -> None:
        r = self.calc.execute({"expression": "2 ** 10"})
        assert r.output == "1024"

    def test_caret_as_power(self) -> None:
        r = self.calc.execute({"expression": "2^10"})
        assert r.output == "1024"

    def test_sqrt(self) -> None:
        r = self.calc.execute({"expression": "sqrt(144)"})
        assert r.output == "12"

    def test_trig_sin(self) -> None:
        r = self.calc.execute({"expression": "sin(0)"})
        assert r.output == "0"

    def test_constant_pi(self) -> None:
        r = self.calc.execute({"expression": "pi"})
        assert float(r.output) == pytest.approx(math.pi)

    def test_complex_expression(self) -> None:
        r = self.calc.execute({"expression": "(2 + 3) * 4 - 1"})
        assert r.output == "19"

    def test_percentage(self) -> None:
        r = self.calc.execute({"expression": "50% of 200"})
        assert r.output == "100"

    def test_percentage_standalone(self) -> None:
        r = self.calc.execute({"expression": "25%"})
        assert r.output == "0.25"

    def test_division_by_zero(self) -> None:
        r = self.calc.execute({"expression": "1 / 0"})
        assert not r.success
        assert "Division by zero" in (r.error or "")

    def test_invalid_expression(self) -> None:
        r = self.calc.execute({"expression": "not a math"})
        assert not r.success

    def test_empty_expression(self) -> None:
        r = self.calc.execute({"expression": ""})
        assert not r.success

    def test_to_schema(self) -> None:
        schema = self.calc.to_schema()
        assert schema.name == "calculator"
        assert "expression" in schema.parameters["properties"]


# ───────── FileReadTool ───────────────────────────────────────────────

class TestFileRead:
    def test_read_success(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = FileReadTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "hello.txt"})
        assert r.success and r.output == "hello world"

    def test_read_json(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"a":1}', encoding="utf-8")
        tool = FileReadTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "data.json"})
        assert r.success

    def test_read_not_found(self, tmp_path: Path) -> None:
        tool = FileReadTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "nope.txt"})
        assert not r.success
        assert "not found" in (r.error or "").lower()

    def test_read_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        tool = FileReadTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "image.png"})
        assert not r.success
        assert "Unsupported" in (r.error or "")

    def test_read_path_traversal(self, tmp_path: Path) -> None:
        tool = FileReadTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "../../etc/passwd"})
        assert not r.success
        assert "escapes" in (r.error or "").lower()

    def test_read_missing_arg(self, tmp_path: Path) -> None:
        tool = FileReadTool(workspace_dir=tmp_path)
        r = tool.execute({})
        assert not r.success


# ───────── FileWriteTool ──────────────────────────────────────────────

class TestFileWrite:
    def test_write_success(self, tmp_path: Path) -> None:
        tool = FileWriteTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "out.txt", "content": "data"})
        assert r.success
        assert (tmp_path / "out.txt").read_text() == "data"

    def test_write_creates_directories(self, tmp_path: Path) -> None:
        tool = FileWriteTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "sub/dir/file.txt", "content": "nested"})
        assert r.success
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"

    def test_write_path_traversal(self, tmp_path: Path) -> None:
        tool = FileWriteTool(workspace_dir=tmp_path)
        r = tool.execute({"path": "../escape.txt", "content": "bad"})
        assert not r.success
        assert "escapes" in (r.error or "").lower()

    def test_write_missing_path(self, tmp_path: Path) -> None:
        tool = FileWriteTool(workspace_dir=tmp_path)
        r = tool.execute({"content": "no path"})
        assert not r.success


# ───────── CodeExecutorTool ───────────────────────────────────────────

class TestCodeExecutor:
    def setup_method(self) -> None:
        self.tool = CodeExecutorTool()

    def test_simple_execution(self) -> None:
        r = self.tool.execute({"code": "print(2 + 2)"})
        assert r.success
        assert "4" in r.output

    def test_execution_error(self) -> None:
        r = self.tool.execute({"code": "raise ValueError('oops')"})
        assert not r.success

    def test_blocked_import(self) -> None:
        r = self.tool.execute({"code": "import os"})
        assert not r.success
        assert "Blocked" in (r.error or "")

    def test_unsupported_language(self) -> None:
        r = self.tool.execute({"code": "console.log(1)", "language": "javascript"})
        assert not r.success
        assert "Unsupported" in (r.error or "")

    def test_empty_code(self) -> None:
        r = self.tool.execute({"code": ""})
        assert not r.success


# ───────── WebSearchTool ──────────────────────────────────────────────

class TestWebSearch:
    def test_search_success(self) -> None:
        mock_results = [
            {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
            {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
        ]
        tool = WebSearchTool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_results
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs.return_value = mock_instance

            r = tool.execute({"query": "test query", "max_results": 2})

        assert r.success
        assert "Result 1" in r.output
        assert "Result 2" in r.output

    def test_search_empty_query(self) -> None:
        tool = WebSearchTool()
        r = tool.execute({"query": ""})
        assert not r.success

    def test_search_network_error(self) -> None:
        tool = WebSearchTool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = ConnectionError("offline")
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_ddgs.return_value = mock_instance

            r = tool.execute({"query": "test"})

        assert not r.success
        assert "failed" in (r.error or "").lower()


# ───────── APICallerTool ──────────────────────────────────────────────

class TestAPICaller:
    @pytest.mark.asyncio
    async def test_get_request(self) -> None:
        tool = APICallerTool(timeout=5)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"ok": true}'

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_async_client.return_value = mock_client

            r = await tool.execute_async({"url": "https://api.example.com/data"})

        assert r.success
        assert "200" in r.output

    @pytest.mark.asyncio
    async def test_domain_blocked(self) -> None:
        tool = APICallerTool(allowed_domains=["safe.example.com"])
        r = await tool.execute_async({"url": "https://evil.example.com/data"})
        assert not r.success
        assert "not allowed" in (r.error or "").lower()

    @pytest.mark.asyncio
    async def test_missing_url(self) -> None:
        tool = APICallerTool()
        r = await tool.execute_async({})
        assert not r.success


# ───────── SummarizeTool ──────────────────────────────────────────────

class TestSummarize:
    @pytest.mark.asyncio
    async def test_summarize(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(content="Short summary.")

        tool = SummarizeTool(llm=mock_llm)
        r = await tool.execute_async({"text": "A long piece of text " * 50})

        assert r.success
        assert r.output == "Short summary."
        mock_llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summarize_empty(self) -> None:
        tool = SummarizeTool(llm=AsyncMock())
        r = await tool.execute_async({"text": ""})
        assert not r.success


# ───────── ExtractTool ────────────────────────────────────────────────

class TestExtract:
    @pytest.mark.asyncio
    async def test_extract_entities(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(content="- Alice\n- Bob")

        tool = ExtractTool(llm=mock_llm)
        r = await tool.execute_async({
            "text": "Alice met Bob in London.",
            "extract_type": "entities",
        })

        assert r.success
        assert "Alice" in r.output

    @pytest.mark.asyncio
    async def test_extract_invalid_type(self) -> None:
        tool = ExtractTool(llm=AsyncMock())
        r = await tool.execute_async({"text": "hello", "extract_type": "nope"})
        assert not r.success
        assert "Unknown" in (r.error or "")

    @pytest.mark.asyncio
    async def test_extract_empty_text(self) -> None:
        tool = ExtractTool(llm=AsyncMock())
        r = await tool.execute_async({"text": "", "extract_type": "entities"})
        assert not r.success
