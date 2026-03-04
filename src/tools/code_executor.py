"""Code execution tool — delegates to the subprocess sandbox."""

from __future__ import annotations

from typing import Any

from src.config.settings import get_settings
from src.tools.base import BaseTool, ToolResult
from src.utils.logger import get_logger
from src.utils.sandbox import execute_sandboxed

log = get_logger(__name__)


class CodeExecutorTool(BaseTool):
    """Execute Python code in a sandboxed subprocess."""

    name = "code_executor"
    description = (
        "Execute Python code in a secure sandbox. "
        "Dangerous imports (os, subprocess, etc.) are blocked."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
            "language": {
                "type": "string",
                "description": "Programming language (only 'python' supported).",
                "enum": ["python"],
            },
        },
        "required": ["code"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code", "")
        language = arguments.get("language", "python")

        if language != "python":
            return ToolResult(error=f"Unsupported language: {language}")
        if not code.strip():
            return ToolResult(error="Empty code")

        settings = get_settings()
        log.info("code_execute", code_length=len(code))

        sandbox_result = execute_sandboxed(
            code,
            timeout=settings.sandbox_timeout,
            max_memory_mb=settings.sandbox_max_memory_mb,
        )

        if sandbox_result.success:
            output_parts = []
            if sandbox_result.output:
                output_parts.append(sandbox_result.output.rstrip("\n"))
            if sandbox_result.return_value and sandbox_result.return_value != sandbox_result.output.strip():
                output_parts.append(f"Return value: {sandbox_result.return_value}")
            return ToolResult(
                output="\n".join(output_parts) if output_parts else "(no output)",
                execution_time=sandbox_result.execution_time,
            )

        return ToolResult(
            error=sandbox_result.error,
            execution_time=sandbox_result.execution_time,
        )
