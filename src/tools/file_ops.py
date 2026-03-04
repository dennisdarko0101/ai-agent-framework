"""File read/write tools with workspace sandboxing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools.base import BaseTool, ToolResult
from src.utils.logger import get_logger

log = get_logger(__name__)

# Extensions FileReadTool will serve
ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml", ".toml"})


def _resolve_safe(workspace: Path, user_path: str) -> Path:
    """Resolve *user_path* inside *workspace*, blocking traversal escapes."""
    resolved = (workspace / user_path).resolve()
    workspace_resolved = workspace.resolve()
    if not str(resolved).startswith(str(workspace_resolved)):
        raise PermissionError(f"Path escapes workspace: {user_path}")
    return resolved


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class FileReadTool(BaseTool):
    """Read file contents from the workspace directory."""

    name = "file_read"
    description = "Read the contents of a file in the workspace."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path within the workspace directory.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace_dir: str | Path = "./workspace") -> None:
        self.workspace = Path(workspace_dir).resolve()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        rel_path = arguments.get("path", "")
        if not rel_path:
            return ToolResult(error="Missing 'path' argument")

        try:
            target = _resolve_safe(self.workspace, rel_path)
        except PermissionError as exc:
            return ToolResult(error=str(exc))

        if not target.exists():
            return ToolResult(error=f"File not found: {rel_path}")

        if target.suffix.lower() not in ALLOWED_EXTENSIONS:
            return ToolResult(
                error=f"Unsupported file type: {target.suffix} "
                f"(allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))})",
            )

        try:
            content = target.read_text(encoding="utf-8")
            log.info("file_read", path=str(target), size=len(content))
            return ToolResult(output=content)
        except Exception as exc:
            return ToolResult(error=f"Read error: {exc}")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class FileWriteTool(BaseTool):
    """Write or create a file in the workspace directory."""

    name = "file_write"
    description = "Write content to a file in the workspace (creates directories as needed)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path within the workspace directory.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_dir: str | Path = "./workspace") -> None:
        self.workspace = Path(workspace_dir).resolve()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        rel_path = arguments.get("path", "")
        content = arguments.get("content", "")
        if not rel_path:
            return ToolResult(error="Missing 'path' argument")

        try:
            target = _resolve_safe(self.workspace, rel_path)
        except PermissionError as exc:
            return ToolResult(error=str(exc))

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            log.info("file_write", path=str(target), size=len(content))
            return ToolResult(output=f"Written {len(content)} bytes to {rel_path}")
        except Exception as exc:
            return ToolResult(error=f"Write error: {exc}")
