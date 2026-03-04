"""Sandboxed Python code execution with timeout and import restrictions."""

from __future__ import annotations

import ast
import subprocess
import sys
import time
from dataclasses import dataclass, field

# Modules that are never allowed in sandboxed code
BLOCKED_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "importlib",
    "ctypes",
    "signal",
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "webbrowser",
    "code",
    "codeop",
    "compile",
    "compileall",
    "pickletools",
    "multiprocessing",
    "threading",
    "_thread",
})


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""

    output: str = ""
    error: str = ""
    return_value: str | None = None
    execution_time: float = 0.0
    timed_out: bool = False
    blocked_imports: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.error and not self.timed_out and not self.blocked_imports


def check_imports(code: str) -> list[str]:
    """Static analysis to detect blocked imports before execution."""
    blocked: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return blocked

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in BLOCKED_MODULES:
                    blocked.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_module = node.module.split(".")[0]
            if root_module in BLOCKED_MODULES:
                blocked.append(node.module)
    return blocked


def execute_sandboxed(
    code: str,
    timeout: int = 30,
    max_memory_mb: int = 256,
) -> SandboxResult:
    """Execute Python code in an isolated subprocess with restrictions.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.
        max_memory_mb: Maximum memory (advisory, not enforced on all platforms).

    Returns:
        SandboxResult with output, errors, and timing.
    """
    # Static import check
    blocked = check_imports(code)
    if blocked:
        return SandboxResult(
            error=f"Blocked imports detected: {', '.join(blocked)}",
            blocked_imports=blocked,
        )

    start = time.monotonic()

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": ""},  # Minimal environment
        )
        elapsed = time.monotonic() - start

        # Try to extract a return value from the last line of stdout
        stdout = result.stdout
        return_value = None
        if stdout.strip():
            lines = stdout.strip().split("\n")
            return_value = lines[-1] if lines else None

        return SandboxResult(
            output=stdout,
            error=result.stderr,
            return_value=return_value,
            execution_time=elapsed,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return SandboxResult(
            error=f"Execution timed out after {timeout}s",
            execution_time=elapsed,
            timed_out=True,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return SandboxResult(
            error=str(exc),
            execution_time=elapsed,
        )
