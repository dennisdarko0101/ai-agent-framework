"""Tests for sandboxed code execution."""

from __future__ import annotations

from src.utils.sandbox import (
    SandboxResult,
    check_imports,
    execute_sandboxed,
)

# ───────────────── Import checking ───────────────────────────────


class TestCheckImports:
    def test_no_blocked_imports(self):
        assert check_imports("import json\nimport math") == []

    def test_blocks_os(self):
        blocked = check_imports("import os")
        assert "os" in blocked

    def test_blocks_subprocess(self):
        blocked = check_imports("import subprocess")
        assert "subprocess" in blocked

    def test_blocks_from_import(self):
        blocked = check_imports("from os.path import join")
        assert "os.path" in blocked

    def test_blocks_sys(self):
        blocked = check_imports("import sys")
        assert "sys" in blocked

    def test_multiple_blocked(self):
        blocked = check_imports("import os\nimport subprocess\nimport socket")
        assert len(blocked) == 3

    def test_safe_imports_pass(self):
        code = "import json\nimport math\nimport re\nimport datetime"
        assert check_imports(code) == []

    def test_syntax_error_returns_empty(self):
        assert check_imports("def foo(") == []


# ───────────────── Sandbox execution ─────────────────────────────


class TestExecuteSandboxed:
    def test_simple_print(self):
        result = execute_sandboxed("print('hello world')")
        assert result.success
        assert "hello world" in result.output

    def test_math_expression(self):
        result = execute_sandboxed("print(2 + 3)")
        assert result.success
        assert "5" in result.output

    def test_blocked_import_prevented(self):
        result = execute_sandboxed("import os\nprint(os.getcwd())")
        assert not result.success
        assert result.blocked_imports
        assert "os" in result.blocked_imports

    def test_subprocess_blocked(self):
        result = execute_sandboxed("import subprocess\nsubprocess.run(['ls'])")
        assert not result.success
        assert "subprocess" in result.blocked_imports

    def test_timeout_enforcement(self):
        result = execute_sandboxed(
            "import time\ntime.sleep(10)",
            timeout=1,
        )
        assert result.timed_out
        assert not result.success

    def test_captures_stderr(self):
        result = execute_sandboxed("raise ValueError('boom')")
        assert result.error or "ValueError" in result.output or not result.success

    def test_execution_time_tracked(self):
        result = execute_sandboxed("print('fast')")
        assert result.execution_time >= 0

    def test_multiline_code(self):
        code = """
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
"""
        result = execute_sandboxed(code)
        assert result.success
        assert "Hello, World!" in result.output

    def test_return_value_is_last_line(self):
        result = execute_sandboxed("print('line1')\nprint('line2')")
        assert result.return_value == "line2"

    def test_empty_code(self):
        result = execute_sandboxed("")
        assert result.success


# ───────────────── SandboxResult ─────────────────────────────────


class TestSandboxResult:
    def test_success_when_clean(self):
        r = SandboxResult(output="ok")
        assert r.success

    def test_failure_on_error(self):
        r = SandboxResult(error="something broke")
        assert not r.success

    def test_failure_on_timeout(self):
        r = SandboxResult(timed_out=True)
        assert not r.success

    def test_failure_on_blocked_imports(self):
        r = SandboxResult(blocked_imports=["os"])
        assert not r.success
