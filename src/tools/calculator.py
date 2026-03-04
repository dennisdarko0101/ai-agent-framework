"""Safe math expression evaluator — no eval(), AST-based."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from src.tools.base import BaseTool, ToolResult

# Allowed binary operators
_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions
_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
}

# Named constants
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _safe_eval(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node containing only safe math."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            args = [_safe_eval(a) for a in node.args]
            return _FUNCTIONS[node.func.id](*args)
        func_name = node.func.id if isinstance(node.func, ast.Name) else ast.dump(node.func)
        raise ValueError(f"Unknown function: {func_name}")

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _preprocess(expression: str) -> str:
    """Normalise percentage syntax before AST parsing.

    - ``50%`` → ``(50/100)``
    - ``50% of 200`` → ``(50/100)*200``
    - ``^`` → ``**``
    """
    # Replace caret with Python power operator
    expression = expression.replace("^", "**")

    # "X% of Y" → "(X/100)*Y"
    expression = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s+of\s+",
        r"(\1/100)*",
        expression,
        flags=re.IGNORECASE,
    )

    # Trailing "X%" → "(X/100)"
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression)

    return expression


class CalculatorTool(BaseTool):
    """Evaluate math expressions safely (no ``eval``)."""

    name = "calculator"
    description = "Evaluate a mathematical expression safely."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Math expression to evaluate. Supports +, -, *, /, //, **, %, "
                    "sqrt, sin, cos, tan, abs, round, log, log10, log2, ceil, floor, "
                    "constants pi/e/tau, and percentage syntax (e.g. '50% of 200')."
                ),
            },
        },
        "required": ["expression"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        expression = arguments.get("expression", "")
        if not expression or not expression.strip():
            return ToolResult(error="Empty expression")

        try:
            preprocessed = _preprocess(expression.strip())
            tree = ast.parse(preprocessed, mode="eval")
            value = _safe_eval(tree)

            # Format: drop trailing .0 for whole numbers
            if isinstance(value, float) and value == int(value) and math.isfinite(value):
                formatted = str(int(value))
            else:
                formatted = str(value)

            return ToolResult(output=formatted)
        except ZeroDivisionError:
            return ToolResult(error="Division by zero")
        except (ValueError, TypeError, SyntaxError, OverflowError) as exc:
            return ToolResult(error=f"Invalid expression: {exc}")
