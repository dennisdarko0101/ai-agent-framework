"""Provider-agnostic function calling: schema definitions, format conversion, parsing, validation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """Unified tool/function schema used across all providers.

    Parameters follow JSON Schema (draft 2020-12 compatible subset).
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    def parameter_names(self) -> list[str]:
        """Return the declared parameter names."""
        return list(self.parameters.get("properties", {}).keys())


# ---------------------------------------------------------------------------
# Anthropic format conversion
# ---------------------------------------------------------------------------

def convert_to_anthropic_tools(schemas: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert ToolSchema list to Anthropic tool-use format.

    Anthropic expects:
        {
            "name": "...",
            "description": "...",
            "input_schema": { JSON Schema }
        }
    """
    tools: list[dict[str, Any]] = []
    for schema in schemas:
        tools.append({
            "name": schema.name,
            "description": schema.description,
            "input_schema": schema.parameters,
        })
    return tools


def parse_anthropic_tool_calls(
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract tool calls from Anthropic response content blocks.

    Anthropic returns content blocks with type == "tool_use":
        {
            "type": "tool_use",
            "id": "toolu_...",
            "name": "get_weather",
            "input": {"location": "London"}
        }
    """
    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append({
                "call_id": block.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                "tool_name": block["name"],
                "arguments": block.get("input", {}),
            })
    return tool_calls


def format_tool_result_for_anthropic(
    call_id: str,
    result: str,
    is_error: bool = False,
) -> dict[str, Any]:
    """Format a tool execution result to feed back into Anthropic messages.

    Returns a content block:
        {
            "type": "tool_result",
            "tool_use_id": "toolu_...",
            "content": "result text",
            "is_error": false
        }
    """
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": result,
        "is_error": is_error,
    }


# ---------------------------------------------------------------------------
# OpenAI format conversion
# ---------------------------------------------------------------------------

def convert_to_openai_tools(schemas: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert ToolSchema list to OpenAI function-calling format.

    OpenAI expects:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { JSON Schema }
            }
        }
    """
    tools: list[dict[str, Any]] = []
    for schema in schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters,
            },
        })
    return tools


def parse_openai_tool_calls(
    tool_calls_data: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI response.

    OpenAI returns tool_calls on the message:
        [
            {
                "id": "call_...",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"London\"}"
                }
            }
        ]
    """
    if not tool_calls_data:
        return []

    tool_calls: list[dict[str, Any]] = []
    for tc in tool_calls_data:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = {"_raw": raw_args}

        tool_calls.append({
            "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
            "tool_name": func.get("name", ""),
            "arguments": arguments,
        })
    return tool_calls


def format_tool_result_for_openai(
    call_id: str,
    result: str,
) -> dict[str, Any]:
    """Format a tool result as an OpenAI tool message.

    Returns:
        {
            "role": "tool",
            "tool_call_id": "call_...",
            "content": "result text"
        }
    """
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tool_arguments(
    arguments: dict[str, Any],
    schema: ToolSchema,
) -> list[str]:
    """Validate tool call arguments against the ToolSchema.

    Returns a list of validation error messages (empty == valid).
    Performs structural checks without pulling in a full JSON Schema validator.
    """
    errors: list[str] = []
    params = schema.parameters
    properties = params.get("properties", {})
    required = params.get("required", [])

    # Check required parameters are present
    for req in required:
        if req not in arguments:
            errors.append(f"Missing required parameter: '{req}'")

    # Check for unknown parameters
    if properties:
        for key in arguments:
            if key not in properties:
                errors.append(f"Unknown parameter: '{key}'")

    # Basic type checking
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, value in arguments.items():
        if key not in properties:
            continue
        expected_type_name = properties[key].get("type")
        if expected_type_name and expected_type_name in type_map:
            expected = type_map[expected_type_name]
            if not isinstance(value, expected):
                errors.append(
                    f"Parameter '{key}' expected type '{expected_type_name}', "
                    f"got '{type(value).__name__}'"
                )

    # Enum validation
    for key, value in arguments.items():
        if key not in properties:
            continue
        enum_values = properties[key].get("enum")
        if enum_values is not None and value not in enum_values:
            errors.append(
                f"Parameter '{key}' value '{value}' not in allowed values: {enum_values}"
            )

    return errors
