"""Tests for the provider-agnostic function calling layer."""

from __future__ import annotations

from src.llm.function_calling import (
    ToolSchema,
    convert_to_anthropic_tools,
    convert_to_openai_tools,
    format_tool_result_for_anthropic,
    format_tool_result_for_openai,
    parse_anthropic_tool_calls,
    parse_openai_tool_calls,
    validate_tool_arguments,
)

# ───────────────────────── Schema basics ─────────────────────────


class TestToolSchema:
    def test_default_parameters(self):
        schema = ToolSchema(name="noop", description="Does nothing")
        assert schema.parameters["type"] == "object"
        assert schema.parameters["properties"] == {}
        assert schema.parameters["required"] == []

    def test_parameter_names(self, weather_schema: ToolSchema):
        names = weather_schema.parameter_names()
        assert "location" in names
        assert "units" in names


# ───────────────────── Anthropic conversion ──────────────────────


class TestAnthropicConversion:
    def test_single_tool(self, weather_schema: ToolSchema):
        tools = convert_to_anthropic_tools([weather_schema])
        assert len(tools) == 1
        t = tools[0]
        assert t["name"] == "get_weather"
        assert t["description"] == "Get current weather for a location"
        assert "input_schema" in t
        assert t["input_schema"]["properties"]["location"]["type"] == "string"

    def test_multiple_tools(self, weather_schema, search_schema):
        tools = convert_to_anthropic_tools([weather_schema, search_schema])
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"get_weather", "web_search"}

    def test_empty_list(self):
        assert convert_to_anthropic_tools([]) == []

    def test_preserves_required_fields(self, weather_schema):
        tools = convert_to_anthropic_tools([weather_schema])
        assert "location" in tools[0]["input_schema"]["required"]


# ───────────────────── OpenAI conversion ─────────────────────────


class TestOpenAIConversion:
    def test_single_tool(self, weather_schema: ToolSchema):
        tools = convert_to_openai_tools([weather_schema])
        assert len(tools) == 1
        t = tools[0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "get_weather"
        assert "parameters" in t["function"]

    def test_multiple_tools(self, weather_schema, search_schema):
        tools = convert_to_openai_tools([weather_schema, search_schema])
        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)

    def test_empty_list(self):
        assert convert_to_openai_tools([]) == []

    def test_preserves_parameter_schema(self, weather_schema):
        tools = convert_to_openai_tools([weather_schema])
        params = tools[0]["function"]["parameters"]
        assert params["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]


# ──────────────── Anthropic tool call parsing ────────────────────


class TestParseAnthropicToolCalls:
    def test_single_tool_use_block(self):
        blocks = [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {"location": "London"},
            },
        ]
        calls = parse_anthropic_tool_calls(blocks)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "get_weather"
        assert calls[0]["arguments"] == {"location": "London"}
        assert calls[0]["call_id"] == "toolu_abc123"

    def test_multiple_tool_use_blocks(self):
        blocks = [
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "b", "input": {"x": 1}},
        ]
        calls = parse_anthropic_tool_calls(blocks)
        assert len(calls) == 2

    def test_no_tool_use_blocks(self):
        blocks = [{"type": "text", "text": "Hello"}]
        assert parse_anthropic_tool_calls(blocks) == []

    def test_empty_blocks(self):
        assert parse_anthropic_tool_calls([]) == []

    def test_missing_id_generates_one(self):
        blocks = [{"type": "tool_use", "name": "foo", "input": {}}]
        calls = parse_anthropic_tool_calls(blocks)
        assert calls[0]["call_id"].startswith("call_")


# ──────────────── OpenAI tool call parsing ───────────────────────


class TestParseOpenAIToolCalls:
    def test_single_function_call(self):
        raw = [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ]
        calls = parse_openai_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "get_weather"
        assert calls[0]["arguments"] == {"location": "Paris"}

    def test_multiple_function_calls(self):
        raw = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "a", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "b", "arguments": '{"x": 1}'},
            },
        ]
        calls = parse_openai_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0]["tool_name"] == "a"
        assert calls[1]["arguments"] == {"x": 1}

    def test_none_input(self):
        assert parse_openai_tool_calls(None) == []

    def test_malformed_json_arguments(self):
        raw = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "foo", "arguments": "not json"},
            }
        ]
        calls = parse_openai_tool_calls(raw)
        assert calls[0]["arguments"] == {"_raw": "not json"}


# ──────────────── Argument validation ────────────────────────────


class TestValidateArguments:
    def test_valid_arguments(self, weather_schema):
        errors = validate_tool_arguments(
            {"location": "London", "units": "celsius"},
            weather_schema,
        )
        assert errors == []

    def test_missing_required(self, weather_schema):
        errors = validate_tool_arguments({}, weather_schema)
        assert any("Missing required" in e for e in errors)

    def test_unknown_parameter(self, weather_schema):
        errors = validate_tool_arguments(
            {"location": "London", "bogus": 42},
            weather_schema,
        )
        assert any("Unknown parameter" in e for e in errors)

    def test_wrong_type(self, weather_schema):
        errors = validate_tool_arguments(
            {"location": 123},
            weather_schema,
        )
        assert any("expected type" in e for e in errors)

    def test_invalid_enum_value(self, weather_schema):
        errors = validate_tool_arguments(
            {"location": "London", "units": "kelvin"},
            weather_schema,
        )
        assert any("not in allowed values" in e for e in errors)

    def test_valid_enum_value(self, weather_schema):
        errors = validate_tool_arguments(
            {"location": "London", "units": "celsius"},
            weather_schema,
        )
        assert errors == []

    def test_optional_param_not_required(self, weather_schema):
        errors = validate_tool_arguments(
            {"location": "London"},
            weather_schema,
        )
        assert errors == []

    def test_integer_type_check(self, search_schema):
        errors = validate_tool_arguments(
            {"query": "test", "max_results": "five"},
            search_schema,
        )
        assert any("expected type" in e for e in errors)


# ──────────────── Tool result formatting ─────────────────────────


class TestFormatToolResults:
    def test_anthropic_result(self):
        result = format_tool_result_for_anthropic("toolu_1", "72°F")
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_1"
        assert result["content"] == "72°F"
        assert result["is_error"] is False

    def test_anthropic_error_result(self):
        result = format_tool_result_for_anthropic("toolu_1", "City not found", is_error=True)
        assert result["is_error"] is True

    def test_openai_result(self):
        result = format_tool_result_for_openai("call_1", "72°F")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert result["content"] == "72°F"
