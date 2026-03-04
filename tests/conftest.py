"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.llm.function_calling import ToolSchema


@pytest.fixture
def weather_schema() -> ToolSchema:
    """A simple weather-lookup tool schema used across tests."""
    return ToolSchema(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            "required": ["location"],
        },
    )


@pytest.fixture
def search_schema() -> ToolSchema:
    """A web search tool schema."""
    return ToolSchema(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                },
            },
            "required": ["query"],
        },
    )


@pytest.fixture
def calculator_schema() -> ToolSchema:
    """A calculator tool schema."""
    return ToolSchema(
        name="calculate",
        description="Evaluate a math expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                },
            },
            "required": ["expression"],
        },
    )
