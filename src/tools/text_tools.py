"""LLM-powered text tools — summarisation and information extraction."""

from __future__ import annotations

from typing import Any

from src.llm.provider import BaseLLMProvider
from src.tools.base import AsyncTool, ToolResult
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------

class SummarizeTool(AsyncTool):
    """Summarise text using an LLM provider."""

    name = "summarize"
    description = "Produce a concise summary of the given text."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to summarise.",
            },
            "max_length": {
                "type": "integer",
                "description": "Approximate max length of the summary in words.",
            },
        },
        "required": ["text"],
    }

    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
        text = arguments.get("text", "")
        max_length = arguments.get("max_length", 100)

        if not text.strip():
            return ToolResult(error="Empty text")

        prompt = (
            f"Summarise the following text in at most {max_length} words. "
            "Return only the summary, no preamble.\n\n"
            f"{text}"
        )

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max(256, max_length * 3),
            )
            log.info("summarize", input_len=len(text), output_len=len(response.content))
            return ToolResult(output=response.content)
        except Exception as exc:
            return ToolResult(error=f"Summarisation failed: {exc}")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

_EXTRACT_PROMPTS: dict[str, str] = {
    "entities": "Extract all named entities (people, organisations, places) as a bulleted list.",
    "dates": "Extract all dates and time references as a bulleted list.",
    "numbers": "Extract all numeric values with their context as a bulleted list.",
    "key_points": "Extract the key points as a bulleted list.",
}


class ExtractTool(AsyncTool):
    """Extract structured information from text using an LLM."""

    name = "extract"
    description = "Extract entities, dates, numbers, or key points from text."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Source text to extract from.",
            },
            "extract_type": {
                "type": "string",
                "description": "What to extract.",
                "enum": ["entities", "dates", "numbers", "key_points"],
            },
        },
        "required": ["text", "extract_type"],
    }

    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
        text = arguments.get("text", "")
        extract_type = arguments.get("extract_type", "")

        if not text.strip():
            return ToolResult(error="Empty text")

        instruction = _EXTRACT_PROMPTS.get(extract_type)
        if instruction is None:
            valid = ", ".join(sorted(_EXTRACT_PROMPTS))
            return ToolResult(error=f"Unknown extract_type: {extract_type!r} (valid: {valid})")

        prompt = f"{instruction}\n\nText:\n{text}"

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            log.info("extract", extract_type=extract_type, input_len=len(text))
            return ToolResult(output=response.content)
        except Exception as exc:
            return ToolResult(error=f"Extraction failed: {exc}")
