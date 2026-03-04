"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

import time
from typing import Any

from src.tools.base import BaseTool, ToolResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Search the web via DuckDuckGo with rate limiting."""

    name = "web_search"
    description = "Search the web using DuckDuckGo and return results with title, snippet, and URL."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
            },
        },
        "required": ["query"],
    }

    # Rate-limit: minimum seconds between requests
    _MIN_INTERVAL = 1.0

    def __init__(self) -> None:
        self._last_request: float = 0.0

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - elapsed)

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query", "").strip()
        max_results = arguments.get("max_results", 5)

        if not query:
            return ToolResult(error="Empty search query")

        try:
            from duckduckgo_search import DDGS  # lazy import
        except ImportError:
            return ToolResult(error="duckduckgo-search package is not installed")

        self._wait_for_rate_limit()
        start = time.monotonic()

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
            self._last_request = time.monotonic()

            if not raw_results:
                return ToolResult(
                    output="No results found.",
                    execution_time=time.monotonic() - start,
                )

            lines: list[str] = []
            for i, r in enumerate(raw_results, 1):
                lines.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No snippet')}\n"
                    f"   URL: {r.get('href', 'N/A')}"
                )

            log.info("web_search", query=query, result_count=len(raw_results))
            return ToolResult(
                output="\n\n".join(lines),
                execution_time=time.monotonic() - start,
            )

        except Exception as exc:
            return ToolResult(
                error=f"Search failed: {exc}",
                execution_time=time.monotonic() - start,
            )
