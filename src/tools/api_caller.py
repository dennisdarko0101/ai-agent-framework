"""HTTP API caller tool with domain allowlisting."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from src.tools.base import AsyncTool, ToolResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class APICallerTool(AsyncTool):
    """Make HTTP requests to external APIs with safety controls."""

    name = "api_caller"
    description = "Make HTTP requests (GET/POST/PUT/PATCH/DELETE) to external APIs."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to request.",
            },
            "method": {
                "type": "string",
                "description": "HTTP method.",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers as key-value pairs.",
            },
            "body": {
                "type": "object",
                "description": "Request body (sent as JSON for non-GET methods).",
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        timeout: int = 30,
        allowed_domains: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.allowed_domains = allowed_domains or []

    def _check_domain(self, url: str) -> str | None:
        """Return an error message if the domain is blocked, else None."""
        if not self.allowed_domains:
            return None
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname not in self.allowed_domains:
            return f"Domain not allowed: {hostname} (allowed: {', '.join(self.allowed_domains)})"
        return None

    async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
        import httpx  # lazy to keep import lightweight

        url = arguments.get("url", "")
        method = arguments.get("method", "GET").upper()
        headers = arguments.get("headers") or {}
        body = arguments.get("body")

        if not url:
            return ToolResult(error="Missing 'url' argument")

        domain_err = self._check_domain(url)
        if domain_err:
            return ToolResult(error=domain_err)

        start = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "method": method,
                "url": url,
                "headers": headers,
                "timeout": self.timeout,
            }
            if body and method != "GET":
                kwargs["json"] = body

            async with httpx.AsyncClient() as client:
                response = await client.request(**kwargs)

            result_text = (
                f"Status: {response.status_code}\n"
                f"Headers: {dict(response.headers)}\n"
                f"Body: {response.text[:4096]}"
            )
            log.info("api_call", url=url, method=method, status=response.status_code)
            return ToolResult(
                output=result_text,
                execution_time=time.monotonic() - start,
            )

        except httpx.TimeoutException:
            return ToolResult(
                error=f"Request timed out after {self.timeout}s",
                execution_time=time.monotonic() - start,
            )
        except Exception as exc:
            return ToolResult(
                error=f"Request failed: {exc}",
                execution_time=time.monotonic() - start,
            )
