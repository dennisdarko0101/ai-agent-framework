# Tools Guide

## Creating a Custom Tool

Extend `BaseTool` and implement `execute()`:

```python
from src.tools.base import BaseTool, ToolResult

class GreetTool(BaseTool):
    name = "greet"
    description = "Return a greeting for a given name."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name to greet"},
        },
        "required": ["name"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(output=f"Hello, {arguments['name']}!")
```

For async tools (HTTP, LLM calls), extend `AsyncTool` and implement `execute_async()`:

```python
from src.tools.base import AsyncTool, ToolResult

class FetchTool(AsyncTool):
    name = "fetch"
    description = "Fetch a URL."
    parameters = { ... }

    async def execute_async(self, arguments: dict) -> ToolResult:
        # async work here
        return ToolResult(output=response_text)
```

Register your tool:

```python
from src.tools.registry import ToolRegistry

registry = ToolRegistry.get_instance()
registry.register(GreetTool())
```

## Tool Schema Format

Every tool exposes a JSON-Schema-compatible parameter definition used by LLM function calling. Call `tool.to_schema()` to get a `ToolSchema` object that can be passed directly to any LLM provider.

```python
schemas = registry.get_schemas()
response = await provider.generate(messages, tools=schemas)
```

## Built-in Tools Reference

### calculator
Evaluate math expressions safely (no `eval()`).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| expression | string | yes | Math expression (e.g. `2 + 3`, `sqrt(16)`, `50% of 200`) |

Supports: `+`, `-`, `*`, `/`, `//`, `**`, `%`, `^`, `sqrt`, `sin`, `cos`, `tan`, `abs`, `round`, `log`, `log10`, `log2`, `ceil`, `floor`, constants `pi`/`e`/`tau`, percentage syntax.

### file_read
Read file contents from the workspace directory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| path | string | yes | Relative path within workspace |

Allowed extensions: `.txt`, `.md`, `.json`, `.csv`, `.py`, `.yaml`, `.yml`, `.toml`. Path traversal outside workspace is blocked.

### file_write
Write or create a file in the workspace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| path | string | yes | Relative path within workspace |
| content | string | yes | Content to write |

Creates intermediate directories automatically. Same workspace restriction as `file_read`.

### code_executor
Execute Python code in a sandboxed subprocess.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| code | string | yes | Python source code |
| language | string | no | `"python"` (only option) |

Dangerous imports (`os`, `subprocess`, `socket`, etc.) are statically blocked before execution. Timeout configured via `SANDBOX_TIMEOUT`.

### web_search
Search the web using DuckDuckGo.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | yes | Search query |
| max_results | integer | no | Max results (default 5) |

Rate-limited to 1 request/second. No API key needed.

### api_caller
Make HTTP requests to external APIs (async).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| url | string | yes | Full URL |
| method | string | no | `GET`/`POST`/`PUT`/`PATCH`/`DELETE` (default `GET`) |
| headers | object | no | HTTP headers |
| body | object | no | JSON body (non-GET methods) |

Configurable timeout and domain allowlist.

### summarize
Summarise text using an LLM (async, requires LLM provider).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | yes | Text to summarise |
| max_length | integer | no | Approximate max words (default 100) |

### extract
Extract structured information from text using an LLM (async, requires LLM provider).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | yes | Source text |
| extract_type | string | yes | `entities`, `dates`, `numbers`, or `key_points` |

## Auto-Discovery

Place tool files in a directory and discover them automatically:

```python
classes = ToolRegistry.discover_tools("./my_tools")
for cls in classes:
    registry.register(cls())
```

Files starting with `_` are skipped. Classes requiring constructor arguments must be instantiated manually.
