# Project Handoff — AI Agent Framework

## Current Status: Phase 1, Step 2 Complete

### What's Done

**Phase 1, Step 1 — Scaffolding, Config, LLM Layer**
- `pyproject.toml` with all dependencies (runtime + dev)
- Full directory structure with `__init__.py` files
- Makefile, .gitignore, LICENSE, .env.example
- GitHub Actions CI workflow (Python 3.11 + 3.12)
- Pydantic Settings with `.env` support, cached singleton
- `BaseLLMProvider` + `ClaudeProvider` + `OpenAIProvider` + `ProviderFactory`
- `ToolSchema`, format conversion, argument validation
- structlog logging, subprocess sandbox with import blocking

**Phase 1, Step 2 — Tool System** (`src/tools/`)

- `BaseTool` abstract class with `name`, `description`, `parameters`, `execute()`, `to_schema()`
- `AsyncTool` variant for async tool execution (HTTP, LLM-dependent tools)
- `ToolResult` dataclass: `output`, `error`, `execution_time`, `success`
- `ToolRegistry` with singleton, registration, lookup, schema generation, async execute
- Auto-discovery: scan a directory for tool classes via `discover_tools()`
- Validation: duplicate names, empty names/descriptions, type checking on register

**Built-in Tools**
- `CalculatorTool` — Safe AST-based math evaluation (no `eval()`), supports arithmetic, trig, sqrt, percentages, constants
- `FileReadTool` / `FileWriteTool` — Workspace-sandboxed file I/O with path traversal protection
- `CodeExecutorTool` — Subprocess sandbox integration with import blocking and timeout
- `WebSearchTool` — DuckDuckGo search with rate limiting (1 req/s)
- `APICallerTool` — Async HTTP client (httpx) with domain allowlist and timeout
- `SummarizeTool` / `ExtractTool` — LLM-powered text processing (entities, dates, numbers, key points)

**Settings Additions**
- `tool_workspace_dir` — Root directory for file tools
- `api_caller_timeout` — HTTP request timeout
- `api_caller_allowed_domains` — Domain allowlist for API caller

**Tests** (67 prior + 52 new = 119+ tests)
- `test_tools.py` — 35+ tests: each tool's execute, parameter validation, error handling, workspace restriction, sandbox integration
- `test_registry.py` — 17 tests: register, retrieve, duplicate rejection, listing, schemas, sync/async execute, singleton, discovery

**Documentation**
- `docs/TOOLS_GUIDE.md` — How to create custom tools, built-in tools reference, schema format, auto-discovery

### Next Steps — Phase 1, Step 3

1. **Agent Core** (`src/agents/`)
   - `BaseAgent` with ReAct-style reasoning loop
   - Tool execution integration via ToolRegistry
   - Step-by-step reasoning with configurable max steps
   - Conversation history management

2. **Memory System** (`src/memory/`)
   - Conversation memory (message history)
   - Summary memory (LLM-compressed)
   - Vector memory (ChromaDB embeddings)

### Architecture Decisions

- **No LangChain** — Everything built from primitives for full control
- **Async-first** — All LLM calls are async; sync tools called directly in async context
- **Provider-agnostic** — ToolSchema converts to any provider format
- **Subprocess sandbox** — Code runs in isolated process, not in-process
- **Static import analysis** — AST-based check before execution
- **Composable tools** — Add a new tool by subclassing BaseTool and registering it
- **Registry singleton** — Global registry with reset for testing
