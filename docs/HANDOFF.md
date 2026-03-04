# Project Handoff — AI Agent Framework

## Current Status: Phase 1, Step 1 Complete

### What's Done

**Scaffolding**
- `pyproject.toml` with all dependencies (runtime + dev)
- Full directory structure with `__init__.py` files
- Makefile, .gitignore, LICENSE, .env.example
- GitHub Actions CI workflow (Python 3.11 + 3.12)

**Configuration** (`src/config/settings.py`)
- Pydantic Settings with `.env` support
- API keys (Anthropic, OpenAI) as SecretStr
- Provider, model, agent, sandbox, memory, and logging settings
- Cached singleton via `get_settings()`

**LLM Provider Layer** (`src/llm/provider.py`)
- `BaseLLMProvider` abstract class with retry + token tracking
- `ClaudeProvider` — Anthropic Messages API with native tool use
- `OpenAIProvider` — OpenAI Chat Completions with function calling
- `ProviderFactory` — Create provider from settings or explicit params
- Data classes: `LLMResponse`, `ToolCall`, `TokenUsage`

**Function Calling** (`src/llm/function_calling.py`)
- `ToolSchema` dataclass — provider-agnostic tool definition
- `convert_to_anthropic_tools()` / `convert_to_openai_tools()`
- `parse_anthropic_tool_calls()` / `parse_openai_tool_calls()`
- `format_tool_result_for_anthropic()` / `format_tool_result_for_openai()`
- `validate_tool_arguments()` — type, required, enum checks

**Utilities**
- `src/utils/logger.py` — structlog setup (console + JSON)
- `src/utils/sandbox.py` — Subprocess sandbox with import blocking, timeout, output capture

**Tests** (37+ tests)
- `test_function_calling.py` — 19 tests: schema conversion, parsing, validation, result formatting
- `test_provider.py` — 14 tests: Claude + OpenAI mocked, token tracking, factory
- `test_sandbox.py` — 14 tests: execution, import blocking, timeout, result dataclass

### Next Steps — Phase 1, Step 2

1. **Tool Registry** (`src/tools/`)
   - `BaseTool` abstract class with execute + schema
   - `ToolRegistry` for dynamic registration and lookup
   - Built-in tools: WebSearchTool, PythonExecutorTool, CalculatorTool

2. **Agent Core** (`src/agents/`)
   - `BaseAgent` with ReAct-style loop
   - Tool execution integration
   - Step-by-step reasoning with configurable max steps

### Architecture Decisions

- **No LangChain** — Everything built from primitives for full control
- **Async-first** — All LLM calls are async
- **Provider-agnostic** — ToolSchema converts to any provider format
- **Subprocess sandbox** — Code runs in isolated process, not in-process
- **Static import analysis** — AST-based check before execution
