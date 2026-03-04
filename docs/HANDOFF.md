# Project Handoff — AI Agent Framework

## Current Status: Phase 1, Step 3 Complete

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
- Built-in tools: Calculator, FileRead/Write, CodeExecutor, WebSearch, APICaller, Summarize, Extract

**Phase 1, Step 3 — Memory System** (`src/memory/`)

- `BaseMemory` abstract class (async interface): `add`, `get_context`, `search`, `clear`, `get_stats`
- `Message` dataclass: `role`, `content`, `metadata`, `timestamp`, `id`, `to_dict()`
- `estimate_tokens()` utility: tiktoken when available, ~4 chars/token fallback
- `ConversationMemory` — Sliding-window buffer with FIFO eviction, token-aware context building
- `SummaryMemory` — Running LLM-generated summary + recent messages, auto-summarisation trigger
- `VectorMemory` — ChromaDB-backed semantic storage/retrieval, metadata filtering, deduplication
- `CompositeMemory` — Merges multiple backends, broadcasts adds, deduplicates by ID + content

**Settings Additions (Step 3)**
- `memory_max_messages` — ConversationMemory buffer size
- `summary_threshold` — Message count that triggers summarisation
- `summary_recent_count` — Recent messages kept after summarisation

**Tests** (132 prior + 37 new = 169 tests)
- `test_memory.py` — 37 tests: Message dataclass, estimate_tokens, ConversationMemory (add, get_context, FIFO, token limit, search, clear, stats), SummaryMemory (threshold trigger, context format, failure handling, mocked LLM), VectorMemory (mocked ChromaDB — add, search, metadata filter, dedup, clear, stats), CompositeMemory (broadcast, merge, dedup, search, clear, stats)

**Documentation**
- `docs/AGENTS_GUIDE.md` — Memory system overview, when to use which type, configuration
- `docs/TOOLS_GUIDE.md` — Custom tool creation, built-in tools reference

### Next Steps — Phase 1, Step 4

1. **Agent Core** (`src/agents/`)
   - `BaseAgent` with ReAct-style reasoning loop
   - Tool execution integration via ToolRegistry
   - Step-by-step reasoning with configurable max steps
   - Memory integration for conversation context

### Architecture Decisions

- **No LangChain** — Everything built from primitives for full control
- **Async-first** — All LLM and memory calls are async
- **Provider-agnostic** — ToolSchema converts to any provider format
- **Subprocess sandbox** — Code runs in isolated process, not in-process
- **Composable tools** — Add a new tool by subclassing BaseTool and registering it
- **Pluggable memory** — Agents work with any BaseMemory implementation
- **Lazy chromadb import** — VectorMemory imports chromadb inside `__init__` to avoid breakage on platforms where chromadb has issues (e.g., Python 3.14 pydantic v1 conflict)
- **Content-based dedup** — CompositeMemory deduplicates by both message ID and (role, content) to handle the case where different backends create different IDs for the same logical message
