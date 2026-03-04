# Project Handoff — AI Agent Framework

## Current Status: Phase 2, Step 6 Complete

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

**Phase 2, Step 4 — Agent Core** (`src/agents/`)

- `AgentAction` dataclass: `tool_name`, `arguments`, `call_id`, `result_output`, `result_error`, `success`
- `AgentStep` dataclass: `step_number`, `reasoning`, `actions`, `observation`, `timestamp`
- `AgentResponse` dataclass: `output`, `steps`, `success`, `error`, `total_tokens`, `execution_time`, `metadata`
- `BaseAgent` ABC: `run(task) -> AgentResponse`, `_build_system_prompt()`
- `ReActAgent` — Full Reason → Act → Observe loop with native tool calling
  - Loads memory context before each run
  - Passes tool schemas to LLM for native function calling
  - Executes tool calls via ToolRegistry, feeds results back
  - Tracks steps, tokens, and timing
  - Max steps safety cap
- Provider message building: `build_assistant_message()`, `build_tool_result_messages()` on BaseLLMProvider, ClaudeProvider, and OpenAIProvider for multi-turn tool-use conversations

**Phase 2, Step 5 — Specialised Agents** (`src/agents/`)

- `PlannerAgent(ReActAgent)` — Task decomposition, `create_plan()` convenience method
- `ResearchAgent(ReActAgent)` — Information gathering, `research(topic, depth)` method
- `CoderAgent(ReActAgent)` — Code writing/debugging, `write_code()` and `debug_code()` methods
- `ReviewerAgent(ReActAgent)` — Quality assessment, `review(content, criteria)` method
- All have domain-specific default system prompts, customisable via constructor

**Phase 2, Step 6 — Multi-Agent Orchestration** (`src/orchestration/`)

- `AgentPipeline` — Sequential chaining: output of stage N → input of stage N+1
  - Stops on failure, tracks per-stage outputs in metadata
  - Accumulates tokens and timing across all stages
- `TaskRouter` — LLM-based task classification and routing
  - Routes to named agents based on classification
  - Optional default agent for unclassifiable tasks
  - Case-insensitive category matching
- `AgentSupervisor` — Iterative delegation and synthesis
  - DELEGATE/DONE protocol for supervisor LLM decisions
  - Handles unknown agent names gracefully
  - Tracks rounds and delegations in metadata

**Tests** (169 prior + 72 new = 241 tests)
- `test_react_agent.py` — 29 tests: AgentAction, AgentStep, AgentResponse, BaseAgent, ReActAgent (no tools, with tools, memory, provider message building)
- `test_specialized_agents.py` — 18 tests: PlannerAgent, ResearchAgent, CoderAgent, ReviewerAgent (defaults, custom prompts, convenience methods)
- `test_orchestration.py` — 25 tests: AgentPipeline (single/multi-stage, failure, tokens), TaskRouter (routing, fallback, errors), AgentSupervisor (delegation, DONE, max rounds, multi-agent)

**Documentation**
- `docs/AGENTS_GUIDE.md` — Agents, orchestration, memory, configuration
- `docs/ARCHITECTURE.md` — Full architecture overview, data flow, directory structure
- `docs/TOOLS_GUIDE.md` — Custom tool creation, built-in tools reference
- `docs/HANDOFF.md` — Project status

### Next Steps — Phase 3

1. **API Server** (`src/api/`)
   - FastAPI server with agent endpoints
   - Streaming support for ReAct steps
   - Session management

2. **CLI Interface** (`src/cli/`)
   - Interactive chat mode
   - Agent selection and configuration

3. **Integration Testing**
   - End-to-end tests with real LLM providers
   - Multi-agent workflow tests

### Architecture Decisions

- **No LangChain** — Everything built from primitives for full control
- **Async-first** — All LLM and memory calls are async
- **Provider-agnostic** — ToolSchema converts to any provider format
- **Subprocess sandbox** — Code runs in isolated process, not in-process
- **Composable tools** — Add a new tool by subclassing BaseTool and registering it
- **Pluggable memory** — Agents work with any BaseMemory implementation
- **Composable orchestration** — Pipeline, Router, Supervisor mix and match freely
- **Lazy chromadb import** — VectorMemory imports chromadb inside `__init__` to avoid breakage on platforms where chromadb has issues
- **Content-based dedup** — CompositeMemory deduplicates by both message ID and (role, content) to handle the case where different backends create different IDs for the same logical message
- **Provider message building** — Each LLM provider knows how to format multi-turn tool-use messages in its native format (Anthropic content blocks vs. OpenAI tool messages)
- **DELEGATE/DONE protocol** — Supervisor uses explicit text signals for delegation decisions, keeping the protocol simple and debuggable
