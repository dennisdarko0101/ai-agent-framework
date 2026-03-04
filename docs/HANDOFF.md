# Project Handoff — AI Agent Framework

## Current Status: Phase 3, Step 9 Complete — All Phases Done

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
- `BaseAgent` ABC: `run(task, event_callback) -> AgentResponse`
- `ReActAgent` — Full Reason → Act → Observe loop with native tool calling
  - Loads memory context before each run
  - Passes tool schemas to LLM for native function calling
  - Executes tool calls via ToolRegistry, feeds results back
  - Tracks steps, tokens, and timing
  - Max steps safety cap
  - Event callback for real-time streaming (thought, action, observation, result, error)
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

**Phase 3, Step 7 — FastAPI + WebSocket Server** (`src/api/`)

- `src/api/schemas.py` — Pydantic request/response models (TaskRequest, PipelineRequest, ResearchRequest, CodeRequest, TaskResponse, StreamEvent, HealthResponse, AgentListResponse, ToolListResponse)
- `src/api/dependencies.py` — Agent creation helpers, tool registration, memory factory
- `src/api/routes.py` — REST endpoints: `/api/v1/run`, `/plan`, `/research`, `/code`, `/pipeline`, `/agents`, `/tools`, `/health`
- `src/api/websocket.py` — WebSocket `/ws/run` endpoint with real-time event streaming
- `src/api/main.py` — FastAPI app with CORS, lifespan, OpenAPI docs

**Phase 3, Step 8 — Example Applications** (`examples/`)

- `examples/research_assistant.py` — Pipeline: PlannerAgent → ResearchAgent → ReviewerAgent
- `examples/code_assistant.py` — Supervisor: CoderAgent + ReviewerAgent with iterative delegation
- `examples/data_analyst.py` — ReActAgent with file read and code executor tools

**Phase 3, Step 9 — Production Readiness**

- `docker/Dockerfile` — Multi-stage build, non-root user, healthcheck
- `docker/docker-compose.yml` — API + ChromaDB services with volumes
- `.github/workflows/ci.yml` — Lint + test matrix (Python 3.11/3.12), coverage upload
- `.github/workflows/cd.yml` — Docker build + push to ghcr.io on tag push
- `README.md` — Full portfolio-ready README with badges, architecture, features, API reference, examples
- `CONTRIBUTING.md` — Setup, workflow, code style, testing, PR checklist
- `docs/DEPLOYMENT.md` — Local, Docker, cloud deployment guide
- `Makefile` — All targets: install, dev, test, lint, format, docker, clean

**Tests** (169 + 72 + 38 = 279 total)

Phase 1: 169 tests
- `test_config.py` — Settings, environment variable loading
- `test_llm.py` — Providers, function calling, tool schema conversion
- `test_tools.py` — All built-in tools, registry, discovery
- `test_memory.py` — All memory backends, composite, token estimation

Phase 2: 72 tests
- `test_react_agent.py` — 29 tests: AgentAction, AgentStep, AgentResponse, BaseAgent, ReActAgent
- `test_specialized_agents.py` — 18 tests: all 4 specialised agents
- `test_orchestration.py` — 25 tests: Pipeline, Router, Supervisor

Phase 3: 38 tests
- `test_schemas.py` — 15 tests: all Pydantic request/response models
- `test_api.py` — 15 tests: all REST endpoints with mocked LLM
- `test_websocket.py` — 8 tests: WebSocket streaming with mocked LLM

### Architecture Decisions

- **No LangChain** — Everything built from primitives for full control
- **Async-first** — All LLM and memory calls are async
- **Provider-agnostic** — ToolSchema converts to any provider format
- **Subprocess sandbox** — Code runs in isolated process, not in-process
- **Composable tools** — Add a new tool by subclassing BaseTool and registering it
- **Pluggable memory** — Agents work with any BaseMemory implementation
- **Composable orchestration** — Pipeline, Router, Supervisor mix and match freely
- **Lazy chromadb import** — VectorMemory imports chromadb inside `__init__` to avoid breakage on platforms where chromadb has issues
- **Content-based dedup** — CompositeMemory deduplicates by both message ID and (role, content)
- **Provider message building** — Each LLM provider formats multi-turn tool-use messages in its native format (Anthropic content blocks vs. OpenAI tool messages)
- **DELEGATE/DONE protocol** — Supervisor uses explicit text signals for delegation decisions
- **Event callback streaming** — Optional `event_callback` on `run()` enables WebSocket streaming without coupling agents to transport
- **Lifespan pattern** — FastAPI uses `@asynccontextmanager` lifespan instead of deprecated `on_event`
- **Dependency injection for tests** — `patch("src.api.dependencies.ProviderFactory")` mocks LLM in API tests

### File Inventory

```
src/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py              # BaseAgent, AgentResponse, AgentStep, AgentAction
│   ├── react.py             # ReActAgent
│   ├── planner.py           # PlannerAgent
│   ├── researcher.py        # ResearchAgent
│   ├── coder.py             # CoderAgent
│   └── reviewer.py          # ReviewerAgent
├── api/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic request/response models
│   ├── dependencies.py      # Agent/tool/memory factory functions
│   ├── routes.py            # REST endpoints
│   ├── websocket.py         # WebSocket streaming endpoint
│   └── main.py              # FastAPI app entry point
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic Settings
├── llm/
│   ├── __init__.py
│   ├── function_calling.py  # ToolSchema, format conversion, parsing
│   └── provider.py          # BaseLLMProvider, Claude, OpenAI, Factory
├── memory/
│   ├── __init__.py
│   ├── base.py              # BaseMemory, Message, estimate_tokens
│   ├── conversation.py      # ConversationMemory
│   ├── summary.py           # SummaryMemory
│   ├── vector_memory.py     # VectorMemory (ChromaDB)
│   └── composite.py         # CompositeMemory
├── orchestration/
│   ├── __init__.py
│   ├── pipeline.py          # AgentPipeline
│   ├── router.py            # TaskRouter
│   └── supervisor.py        # AgentSupervisor
├── tools/
│   ├── __init__.py
│   ├── base.py              # BaseTool, AsyncTool, ToolResult
│   ├── registry.py          # ToolRegistry
│   ├── calculator.py
│   ├── file_ops.py
│   ├── code_executor.py
│   ├── web_search.py
│   ├── api_caller.py
│   └── text_tools.py
└── utils/
    ├── __init__.py
    ├── logger.py            # structlog setup
    └── sandbox.py           # subprocess sandboxing

tests/
├── conftest.py
├── unit/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_llm.py
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_react_agent.py
│   ├── test_specialized_agents.py
│   ├── test_orchestration.py
│   └── test_schemas.py
└── integration/
    ├── __init__.py
    ├── test_api.py
    └── test_websocket.py

examples/
├── research_assistant.py
├── code_assistant.py
└── data_analyst.py

docker/
├── Dockerfile
└── docker-compose.yml

docs/
├── ARCHITECTURE.md
├── AGENTS_GUIDE.md
├── TOOLS_GUIDE.md
├── DEPLOYMENT.md
└── HANDOFF.md

.github/workflows/
├── ci.yml
└── cd.yml
```
