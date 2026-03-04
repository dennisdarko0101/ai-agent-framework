# Architecture

## Overview

The AI Agent Framework is built from primitives — no LangChain or similar meta-frameworks. Every layer is async-first, provider-agnostic, and composable.

```
┌──────────────────────────────────────────────────┐
│              REST API + WebSocket                 │
│         FastAPI with real-time streaming           │
├──────────────────────────────────────────────────┤
│                 Orchestration                     │
│      Pipeline  ·  TaskRouter  ·  Supervisor       │
├──────────────────────────────────────────────────┤
│                    Agents                         │
│  ReActAgent · Planner · Researcher · Coder · Rev. │
├─────────────┬─────────────┬──────────────────────┤
│  LLM Layer  │ Tool System │   Memory System       │
│ Claude/GPT  │  Registry   │ Conv/Summary/Vector   │
├─────────────┴─────────────┴──────────────────────┤
│        Pydantic Settings  ·  structlog            │
└──────────────────────────────────────────────────┘
```

## Layer Responsibilities

### API Layer (`src/api/`)

The top layer exposes the framework over HTTP and WebSocket.

- `main.py` — FastAPI app with CORS, lifespan startup, OpenAPI docs
- `routes.py` — REST endpoints for task execution, planning, research, code generation, pipelines, and introspection
- `websocket.py` — WebSocket `/ws/run` endpoint for real-time agent streaming
- `schemas.py` — Pydantic models for all request/response types
- `dependencies.py` — Factory functions for agents, tools, memory, LLM providers

#### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health with agent/tool counts |
| `/api/v1/run` | POST | Execute a task with any agent type |
| `/api/v1/plan` | POST | Create a structured plan |
| `/api/v1/research` | POST | Research a topic |
| `/api/v1/code` | POST | Generate code |
| `/api/v1/pipeline` | POST | Run a multi-agent pipeline |
| `/api/v1/agents` | GET | List available agent types |
| `/api/v1/tools` | GET | List registered tools |

#### WebSocket Streaming

The `/ws/run` endpoint accepts a JSON message and streams `StreamEvent` objects as the agent reasons:

```
Client → {"task": "...", "agent_type": "react", "max_steps": 10}
Server ← {"event_type": "thought", "data": "...", "step_number": 1, "timestamp": ...}
Server ← {"event_type": "action", "data": "...", "step_number": 1, "timestamp": ...}
Server ← {"event_type": "observation", "data": "...", "step_number": 1, "timestamp": ...}
Server ← {"event_type": "result", "data": "final answer", "step_number": 3, "timestamp": ...}
```

Event types: `thought`, `action`, `observation`, `result`, `error`

### Configuration (`src/config/`)

- `Settings` — Pydantic `BaseSettings` with `.env` support
- `get_settings()` — Cached singleton
- All API keys, model defaults, limits, and feature flags

### LLM Layer (`src/llm/`)

- `BaseLLMProvider` — Abstract async provider
- `ClaudeProvider` / `OpenAIProvider` — Concrete implementations
- `ProviderFactory.create()` — Instantiate from settings
- `ToolSchema` — Provider-agnostic tool definition
- Format conversion: `convert_to_anthropic_tools()`, `convert_to_openai_tools()`
- Tool call parsing: `parse_anthropic_tool_calls()`, `parse_openai_tool_calls()`
- Multi-turn support: `build_assistant_message()`, `build_tool_result_messages()`

### Tool System (`src/tools/`)

- `BaseTool` / `AsyncTool` — Sync and async tool abstractions
- `ToolResult` — Standardised execution output
- `ToolRegistry` — Singleton catalogue with auto-discovery
- Built-in tools: Calculator, FileRead/Write, CodeExecutor, WebSearch, APICaller, Summarize, Extract

### Memory System (`src/memory/`)

- `BaseMemory` — Async ABC for all backends
- `Message` — Dataclass with role, content, metadata, timestamp, ID
- `ConversationMemory` — Sliding window with FIFO eviction
- `SummaryMemory` — LLM-compressed history + recent messages
- `VectorMemory` — ChromaDB semantic storage/retrieval
- `CompositeMemory` — Merges and deduplicates multiple backends

### Agents (`src/agents/`)

- `BaseAgent` — ABC with `run(task, event_callback) -> AgentResponse`
- `ReActAgent` — Core Reason → Act → Observe loop
- `PlannerAgent` — Task decomposition specialist
- `ResearchAgent` — Information gathering specialist
- `CoderAgent` — Code writing and debugging specialist
- `ReviewerAgent` — Quality assessment specialist

### Orchestration (`src/orchestration/`)

- `AgentPipeline` — Sequential chaining (output of N → input of N+1)
- `TaskRouter` — LLM classification → route to specialist
- `AgentSupervisor` — Iterative delegation with DELEGATE/DONE protocol

## Key Design Decisions

### No LangChain

Everything is built from primitives for full control over the reasoning loop, tool execution, and message formatting. This avoids version-lock, abstraction leaks, and hidden complexity.

### Async-First

All LLM calls, memory operations, and tool executions are async. This enables concurrent tool execution and non-blocking I/O without threading.

### Provider-Agnostic

`ToolSchema` defines tools once; the framework converts to Anthropic or OpenAI format automatically. `build_assistant_message()` and `build_tool_result_messages()` on each provider handle the different multi-turn message formats.

### Event Callback Streaming

Agents accept an optional `event_callback` parameter on `run()`. The WebSocket handler creates a callback that sends events over the connection. This decouples agents from any specific transport — the same agent works headless, over REST, or streaming via WebSocket.

### Composable Architecture

Every component (agent, tool, memory, orchestration pattern) can be mixed and matched freely:

```python
# Any agent + any memory + any tools
agent = ReActAgent(llm=claude, tools=registry, memory=vector_memory)

# Any agents in a pipeline
pipe = AgentPipeline([planner, coder, reviewer])

# Any agents in a supervisor
sup = AgentSupervisor(llm=claude, agents={"plan": planner, "code": coder})
```

### Subprocess Sandbox

Code execution runs in an isolated subprocess with AST-based import blocking and configurable memory/time limits. The agent process is never at risk.

### Lazy Heavy Dependencies

ChromaDB is imported lazily inside `VectorMemory.__init__` to avoid breaking environments where it isn't installed (e.g., pydantic v1 conflict on Python 3.14).

### Dependency Injection for Testing

API tests mock the LLM provider via `patch("src.api.dependencies.ProviderFactory")`, avoiding real API calls while exercising the full request/response path.

## Data Flow

### Single Agent Run

```
User task
  → Memory.get_context() → prior messages
  → LLM.generate(messages + tools) → LLMResponse
  → If tool_calls: ToolRegistry.execute() → ToolResult
    → build_tool_result_messages() → append to messages
    → loop back to LLM.generate()
  → If no tool_calls: return AgentResponse
  → Memory.add(assistant, response)
```

### REST API Request

```
POST /api/v1/run {"task": "...", "agent_type": "react"}
  → dependencies.create_agent() → ReActAgent
  → agent.run(task) → AgentResponse
  → routes._to_task_response() → TaskResponse JSON
```

### WebSocket Streaming

```
WS /ws/run → accept connection
  → receive JSON {"task": "...", "agent_type": "react"}
  → create agent with event_callback
  → agent.run(task, event_callback=on_event)
    → on_event("thought", ...) → send StreamEvent
    → on_event("action", ...) → send StreamEvent
    → on_event("observation", ...) → send StreamEvent
  → send final "result" event
```

### Pipeline

```
Task → Agent₁.run() → output₁ → Agent₂.run(output₁) → output₂ → ... → final
```

### Supervisor

```
Task → Supervisor LLM → "DELEGATE coder: write X"
  → coder.run("write X") → result
  → Supervisor LLM (with result) → "DELEGATE reviewer: review X"
  → reviewer.run("review X") → result
  → Supervisor LLM → "DONE: final answer"
```

## Directory Structure

```
src/
├── agents/
│   ├── base.py          # BaseAgent, AgentResponse, AgentStep, AgentAction
│   ├── react.py         # ReActAgent
│   ├── planner.py       # PlannerAgent
│   ├── researcher.py    # ResearchAgent
│   ├── coder.py         # CoderAgent
│   └── reviewer.py      # ReviewerAgent
├── api/
│   ├── schemas.py       # Pydantic request/response models
│   ├── dependencies.py  # Agent/tool/memory factory functions
│   ├── routes.py        # REST endpoints
│   ├── websocket.py     # WebSocket streaming endpoint
│   └── main.py          # FastAPI app entry point
├── config/
│   └── settings.py      # Pydantic Settings
├── llm/
│   ├── function_calling.py  # ToolSchema, format conversion, parsing
│   └── provider.py          # BaseLLMProvider, Claude, OpenAI, Factory
├── memory/
│   ├── base.py          # BaseMemory, Message, estimate_tokens
│   ├── conversation.py  # ConversationMemory
│   ├── summary.py       # SummaryMemory
│   ├── vector_memory.py # VectorMemory (ChromaDB)
│   └── composite.py     # CompositeMemory
├── orchestration/
│   ├── pipeline.py      # AgentPipeline
│   ├── router.py        # TaskRouter
│   └── supervisor.py    # AgentSupervisor
├── tools/
│   ├── base.py          # BaseTool, AsyncTool, ToolResult
│   ├── registry.py      # ToolRegistry
│   ├── calculator.py    # Calculator
│   ├── file_ops.py      # FileRead, FileWrite
│   ├── code_executor.py # CodeExecutor
│   ├── web_search.py    # WebSearch
│   ├── api_caller.py    # APICaller
│   └── text_tools.py    # Summarize, Extract
└── utils/
    ├── logger.py        # structlog setup
    └── sandbox.py       # subprocess sandboxing

tests/
├── conftest.py
├── unit/            # 241 unit tests
└── integration/     # 38 API + WebSocket tests

examples/            # Research assistant, code assistant, data analyst
docker/              # Dockerfile + docker-compose
docs/                # Architecture, deployment, guides
```
