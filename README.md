# AI Agent Framework

[![CI](https://github.com/your-username/ai-agent-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ai-agent-framework/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production multi-agent framework built from scratch — no LangChain, no abstractions you can't see through. ReAct reasoning, native tool calling, pluggable memory, multi-agent orchestration, and real-time WebSocket streaming.

## Architecture

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

## Features

- **ReAct Reasoning** — Step-by-step Reason → Act → Observe loop with native tool calling
- **Multi-Agent Orchestration** — Pipeline (sequential), Router (classification), Supervisor (delegation)
- **5 Specialised Agents** — Planner, Researcher, Coder, Reviewer + general ReAct
- **8 Built-in Tools** — Calculator, file I/O, code executor, web search, API caller, summarise, extract
- **Pluggable Memory** — Conversation buffer, LLM summaries, ChromaDB vector search, composite
- **Provider-Agnostic** — Claude and OpenAI with identical interfaces, add more in minutes
- **Real-time Streaming** — WebSocket endpoint streams thoughts, actions, and observations live
- **REST API** — FastAPI with OpenAPI docs, versioned endpoints, CORS
- **Production-Ready** — Docker, CI/CD, structured logging, subprocess sandboxing

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/ai-agent-framework.git
cd ai-agent-framework
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest tests/ -v

# Start the API server
make run-api
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/v1/run` | POST | Execute a task with any agent |
| `/api/v1/plan` | POST | Create a structured plan |
| `/api/v1/research` | POST | Research a topic |
| `/api/v1/code` | POST | Generate code |
| `/api/v1/pipeline` | POST | Run a multi-agent pipeline |
| `/api/v1/agents` | GET | List available agents |
| `/api/v1/tools` | GET | List available tools |
| `/ws/run` | WS | Real-time agent streaming |

### Execute a Task

```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is 2+2?", "agent_type": "react"}'
```

### Multi-Agent Pipeline

```bash
curl -X POST http://localhost:8000/api/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Build a sorting algorithm",
    "stages": [
      {"agent_type": "planner"},
      {"agent_type": "coder"},
      {"agent_type": "reviewer"}
    ]
  }'
```

### WebSocket Streaming

```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/run") as ws:
        await ws.send(json.dumps({"task": "Explain quantum computing"}))
        async for message in ws:
            event = json.loads(message)
            print(f"[{event['event_type']}] {event['data']}")
            if event["event_type"] in ("result", "error"):
                break

asyncio.run(main())
```

## Usage Examples

### Single Agent

```python
import asyncio
from src.agents.react import ReActAgent
from src.llm.provider import ProviderFactory
from src.tools.registry import ToolRegistry
from src.tools.calculator import CalculatorTool

async def main():
    llm = ProviderFactory.create()
    tools = ToolRegistry()
    tools.register(CalculatorTool())

    agent = ReActAgent(llm=llm, tools=tools)
    result = await agent.run("What is 15% of 200?")
    print(result.output)

asyncio.run(main())
```

### Pipeline Orchestration

```python
from src.agents.planner import PlannerAgent
from src.agents.coder import CoderAgent
from src.agents.reviewer import ReviewerAgent
from src.orchestration.pipeline import AgentPipeline

pipe = AgentPipeline(agents=[
    PlannerAgent(llm=llm),
    CoderAgent(llm=llm, tools=tools),
    ReviewerAgent(llm=llm),
])
result = await pipe.run("Build a REST API with auth")
```

### Supervisor Orchestration

```python
from src.orchestration.supervisor import AgentSupervisor

supervisor = AgentSupervisor(
    llm=llm,
    agents={"coder": coder, "reviewer": reviewer},
    max_rounds=5,
)
result = await supervisor.run("Write and review a binary search")
```

## Adding Custom Tools

```python
from src.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input text"},
        },
        "required": ["input"],
    }

    def execute(self, arguments):
        return ToolResult(output=f"Processed: {arguments['input']}")

# Register it
registry.register(MyTool())
```

## Adding Custom Agents

```python
from src.agents.react import ReActAgent

class DomainExpert(ReActAgent):
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            system_prompt="You are an expert in X. Always verify claims with tools.",
            name="domain-expert",
            max_steps=15,
            **kwargs,
        )
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| LLM Providers | Anthropic Claude, OpenAI GPT |
| API Server | FastAPI + Uvicorn |
| Streaming | WebSockets |
| Vector Store | ChromaDB |
| Configuration | Pydantic Settings + dotenv |
| Logging | structlog |
| Testing | pytest + pytest-asyncio |
| Linting | ruff |
| CI/CD | GitHub Actions |
| Container | Docker |

## Project Structure

```
src/
├── agents/          # ReAct + specialised agents
├── api/             # FastAPI REST + WebSocket server
├── config/          # Pydantic Settings
├── llm/             # Provider abstraction (Claude, OpenAI)
├── memory/          # Conversation, summary, vector, composite
├── orchestration/   # Pipeline, router, supervisor
├── tools/           # Built-in tools + registry
└── utils/           # Logging, sandboxing

tests/
├── unit/            # 241 unit tests
└── integration/     # 38 API + WebSocket tests

examples/            # Research assistant, code assistant, data analyst
docker/              # Dockerfile + docker-compose
docs/                # Architecture, deployment, guides
```

## Development

```bash
make dev          # Install with dev dependencies
make test         # Run all tests
make test-cov     # Tests with coverage report
make lint         # Run ruff linter
make format       # Auto-format code
make run-api      # Start the API server
```

## License

[MIT](LICENSE)
