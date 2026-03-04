# AI Agent Framework

Production multi-agent framework where AI agents use tools, maintain memory, and orchestrate complex tasks. Built from scratch — no LangChain — to demonstrate deep understanding of agent architectures.

## Features

- **Provider-agnostic LLM layer** — Claude (Anthropic) and OpenAI with unified interface
- **Native function calling** — Tool schemas convert automatically per provider
- **Sandboxed code execution** — Safe Python execution with timeout and import restrictions
- **Structured logging** — JSON and console output via structlog
- **Pydantic Settings** — Type-safe configuration from environment variables

### Planned (upcoming phases)

- Tool registry with dynamic discovery
- Conversation, summary, and vector memory backends
- Multi-agent orchestration (sequential, parallel, hierarchical)
- FastAPI server with WebSocket streaming
- Docker deployment

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd ai-agent-framework
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run tests
make test

# Lint and type check
make lint
make typecheck
```

## Project Structure

```
src/
├── config/          # Pydantic settings
├── llm/             # LLM providers and function calling
│   ├── provider.py        # Base + Claude + OpenAI providers
│   └── function_calling.py # Schema conversion, parsing, validation
├── agents/          # Agent implementations (Phase 2)
├── tools/           # Tool registry and built-in tools (Phase 2)
├── memory/          # Memory backends (Phase 2)
├── orchestration/   # Multi-agent orchestration (Phase 3)
├── api/             # FastAPI server (Phase 3)
└── utils/
    ├── logger.py    # Structured logging
    └── sandbox.py   # Sandboxed code execution
tests/
├── unit/            # Unit tests (mocked)
└── integration/     # Integration tests (require API keys)
```

## Development

```bash
make dev          # Install with dev dependencies
make test         # Run all tests
make test-unit    # Unit tests only
make test-cov     # Tests with coverage report
make lint         # Ruff linting
make format       # Auto-format with Ruff
make typecheck    # mypy type checking
```

## Architecture

The framework is built in layers:

1. **LLM Layer** — Provider abstraction with retry logic, token tracking, and native tool use
2. **Function Calling** — Provider-agnostic schema definition, format conversion, argument validation
3. **Tools** — Registry pattern with built-in tools (web search, code execution, file ops)
4. **Memory** — Pluggable backends: conversation history, summarization, vector (ChromaDB)
5. **Agents** — ReAct-style agents with configurable tools and memory
6. **Orchestration** — Multi-agent coordination patterns
7. **API** — FastAPI server with WebSocket streaming for real-time agent output

## License

MIT
