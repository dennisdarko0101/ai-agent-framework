# Contributing

## Setup

```bash
git clone https://github.com/your-username/ai-agent-framework.git
cd ai-agent-framework
pip install -e ".[dev]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/my-feature`
2. Make changes
3. Run checks: `make lint && make test`
4. Commit and push
5. Open a pull request

## Code Style

- **Linter:** ruff (run `make lint`)
- **Formatter:** ruff format (run `make format`)
- **Line length:** 100 characters
- **Type hints:** Required on all public functions
- **Imports:** `from __future__ import annotations` in every file
- **Logging:** `structlog` via `get_logger(__name__)`

## Testing

- All tests in `tests/unit/` and `tests/integration/`
- Use `pytest` with `@pytest.mark.asyncio` for async tests
- Mock LLM providers with `AsyncMock()` — never call real APIs in tests
- Target: 80%+ coverage

```bash
make test         # All tests
make test-unit    # Unit only
make test-cov     # With coverage report
```

## Adding a Tool

1. Create `src/tools/my_tool.py`
2. Subclass `BaseTool` or `AsyncTool`
3. Set `name`, `description`, `parameters`
4. Implement `execute()` or `execute_async()`
5. Add tests in `tests/unit/test_tools.py`

## Adding an Agent

1. Create `src/agents/my_agent.py`
2. Subclass `ReActAgent`
3. Set a default system prompt
4. Add convenience methods
5. Add tests in `tests/unit/`
6. Register in `src/api/dependencies.py`

## Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] New code has tests
- [ ] Docs updated if needed
