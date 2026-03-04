.PHONY: install dev test test-unit test-integration test-cov lint lint-fix format format-check typecheck clean run-api docker-build docker-up docker-down

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	cd docker && docker build -t ai-agent-framework -f Dockerfile ..

docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down
