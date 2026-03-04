# Deployment Guide

## Local Development

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Add your API keys to .env

# Run
make run-api
# API available at http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

## Docker

### Single Container

```bash
cd docker
docker build -t ai-agent-framework -f Dockerfile ..
docker run -p 8000:8000 --env-file ../.env ai-agent-framework
```

### Docker Compose (API + ChromaDB)

```bash
cd docker
docker-compose up -d

# API:      http://localhost:8000
# ChromaDB: http://localhost:8001
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `DEFAULT_LLM_PROVIDER` | `claude` | `claude` or `openai` |
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Model identifier |
| `MAX_AGENT_STEPS` | `10` | Max reasoning steps |
| `MEMORY_TYPE` | `conversation` | `conversation`, `summary`, `vector` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `API_HOST` | `0.0.0.0` | Server bind host |
| `API_PORT` | `8000` | Server bind port |

## Cloud Deployment

### Fly.io

```bash
fly launch --name ai-agent-api
fly secrets set ANTHROPIC_API_KEY=sk-ant-xxx
fly deploy
```

### Railway

1. Connect your GitHub repository
2. Set environment variables in the dashboard
3. Railway auto-detects the Dockerfile

### AWS ECS / GCP Cloud Run

Use the Docker image from the CD pipeline:

```
ghcr.io/your-username/ai-agent-framework:latest
```

## Health Checks

The API exposes a `/health` endpoint that returns:

```json
{"status": "ok", "version": "0.1.0", "agents": 5, "tools": 7}
```

Docker's `HEALTHCHECK` instruction is configured in the Dockerfile.

## Monitoring

Structured logs (JSON mode) are enabled with `LOG_JSON=true`:

```bash
LOG_JSON=true uvicorn src.api.main:app --host 0.0.0.0
```

Logs include request traces, agent steps, tool executions, and token usage — ready for any log aggregator (Datadog, CloudWatch, ELK).
