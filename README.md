# A2A Insurance Multi-Agent System

A learning-grade, production-styled implementation of Google's **Agent-to-Agent (A2A) protocol** over **JSON-RPC 2.0**, with a routing **gateway**, two specialist **agents**, **MCP tools**, and an observability/safety/reporting stack.

**Domain:** Health insurance claims processing.

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 2. Run all three services (in separate terminals OR via `make run`)
uvicorn agents.claims_agent.main:app --port 8001
uvicorn agents.policy_agent.main:app --port 8002
uvicorn gateway.main:app --port 8000

# 3. Try it
python -m tests.client_demo
```

Or with Docker:
```bash
docker-compose up --build
```

## Architecture

```
Client ──JSON-RPC──> Gateway (8000) ──JSON-RPC──> Claims Agent (8001) ──> [claim_lookup, fraud_check]
                       │                          
                       └─────────────────────────> Policy Agent (8002) ──> [policy_lookup, coverage_check]
```

## Project Layout

```
a2a-insurance-system/
├── common/            # Shared schemas, logging, observability, LLM client, guardrails
├── config/            # Pydantic settings (env-driven)
├── gateway/           # Routing gateway (FastAPI)
├── agents/
│   ├── base_agent.py  # A2A protocol implementation + LangGraph loop
│   ├── claims_agent/  # Validates claims, runs fraud checks
│   └── policy_agent/  # Answers coverage questions
├── mcp_tools/         # Tool implementations (MCP-compatible shape)
├── reports/           # HTML run-report generator
├── tests/             # Unit tests + end-to-end demo client
├── docs/              # Design notes & decision rationale
└── docker-compose.yml
```

## API Examples

### Direct A2A call (skip routing) → claims agent
```bash
curl -X POST http://localhost:8000/ \
  -H "Authorization: Bearer dev-local-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "1",
    "params": {
      "agent": "claims-agent",
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Validate claim CLM-1002"}]
      }
    }
  }'
```

### Auto-routed call → gateway picks the agent
```bash
curl -X POST http://localhost:8000/ \
  -H "Authorization: Bearer dev-local-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "2",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Is procedure 99213 covered under POL-555?"}]
      }
    }
  }'
```

### Discover agents
```bash
curl http://localhost:8001/.well-known/agent-card.json
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — design decisions & rationale
- [docs/observability.md](docs/observability.md) — Langfuse + LangSmith setup
- [docs/guardrails.md](docs/guardrails.md) — adding production guardrails
- [docs/extending.md](docs/extending.md) — adding agents, tools, methods

## Test
```bash
make test       # unit tests (no LLM)
make demo       # end-to-end (needs running services + ANTHROPIC_API_KEY)
make fmt        # format with black + ruff
make lint       # ruff + mypy
```
