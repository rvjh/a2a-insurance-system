# Extending the System

## Adding a new agent

Say you want a `Billing Agent` that handles invoicing.

### 1. Add MCP tools

`mcp_tools/billing_tools.py`:
```python
from mcp_tools.base import MCPTool

class InvoiceLookupTool(MCPTool):
    name = "invoice_lookup"
    description = "Get an invoice by invoice_id."
    input_schema = {
        "type": "object",
        "properties": {"invoice_id": {"type": "string"}},
        "required": ["invoice_id"],
    }
    def execute(self, *, invoice_id: str):
        # ... real DB call
        return {...}
```

### 2. Create the agent module

`agents/billing_agent/main.py` — copy from `claims_agent/main.py`, change:
- `name="billing-agent"`
- system prompt
- registered tools
- skills

### 3. Register with the gateway

In `config/settings.py`:
```python
billing_agent_url: str = "http://localhost:8003"
```

In `gateway/main.py` lifespan, add `settings.billing_agent_url` to the `targets` list.

That's it. The gateway auto-discovers via the agent card on next restart.

### 4. Update routing keywords

In `gateway/router.py`, add:
```python
if any(w in text_lc for w in ("invoice", "bill", "payment")):
    if card.name == "billing-agent":
        score += 3
```

Or rely on the LLM router fallback — it'll pick correctly if the agent description is clear.

---

## Adding a new MCP tool to an existing agent

1. Subclass `MCPTool` in the appropriate `mcp_tools/*.py` file.
2. Register in the agent's `build_*_agent()` function:
   ```python
   registry.register(MyNewTool())
   ```
3. The agent's system prompt may need updating to mention when to use the new tool.
4. Add a unit test in `tests/test_units.py`.

---

## Adding a new A2A method

Say you want `message/stream` for SSE streaming.

### Agent side (`agents/base_agent.py`)

```python
from sse_starlette.sse import EventSourceResponse

# Inside build_app(), add a new endpoint or extend the JSON-RPC dispatcher:
elif rpc_req.method == "message/stream":
    # Return EventSourceResponse with chunks of Task updates
    ...
```

### Gateway side (`gateway/main.py`)

The gateway needs to proxy SSE — use `httpx.stream()` and a passthrough:
```python
async def stream_passthrough(...):
    async with http_client.stream("POST", agent_url, json=...) as r:
        async for chunk in r.aiter_bytes():
            yield chunk
```

---

## Multi-agent orchestration (when one user request needs both agents)

Right now the gateway routes to ONE agent. For workflows like "validate this claim AND tell me what was covered", you need an **orchestrator**.

### Option A: LangGraph supervisor pattern

A meta-agent that owns a graph with TWO nodes (one per downstream agent), calling them in sequence/parallel and merging results.

```python
from langgraph.graph import StateGraph

class OrchestratorState(TypedDict):
    user_request: str
    claims_result: dict
    policy_result: dict
    final_answer: str

graph = StateGraph(OrchestratorState)
graph.add_node("call_claims", lambda s: {"claims_result": call_agent("claims-agent", s["user_request"])})
graph.add_node("call_policy", lambda s: {"policy_result": call_agent("policy-agent", s["user_request"])})
graph.add_node("synthesize", lambda s: {"final_answer": synthesize(s)})
graph.add_edge("call_claims", "call_policy")
graph.add_edge("call_policy", "synthesize")
```

Deploy this orchestrator as a third agent at `:8003`. Clients hit it like any other agent.

### Option B: CrewAI

If you prefer role-based collaboration:

```python
from crewai import Agent, Task, Crew

claims_role = Agent(role="Claims Validator", goal="...", backstory="...", tools=[...])
policy_role = Agent(role="Policy Analyst", goal="...", backstory="...", tools=[...])

crew = Crew(agents=[claims_role, policy_role], tasks=[...], process=Process.sequential)
result = crew.kickoff(inputs={"claim_id": "CLM-1002"})
```

CrewAI handles message-passing between agents automatically. Good when you want declarative role definitions; less control than LangGraph.

### Recommendation
Start with LangGraph for orchestration in your project. CrewAI is a higher-level abstraction; switching to it later is straightforward if you outgrow LangGraph's verbosity.

---

## Production hardening checklist

When moving from this scaffold to production:

- [ ] Replace in-memory task store with Redis or Postgres (see architecture.md §10)
- [ ] Add JWT auth instead of static bearer (use `python-jose`)
- [ ] Add `slowapi` rate limiting on the gateway
- [ ] Add mTLS between gateway and agents
- [ ] Run agents behind a real ASGI server (gunicorn + uvicorn workers, or hypercorn)
- [ ] Move secrets to AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault
- [ ] Set up CI to run `make test` + `make lint` on every PR
- [ ] Add Sentry / Rollbar for error tracking
- [ ] Set up Langfuse self-hosted (or accept SaaS)
- [ ] Define an SLO: e.g., 99% of requests < 8s, 99.5% success rate
- [ ] Write runbooks: "What if claims-agent fails?", "What if Anthropic API is down?"
- [ ] Add a fallback mode: if LLM fails, return a graceful "service degraded" response

---

## Swapping LLM providers

Our `LLMClient` is Anthropic-specific. To support other providers:

1. Define an abstract `LLMClient` interface (just `complete()`).
2. Provide concrete implementations: `AnthropicLLMClient`, `OpenAILLMClient`, `BedrockLLMClient`.
3. Inject via DI in `build_*_agent()`.

LangChain has provider-agnostic abstractions (`BaseChatModel`) if you'd rather rely on it. Trade-off: less control over tool-use semantics, more abstraction overhead.

---

## CI / CD example (GitHub Actions)

`.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: mypy --ignore-missing-imports common gateway agents mcp_tools
      - run: pytest tests/test_units.py -v
```
