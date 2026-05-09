# Architecture & Design Rationale

This document explains **why** each piece is built the way it is. Read this top-to-bottom once; refer back when you need to justify a design choice in your project.

## 1. Why JSON-RPC 2.0 (and not REST or gRPC)?

A2A standardizes on JSON-RPC 2.0 because:

- **Single endpoint, many methods.** REST forces one URL per resource. A2A has `message/send`, `tasks/get`, `tasks/cancel`, `tasks/resubscribe`, etc. — all over `POST /`. Adding methods doesn't break URL routing.
- **Symmetric.** Agent-to-agent calls look identical to client-to-agent calls. Same envelope, same handlers.
- **Deterministic error model.** Every error has a numeric `code` and a `message`. Easier to handle than HTTP status codes that mean different things on different routes.
- **Explicit IDs.** The `id` field is the natural correlation handle for tracing.

We deliberately keep JSON-RPC errors at HTTP **200** — JSON-RPC errors are application errors, not transport errors. Only network/transport failures return non-200.

## 2. Why a Gateway in front of agents?

Your project will have these exact same needs:

- **One auth boundary.** Validating bearer tokens / JWTs at every agent duplicates code and risk. Gateway = single funnel.
- **Cross-cutting guardrails.** Input PII redaction, prompt-injection detection, rate limiting — done once at the edge.
- **Routing & discovery.** Clients shouldn't know agent URLs. They send intent; the gateway maps to capability.
- **Observability hub.** Every request gets a trace_id at the gateway and propagates to agents.
- **Future-proofing.** Add a new agent? Just deploy it; the gateway re-discovers it via `/.well-known/agent-card.json`. No client change.

The gateway in this scaffold is intentionally thin — it's a **router**, not an orchestrator. Multi-agent orchestration (where the answer requires multiple agents working together) belongs in a dedicated orchestrator agent above the gateway, or you push that logic into a LangGraph supervisor. We mention this in `docs/extending.md`.

## 3. Why Agent Cards at `/.well-known/agent-card.json`?

This is the A2A spec convention. The benefits:

- **Self-describing services.** Anyone (gateway, other agents, monitoring tools) can discover capabilities without out-of-band docs.
- **Dynamic registration.** Gateway boots, polls each agent's card, builds its registry. No hardcoded routes.
- **Versionable.** The `version` field lets clients detect schema changes.

In production, you'd add **mTLS** between gateway and agents, and **JWS-signed** Agent Cards so the gateway can verify authenticity (A2A spec supports this).

## 4. Why MCP for tools?

MCP (Model Context Protocol) is Anthropic's open standard for connecting LLMs to data sources. We don't run a full MCP server in this scaffold (would add complexity), but we use the **same tool shape** Anthropic's API expects: `name`, `description`, `input_schema`. This means:

- Migrating to a real MCP server (separate process, stdio/SSE transport) is a wrapping exercise — the tool logic doesn't change.
- The same tools can be reused outside this app (e.g., directly in Claude Desktop) by exposing them via `mcp` Python package.

**Why two tools per agent (and not one big tool)?** Tool decomposition matters for LLM tool-use accuracy:
- Smaller tools → clearer descriptions → better LLM judgment about when to call them.
- Easier to evolve and version independently.
- Easier to test and mock.

## 5. Why LangGraph for the agent reasoning loop?

We chose **LangGraph** over alternatives because:

| Option | Pros | Cons |
|--------|------|------|
| Hand-rolled `while` loop | Simple, no deps | No checkpointing, no built-in tracing, hard to extend |
| **LangGraph** | Stateful, resumable, native LangSmith tracing, declarative graph | LangChain dependency surface |
| CrewAI | Role-based agents, opinionated multi-agent patterns | Less flexible state model, harder to debug |
| AutoGen | Conversational multi-agent | Heavier, focuses on agent-to-agent chat, not tools |

For your project: **start with LangGraph** for individual agent loops, **consider CrewAI** if you find yourself wanting role-based collaboration patterns (e.g., "Researcher", "Critic", "Writer"). They're not mutually exclusive — you can run a CrewAI workflow inside a LangGraph node.

The base agent has a **fallback plain-loop** so the system runs even if LangGraph isn't installed. This is a deliberate "graceful degradation" pattern — your prod code should always work without optional deps where possible.

## 6. Why Langfuse AND LangSmith?

They serve different needs:

- **LangSmith** — best for *development*. Activates via env vars; LangChain/LangGraph automatically emit traces. You see the exact tree of LLM calls, tool calls, prompts, and responses. No code changes needed.
- **Langfuse** — best for *production*. Self-hostable (compliance!), has prompt-management UI (your team can edit prompts without redeploying), supports custom evals, and shows cost dashboards.

We expose a unified `trace_span` facade so application code doesn't care which is enabled. Both gracefully no-op when their env vars are missing.

**For your project:** start with LangSmith for fast feedback. Add Langfuse before going to prod, especially if you handle PHI/PII (Langfuse self-hosted keeps data in your VPC).

## 7. Why Pydantic for everything?

- **Validation at boundaries.** Every JSON-RPC request is parsed via `JsonRpcRequest`. Bad payloads fail with structured errors before reaching business logic.
- **Type safety.** Combined with mypy, you catch entire classes of bugs at static-check time.
- **Settings.** `pydantic-settings` makes env var handling type-safe.

## 8. Guardrails: why our own + when to use guardrails-ai?

Our `common/guardrails.py` is intentionally minimal — regex + heuristics. Reasons:

- **Zero deps.** Easy to read, modify, deploy.
- **Fast.** No model inference per request. Critical for input-side guardrails (every request hits them).

**Use the full `guardrails-ai` package** when you need:

- Schema-validated LLM outputs (e.g., "this output MUST match this Pydantic model")
- Built-in detectors (toxicity, bias, hallucination via FactualityValidator, etc.)
- Hub of community-built validators

**Use Microsoft Presidio** when you need production-grade PII detection (multi-locale, ML-based NER for names/addresses). Our regex-based SSN detector is a starting point only.

See `docs/guardrails.md` for plug-in instructions.

## 9. Why structured JSON logging with correlation IDs?

In a multi-agent system, ONE user request creates a tree of operations:

```
trace_id=abc123
  gateway.request
    gateway.guardrails.input
    router.route → claims-agent
    gateway.forward → http POST :8001
      [INSIDE CLAIMS AGENT]
      agent.call_llm (iter=0)
      tool.claim_lookup
      agent.call_llm (iter=1)
      tool.fraud_check
      agent.call_llm (iter=2) → final response
    gateway.guardrails.output
```

Without `trace_id` threaded through every log, you cannot reconstruct what happened when something fails. With it, `grep trace_id=abc123 logs.json` gives you the full story — and any log aggregator (Datadog, ELK) reconstructs the tree visually.

## 10. Why an in-memory task store (and how to fix for prod)?

`BaseA2AAgent._task_store` is a plain dict. Fine for learning, **NOT** for production:

- **Lost on restart.** Tasks in `WORKING` state vanish.
- **Not shared.** Multiple agent replicas can't see each other's tasks.

For prod, replace with:
- **Redis** for short-lived state (TTL on completed tasks)
- **PostgreSQL** for durable history (especially for INPUT_REQUIRED tasks waiting on user)
- **DynamoDB / Cosmos DB** if cloud-native

Wrap behind a `TaskStore` interface so swapping backends is a one-file change.

## 11. Pricing and cost control

Three model tiers (`MODEL_FAST`, `MODEL_BALANCED`, `MODEL_DEEP`) lets you assign workloads by cost-vs-quality:

- Routing classification → Haiku (cheap, fast)
- Standard agent reasoning → Sonnet (balanced)
- Complex multi-step reasoning → Opus (when accuracy beats cost)

The report generator estimates per-run cost. Pipe these into Langfuse for trend dashboards.

## 12. What's intentionally NOT included (and why)

| Missing | Why & how to add |
|---------|------------------|
| Streaming (`message/stream`) | Adds Server-Sent Events complexity. Add when you need partial responses for UX. Use `EventSourceResponse` from `sse-starlette`. |
| Push notifications | A2A spec supports webhooks back to clients. Add when long tasks need progress callbacks. |
| Authentication beyond static bearer | Use `python-jose` for JWT, validate per-request. Tie to your IdP (Okta, Azure AD). |
| Rate limiting | Use `slowapi` or a Redis-backed limiter on the gateway. |
| Persistent task store | See section 10. |
| Real DB for claims/policies | Replace mock dicts with SQLAlchemy + a real DB. |
| HITL (human-in-the-loop) | Use the `INPUT_REQUIRED` task state — pause graph, await user input, resume from checkpoint. LangGraph supports this natively. |

These are all 1-3 day additions on top of the existing scaffold. The architecture is designed to absorb them without restructuring.
