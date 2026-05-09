# Code Walkthrough — Why Each File Exists & What It Optimizes For

This is the answer to "explain each codebase, why I chose it that way, and the best outcome."

## `common/schemas.py`
**What:** Pydantic models for JSON-RPC 2.0 + the A2A protocol (Task, Message, Part, Artifact, AgentCard).

**Why this way:**
- One file, one source of truth. Every other module imports schemas from here, so wire-format changes are atomic.
- Inheritance-free design — Pydantic models are values, not behavior. Keeps things simple to test.
- `Part = Union[TextPart, DataPart, FilePart]` with discriminated `kind` field — Pydantic auto-routes deserialization, you never write `if isinstance(...)` manually.
- `id` auto-generates a UUID — clients don't have to remember it, but can override.

**Best outcome:** wire compliance with A2A spec. If a third-party A2A client/agent talks to this system, the JSON shapes match.

## `common/logging_utils.py`
**What:** JSON formatter + ContextVars for trace_id and agent_name.

**Why this way:**
- `ContextVar` (not threading.local!) survives `await` boundaries — required for FastAPI's async handlers.
- JSON output → every observability platform indexes it natively.
- Three lines of setup at startup, then every other module just `get_logger(__name__)`.

**Best outcome:** correlate every log line for one user request — across gateway → agents → tools — with `grep trace_id=xxx`.

## `common/observability.py`
**What:** Unified facade over Langfuse + LangSmith.

**Why this way:**
- Both backends activate via env vars only — no code changes between dev/prod.
- `trace_span` context manager is the only API the rest of the code uses; backend swap is one file.
- Lazy imports — system runs without these heavy packages installed.

**Best outcome:** observability becomes a free side-effect of normal code. No "instrumentation pass" needed.

## `common/llm_client.py`
**What:** Wrapper around `anthropic.Anthropic` with model-tier presets.

**Why this way:**
- Single chokepoint for retries, observability, cost tracking.
- Constants `MODEL_FAST/BALANCED/DEEP` document intent — code says "use Haiku because this is cheap classification" via `MODEL_FAST`.
- Lazy SDK import — keeps test runs fast, allows partial functionality without `anthropic` installed.

**Best outcome:** swapping models is a one-line change. Adding cost tracking, fallback to a different provider, request caching — all happen in one place.

## `common/guardrails.py`
**What:** Input + output safety checks (PII, prompt injection, hallucination heuristic).

**Why this way:**
- Plain Python regex — no model inference per request, no extra deps.
- `GuardrailViolation` is a structured object, not a string — downstream code can render UI, log structured data, or feed into evaluation pipelines.
- `passed` and `has_blocking()` make the integration point one line at the call site.

**Best outcome:** safety is enforced uniformly without the team having to memorize what to check. Adding a new check = one new function in this file.

## `config/settings.py`
**What:** Single Pydantic Settings class for all env-driven config.

**Why this way:**
- Type-validated env vars — `int` fields fail loudly at startup if misconfigured, not silently at runtime.
- `.env` file support for local dev; same code reads real env vars in prod.
- Importable as `from config.settings import settings` — no global mutable state to thread through code.

**Best outcome:** zero "what env var is that?" surprises. New devs run `cp .env.example .env` and they're productive.

## `mcp_tools/base.py`
**What:** Abstract `MCPTool` + `ToolRegistry`.

**Why this way:**
- `to_anthropic_tool()` produces exactly the dict shape Claude expects — agents don't transform tools manually.
- `ToolRegistry` per agent — enforces the principle "tool == agent capability." No global tool soup.
- Same shape as the real MCP package — migrating to a stand-alone MCP server is a wrapping job.

**Best outcome:** new tool = subclass + register. ~20 lines per tool.

## `mcp_tools/claims_tools.py` and `policy_tools.py`
**What:** Concrete tools for each agent.

**Why this way:**
- Mock data lives next to tool code so the demo works without external infra.
- Each tool has a focused, single responsibility (lookup OR check, not both).
- `description` text is written **for the LLM**, not humans — describes when to use the tool, what to expect.

**Best outcome:** the LLM reliably picks the right tool because descriptions are unambiguous. Each tool can be unit-tested without an LLM in the loop.

## `agents/base_agent.py`
**What:** Reusable A2A agent server: JSON-RPC dispatcher + LangGraph reasoning loop + task store + agent-card endpoint.

**Why this way:**
- All A2A protocol logic lives here once. Concrete agents only specify card, tools, system prompt.
- LangGraph for the loop = automatic LangSmith tracing, easy to add nodes (critic, retry, HITL).
- Fallback plain loop if LangGraph isn't installed = graceful degradation.
- `_task_store` is in-memory but isolated behind methods — swap to Redis later by changing 3 methods.

**Best outcome:** new agent = ~30 lines (just card + prompt + tool list). Zero protocol code to write.

## `agents/claims_agent/main.py` and `policy_agent/main.py`
**What:** Concrete agents — each defines its identity, prompts, tools.

**Why this way:**
- Tiny files (~40 lines each). Easy to compare two agents side-by-side.
- System prompts are explicit, opinionated, and enforce "must call tool X first" — reduces hallucination risk.
- Hard boundaries between agent responsibilities ("Claims agent does NOT make coverage decisions") — prevents responsibility creep.

**Best outcome:** clear agent boundaries + reproducible behavior + easy A/B testing of prompts.

## `gateway/router.py`
**What:** Three-tier routing strategy: explicit, keyword, LLM fallback.

**Why this way:**
- Cheap-first ordering. Explicit (free) → keyword (free) → LLM (~10ms + cost). Most requests hit the cheap path.
- `score`-based keyword matching beats binary if/else when multiple agents could match.
- LLM router uses Haiku — minimal cost for ambiguous cases.
- All paths log which strategy was used, so over time you can tune the cheap layers based on real traffic.

**Best outcome:** smart routing without paying for an LLM call on every request.

## `gateway/main.py`
**What:** FastAPI app: auth, JSON-RPC parsing, guardrails, route, forward.

**Why this way:**
- Thin gateway — no business logic. Easy to reason about.
- `lifespan` registers agents on startup with retries (handles agents not yet ready).
- JSON-RPC errors return HTTP 200 — protocol-correct.
- Guardrails redact-then-forward — agents never see raw PII.

**Best outcome:** a single hardened entrypoint that is easy to scale horizontally (stateless) and easy to swap in front of any A2A-compliant agent.

## `reports/report_generator.py`
**What:** Pure-function HTML renderer from a task result.

**Why this way:**
- Pure function (input dict → output string) — trivial to test, no I/O.
- Self-contained HTML (inline CSS) — no static asset hosting needed; reports work via `file://` or email attachment.
- Includes raw JSON in `<details>` block — devs can debug, stakeholders can skim.

**Best outcome:** every interesting run becomes a shareable artifact. Audit trail for free. Useful for stakeholder demos and incident analysis.

## `tests/client_demo.py`
**What:** End-to-end smoke test that hits the gateway with three scenarios and saves reports.

**Why this way:**
- Doubles as a usage example for clients integrating with this system.
- Generates real artifacts in `reports/` — you can show stakeholders what one run looks like.

**Best outcome:** "does my system actually work end-to-end?" answered in one command.

## `tests/test_units.py`
**What:** Unit tests for schemas, guardrails, tools, registry — no LLM calls.

**Why this way:**
- Fast (sub-second). Run on every save in your editor.
- Test the contracts, not the LLM. The LLM is tested via the demo client + Langfuse evals.

**Best outcome:** tight feedback loop. CI fails fast on regressions in the deterministic layer.

## `Makefile`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml`
**What:** Operational glue.

**Why this way:**
- `make` commands match what you actually run. No memorizing flags.
- Dockerfile is minimal (`pip install` + `COPY .`) — trades image size for build speed during iteration.
- `pyproject.toml` consolidates black/ruff/mypy/pytest config — one file, no XX dotfiles.

**Best outcome:** new dev → productive in ~5 minutes (`make install && make run`).

---

## The big-picture "best outcome"

The system is designed so that **the simple things stay simple** (tests run without an LLM key; system runs without Langfuse) and **the hard things become possible** (swap LLM provider, scale horizontally, add agents, add guardrails) — without rewriting the core.

Most "simple agent demos" you see online optimize for "look how easy this is" and collapse the moment you try to make them production-grade. This scaffold optimizes for "what does the eventual production system look like?" and removes scaffolding-tax along the way.

You will use these patterns directly in your real project.
