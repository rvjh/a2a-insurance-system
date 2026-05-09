# Observability Setup

## Langfuse (recommended for prod)

### Option A: Cloud (fastest)
1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a project → Settings → API keys → copy public + secret key
3. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
4. Restart services. You'll see `Langfuse enabled at ...` in the logs.

### Option B: Self-hosted (for compliance / PHI)
```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
# Browse http://localhost:3000, register, create project, copy keys
```
Set `LANGFUSE_HOST=http://localhost:3000` in `.env`.

### What you get
- **Traces:** every JSON-RPC request as a tree (gateway → agent → tool calls)
- **Costs:** per-request token spend, broken down by model
- **Sessions:** group related requests by `context_id`
- **Datasets & evals:** save problem cases, run regression evals
- **Prompt management:** edit system prompts in UI, version them, rollback

### Best practices
- Use `metadata` on spans to tag environment (`prod`/`staging`/`dev`), agent name, user role.
- Add **scores** programmatically when you have ground truth:
  ```python
  langfuse_client.score(trace_id=tid, name="correct_disposition", value=1.0)
  ```
- Build a small eval dataset of 30–50 representative inputs. Run weekly.

---

## LangSmith (recommended for development)

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. API key → copy
3. Add to `.env`:
   ```
   LANGCHAIN_API_KEY=ls__...
   LANGCHAIN_PROJECT=a2a-insurance-system
   ```
4. Restart. LangChain/LangGraph automatically emit traces.

### What you get
- **Tree view of every LangGraph run** — see each node, its inputs/outputs
- **Playground** — re-run any LLM call with edited inputs/prompts
- **Annotations** — manually score outputs, build datasets
- **Comparisons** — diff two runs side-by-side (priceless for prompt iteration)

### When to use LangSmith vs Langfuse
- LangSmith: writing prompts, debugging a failing run today
- Langfuse: weekly cost report, prompt-version A/B tests, prod monitoring

You can run both simultaneously — they don't conflict.

---

## OpenTelemetry (advanced, vendor-neutral)

If your org standardizes on OTel + Datadog/Honeycomb/Jaeger, swap our `trace_span` for OTel spans:

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("gateway.request") as span:
    span.set_attribute("agent", target_card.name)
    ...
```

Anthropic SDK has OTel integration via `opentelemetry-instrumentation-anthropic`. LangChain has `langchain-otel`. Both Langfuse and LangSmith can ingest OTel data.

---

## Reading traces — what to look for

**Healthy run:**
- 1 LLM call → tool calls → 1 LLM call (final answer)
- Total latency 2–8s
- Tool errors: 0

**Smell test red flags:**
- LLM loops > 3 iterations → prompt is unclear or tool descriptions overlap
- Tool errors in production → mock/stub still wired up?
- `output_tokens > 1500` → likely the model is rambling; tighten the system prompt
- Same `claim_lookup` called 3 times in one task → LLM forgot prior result; add memory or tighten prompt

---

## Cost tracking

Run reports (in `reports/`) include estimated cost. Verify the per-million-token rates in `reports/report_generator.py` against the [Anthropic pricing page](https://www.anthropic.com/pricing) before relying on them.

For ongoing cost monitoring in production:
- Langfuse dashboards (built-in)
- Or: emit a custom metric per request → CloudWatch / Prometheus → Grafana
