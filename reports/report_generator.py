"""
Run report generator.

Generates a self-contained HTML report after a multi-agent workflow:
  - Request summary
  - Each agent's contribution
  - Tool calls & their results
  - Token usage and cost
  - Guardrail outcomes
  - Latency breakdown

Why HTML (not just JSON): stakeholders read reports, devs read JSON.
HTML works for both — it includes the raw JSON in a `<details>` block.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A2A Run Report — {run_id}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #e8eef3; padding-bottom: .3rem; }}
  .pill {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px;
          font-size: .8rem; font-weight: 600; }}
  .ok {{ background: #d4f4dd; color: #1d6e3a; }}
  .warn {{ background: #fff3cd; color: #856404; }}
  .err {{ background: #f8d7da; color: #842029; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ text-align: left; padding: .5rem .8rem; border-bottom: 1px solid #e8eef3;
           vertical-align: top; }}
  th {{ background: #f7f9fb; font-weight: 600; }}
  pre {{ background: #f7f9fb; padding: .8rem; border-radius: 6px;
        overflow-x: auto; font-size: .85rem; }}
  .meta {{ color: #6c757d; font-size: .9rem; }}
  details {{ margin: .5rem 0; }}
  summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
<h1>A2A Run Report</h1>
<p class="meta">Run ID: <code>{run_id}</code> · Generated {generated_at}</p>

<h2>Request</h2>
<table>
  <tr><th>Method</th><td>{method}</td></tr>
  <tr><th>Routed to</th><td>{routed_agent}</td></tr>
  <tr><th>Status</th><td><span class="pill {status_class}">{status}</span></td></tr>
  <tr><th>Total latency</th><td>{latency_ms} ms</td></tr>
</table>

<h2>User Input</h2>
<pre>{user_input}</pre>

<h2>Agent Output</h2>
<pre>{agent_output}</pre>

<h2>Tool Calls ({tool_count})</h2>
{tool_calls_html}

<h2>Token Usage</h2>
<table>
  <tr><th>Input tokens</th><td>{input_tokens}</td></tr>
  <tr><th>Output tokens</th><td>{output_tokens}</td></tr>
  <tr><th>Estimated cost (USD)</th><td>${cost_usd:.4f}</td></tr>
</table>

<h2>Guardrails</h2>
{guardrails_html}

<details>
<summary>Raw JSON (full task)</summary>
<pre>{raw_json}</pre>
</details>
</body>
</html>
"""


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# Approximate pricing — update from Anthropic's pricing page.
# Per million tokens; values are placeholders, refresh before reporting.
MODEL_PRICING = {
    "claude-haiku-4-5":  {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-opus-4-5":   {"input": 15.0, "output": 75.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-5"])
    return (input_tokens / 1_000_000) * p["input"] + \
           (output_tokens / 1_000_000) * p["output"]


def render_report(
    *,
    run_id: str,
    method: str,
    routed_agent: str,
    status: str,
    latency_ms: int,
    user_input: str,
    agent_output: str,
    tool_calls: list[dict[str, Any]],
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "claude-sonnet-4-5",
    guardrail_violations: list[dict[str, Any]] | None = None,
    raw_task: dict[str, Any] | None = None,
) -> str:
    """Returns rendered HTML."""
    status_class = {"completed": "ok", "failed": "err"}.get(status, "warn")

    # Tool calls block
    if tool_calls:
        rows = []
        for i, tc in enumerate(tool_calls, 1):
            rows.append(
                f"<details><summary>{i}. {_esc(tc.get('tool', '?'))}</summary>"
                f"<pre>{_esc(json.dumps(tc.get('result', {}), indent=2))}</pre>"
                f"</details>"
            )
        tool_calls_html = "\n".join(rows)
    else:
        tool_calls_html = "<p class='meta'>No tools called.</p>"

    # Guardrails block
    if guardrail_violations:
        rows = "".join(
            f"<tr><td><span class='pill {('err' if v.get('severity') == 'block' else 'warn')}'>"
            f"{_esc(v.get('severity', 'info'))}</span></td>"
            f"<td>{_esc(v.get('rule'))}</td><td>{_esc(v.get('message'))}</td></tr>"
            for v in guardrail_violations
        )
        guardrails_html = (
            f"<table><tr><th>Severity</th><th>Rule</th><th>Message</th></tr>{rows}</table>"
        )
    else:
        guardrails_html = "<p class='meta'>No violations.</p>"

    cost = estimate_cost(model, input_tokens, output_tokens)

    return REPORT_TEMPLATE.format(
        run_id=_esc(run_id),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        method=_esc(method),
        routed_agent=_esc(routed_agent),
        status=_esc(status),
        status_class=status_class,
        latency_ms=latency_ms,
        user_input=_esc(user_input),
        agent_output=_esc(agent_output),
        tool_count=len(tool_calls),
        tool_calls_html=tool_calls_html,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        guardrails_html=guardrails_html,
        raw_json=_esc(json.dumps(raw_task or {}, indent=2, default=str)),
    )


def save_report(html: str, out_dir: str = "reports", filename: str | None = None) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = filename or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
