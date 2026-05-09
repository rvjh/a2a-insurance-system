"""
Test client — exercises the gateway end-to-end and emits a report.

Usage:
    python -m tests.client_demo

Make sure all three services are running first:
    uvicorn agents.claims_agent.main:app --port 8001 &
    uvicorn agents.policy_agent.main:app --port 8002 &
    uvicorn gateway.main:app --port 8000 &
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

import httpx

from common.schemas import JsonRpcRequest, Message, TextPart
from config.settings import settings
from reports.report_generator import render_report, save_report


async def call_gateway(client: httpx.AsyncClient, user_text: str,
                        agent: str | None = None) -> tuple[dict, int]:
    msg = Message(role="user", parts=[TextPart(text=user_text)])
    params: dict = {"message": msg.model_dump(mode="json")}
    if agent:
        params["agent"] = agent
    req = JsonRpcRequest(method="message/send", params=params, id=str(uuid.uuid4()))

    t0 = time.time()
    resp = await client.post(
        f"http://localhost:{settings.gateway_port}/",
        json=req.model_dump(exclude_none=True),
        headers={"Authorization": f"Bearer {settings.api_bearer_token}"},
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    return resp.json(), elapsed_ms


async def main() -> None:
    scenarios = [
        ("Validate claim CLM-1002 and check for fraud", None),
        ("What's the coverage for procedure 99213 under policy POL-555?", None),
        ("Look up claim CLM-1003", "claims-agent"),
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, (text, agent) in enumerate(scenarios, 1):
            print(f"\n{'='*60}\nScenario {i}: {text}\n{'='*60}")
            try:
                rpc_resp, latency_ms = await call_gateway(client, text, agent)
            except Exception as e:
                print(f"Request failed: {e}")
                continue

            if "error" in rpc_resp and rpc_resp["error"]:
                print(f"RPC error: {rpc_resp['error']}")
                continue

            task = rpc_resp.get("result", {})
            status = task.get("status", {}).get("state", "unknown")
            history = task.get("history", [])
            agent_msgs = [m for m in history if m.get("role") == "agent"]
            agent_text = ""
            tool_results: list = []
            if agent_msgs:
                last_agent_msg = agent_msgs[-1]
                for part in last_agent_msg.get("parts", []):
                    if part.get("kind") == "text":
                        agent_text += part.get("text", "")
                    elif part.get("kind") == "data":
                        tool_results = part.get("data", {}).get("tool_results", [])

            print(f"Status: {status}")
            print(f"Latency: {latency_ms} ms")
            print(f"Agent says:\n{agent_text[:600]}{'...' if len(agent_text) > 600 else ''}")
            print(f"Tools called: {[t.get('tool') for t in tool_results]}")

            html = render_report(
                run_id=str(rpc_resp.get("id")),
                method="message/send",
                routed_agent=task.get("metadata", {}).get("routed_agent",
                              agent_msgs[-1].get("parts", [{}])[0].get("name", "?")
                              if agent_msgs else "?"),
                status=status,
                latency_ms=latency_ms,
                user_input=text,
                agent_output=agent_text,
                tool_calls=tool_results,
                raw_task=task,
            )
            path = save_report(html, filename=f"scenario-{i}.html")
            print(f"Report saved: {path}")


if __name__ == "__main__":
    asyncio.run(main())
