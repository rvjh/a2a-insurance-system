"""
A2A Gateway — single entrypoint for clients.

RESPONSIBILITIES:
  - Authenticate clients (Bearer token; JWT in production)
  - Apply input guardrails (PII, prompt-injection)
  - Discover agents (reads /.well-known/agent-card.json from each)
  - Route requests to the right agent
  - Forward JSON-RPC requests + return responses unchanged
  - Emit observability traces

Run: uvicorn gateway.main:app --port 8000
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse

from common.schemas import (
    JsonRpcRequest, JsonRpcResponse, JsonRpcError, RpcErrorCode,
    MessageSendParams, TextPart,
)
from common.llm_client import LLMClient
from common.guardrails import run_input_guardrails
from common.logging_utils import configure_logging, get_logger, trace_id_ctx, agent_name_ctx
from common.observability import init_observability, trace_span
from config.settings import settings
from gateway.router import AgentRegistry, AgentRouter

logger = get_logger(__name__)

# --- Lifespan: register agents on startup ---------------------------------

registry = AgentRegistry()
http_client: Optional[httpx.AsyncClient] = None
router: Optional[AgentRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, router
    configure_logging(settings.log_level)
    init_observability()
    http_client = httpx.AsyncClient(timeout=settings.request_timeout_s)

    # Discover registered agents (retry a few times — they may still be starting)
    targets = [settings.claims_agent_url, settings.policy_agent_url]
    for url in targets:
        for attempt in range(5):
            try:
                await registry.register_from_url(url, http_client)
                break
            except Exception as e:
                logger.warning(f"Agent discovery for {url} failed (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)

    router = AgentRouter(registry, llm=LLMClient())
    logger.info(f"Gateway started, agents registered: {[c.name for c in registry.all()]}")
    yield
    if http_client:
        await http_client.aclose()


app = FastAPI(title="A2A Insurance Gateway", version="1.0.0", lifespan=lifespan)


# --- Auth dependency ------------------------------------------------------

def _check_auth(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != settings.api_bearer_token:
        raise HTTPException(403, "Invalid token")


# --- Endpoints ------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agents": [c.name for c in registry.all()],
    }


@app.get("/agents")
async def list_agents(authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    _check_auth(authorization)
    return {"agents": [c.model_dump() for c in registry.all()]}


@app.post("/")
async def jsonrpc_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    """
    Single JSON-RPC endpoint. Clients POST JSON-RPC envelopes; we route to
    the right agent based on either explicit `agent` field or content.

    Custom param `agent` (optional) tells the router which agent to use:
      { "jsonrpc": "2.0", "method": "message/send",
        "params": { "agent": "claims-agent", "message": {...} } }
    """
    _check_auth(authorization)
    try:
        body = await request.json()
    except Exception:
        return _err(None, RpcErrorCode.PARSE_ERROR, "Invalid JSON")

    try:
        rpc_req = JsonRpcRequest(**body)
    except Exception as e:
        return _err(body.get("id"), RpcErrorCode.INVALID_REQUEST, str(e))

    trace_id_ctx.set(str(rpc_req.id))
    agent_name_ctx.set("gateway")

    with trace_span("gateway.request", metadata={"method": rpc_req.method}) as span:
        # Pass-through methods (the agent owns the task store, so tasks/get
        # must go to the right agent — we require an explicit `agent` param).
        if rpc_req.method in ("tasks/get", "tasks/cancel"):
            agent_name = (rpc_req.params or {}).get("agent")
            if not agent_name:
                return _err(rpc_req.id, RpcErrorCode.INVALID_PARAMS,
                            "Specify 'agent' for tasks/* methods")
            card = registry.get(agent_name)
            if not card:
                return _err(rpc_req.id, RpcErrorCode.AGENT_UNAVAILABLE,
                            f"Unknown agent {agent_name}")
            return await _forward(rpc_req, card.url)

        if rpc_req.method != "message/send":
            return _err(rpc_req.id, RpcErrorCode.METHOD_NOT_FOUND,
                        f"Method {rpc_req.method} not supported by gateway")

        # message/send — apply guardrails + route
        params_dict = rpc_req.params or {}
        explicit_agent = params_dict.pop("agent", None)
        try:
            params = MessageSendParams(**params_dict)
        except Exception as e:
            return _err(rpc_req.id, RpcErrorCode.INVALID_PARAMS, f"Bad params: {e}")

        # Input guardrails on user text
        user_text = "\n".join(
            p.text for p in params.message.parts if isinstance(p, TextPart)
        )
        grail = run_input_guardrails(
            user_text,
            allowed_topics=None,   # let agents decide topical scope
            redact_pii=True,
        )
        if grail.has_blocking():
            details = [v.message for v in grail.violations
                       if v.severity.value == "block"]
            span["output"] = {"blocked": True, "violations": details}
            return _err(
                rpc_req.id, RpcErrorCode.GUARDRAIL_VIOLATION,
                "Input failed safety checks", data={"violations": details},
            )

        # Replace user text with sanitized version (PII redacted)
        if grail.sanitized_text and grail.sanitized_text != user_text:
            new_parts = []
            replaced = False
            for p in params.message.parts:
                if isinstance(p, TextPart) and not replaced:
                    new_parts.append(TextPart(text=grail.sanitized_text))
                    replaced = True
                else:
                    new_parts.append(p)
            params.message.parts = new_parts

        # Route + forward
        try:
            target_card = await router.route(
                explicit_agent=explicit_agent, user_text=user_text,
            )
        except Exception as e:
            logger.exception("Routing failed")
            return _err(rpc_req.id, RpcErrorCode.INTERNAL_ERROR, f"Routing error: {e}")

        # Reserialize params (without the gateway-only `agent` field)
        rpc_req.params = {"message": params.message.model_dump(mode="json")}
        if params.configuration:
            rpc_req.params["configuration"] = params.configuration
        span["output"] = {"routed_to": target_card.name}
        return await _forward(rpc_req, target_card.url)


async def _forward(rpc_req: JsonRpcRequest, agent_url: str) -> JSONResponse:
    assert http_client is not None
    try:
        with trace_span("gateway.forward", metadata={"target": agent_url}):
            resp = await http_client.post(
                agent_url, json=rpc_req.model_dump(exclude_none=True),
            )
            resp.raise_for_status()
            return JSONResponse(resp.json())
    except httpx.HTTPError as e:
        logger.exception("Agent call failed")
        return _err(rpc_req.id, RpcErrorCode.AGENT_UNAVAILABLE, f"Agent error: {e}")


def _err(rid: Any, code: int, msg: str, data: Any = None) -> JSONResponse:
    return JSONResponse(
        JsonRpcResponse(
            id=rid, error=JsonRpcError(code=code, message=msg, data=data),
        ).model_dump(exclude_none=True),
        status_code=200,  # JSON-RPC errors are 200; HTTP errors only for transport failures
    )
