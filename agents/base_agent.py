"""
Base Agent — implements A2A protocol over JSON-RPC.

Each concrete agent (ClaimsAgent, PolicyAgent) extends this and provides:
  - an AgentCard (its identity + skills)
  - a ToolRegistry (its MCP tools)
  - a `process_task` method (the core LLM+tools loop)

WHY LANGGRAPH FOR THE AGENT LOOP:
  - Stateful, cyclical control flow (LLM <-> tool calls) modeled as a graph
  - Built-in checkpointing — pause/resume tasks, useful for INPUT_REQUIRED state
  - Native LangSmith tracing
  - Easier to add new control nodes (e.g., critic, reflection, human-in-loop)
  - More declarative than hand-rolling a while-loop, scales as agents grow

We use LangGraph here. CrewAI is a great alternative if you prefer
role-based multi-agent orchestration with less code; we'll discuss it
in the docs section.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from common.schemas import (
    AgentCard, JsonRpcRequest, JsonRpcResponse, JsonRpcError, RpcErrorCode,
    Task, TaskState, TaskStatus, Message, TextPart, DataPart, MessageSendParams,
    Artifact,
)
from common.llm_client import LLMClient
from common.observability import trace_span
from common.logging_utils import get_logger, agent_name_ctx, trace_id_ctx
from common.guardrails import run_output_guardrails
from mcp_tools.base import ToolRegistry

logger = get_logger(__name__)


class AgentState(TypedDict):
    """LangGraph state schema — single source of truth across nodes."""
    messages: list[dict[str, Any]]   # Anthropic-format messages
    tool_results: list[dict[str, Any]]
    final_text: str
    iteration: int
    max_iterations: int


class BaseA2AAgent(ABC):
    """A2A agent server. Exposes JSON-RPC methods and an Agent Card."""

    def __init__(
        self,
        agent_card: AgentCard,
        tool_registry: ToolRegistry,
        llm: LLMClient,
        system_prompt: str,
        max_tool_iterations: int = 6,
    ) -> None:
        self.card = agent_card
        self.tools = tool_registry
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations
        self._task_store: dict[str, Task] = {}  # in-memory; use Redis in prod
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # LangGraph: agent reasoning loop
    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        """
        Build a LangGraph state machine:
            START -> call_llm -> [tools? -> run_tools -> call_llm] -> END

        WHY: makes the loop explicit & inspectable in LangSmith. Each node
        becomes a span. You can later inject a `critic` node between
        `call_llm` and `END` without rewriting the loop.
        """
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            logger.warning("langgraph not installed — falling back to plain loop")
            return None

        graph = StateGraph(AgentState)
        graph.add_node("call_llm", self._node_call_llm)
        graph.add_node("run_tools", self._node_run_tools)
        graph.set_entry_point("call_llm")

        def _route(state: AgentState) -> str:
            last = state["messages"][-1] if state["messages"] else None
            if state["iteration"] >= state["max_iterations"]:
                return "end"
            if last and last.get("role") == "assistant":
                content = last.get("content", [])
                if isinstance(content, list) and any(
                    block.get("type") == "tool_use" for block in content
                ):
                    return "tools"
            return "end"

        graph.add_conditional_edges(
            "call_llm", _route, {"tools": "run_tools", "end": END},
        )
        graph.add_edge("run_tools", "call_llm")
        return graph.compile()

    def _node_call_llm(self, state: AgentState) -> AgentState:
        with trace_span("agent.call_llm", metadata={"iter": state["iteration"]}) as span:
            resp = self.llm.complete(
                system=self.system_prompt,
                messages=state["messages"],
                tools=self.tools.list_anthropic_tools(),
                max_tokens=2048,
            )
            # Reconstruct an assistant message with both text and tool_use blocks
            assistant_content: list[dict[str, Any]] = []
            if resp["text"]:
                assistant_content.append({"type": "text", "text": resp["text"]})
            for tu in resp["tool_uses"]:
                assistant_content.append({
                    "type": "tool_use", "id": tu["id"],
                    "name": tu["name"], "input": tu["input"],
                })
            state["messages"].append({"role": "assistant", "content": assistant_content})
            state["iteration"] += 1
            if not resp["tool_uses"]:
                state["final_text"] = resp["text"]
            span["output"] = {"used_tools": [t["name"] for t in resp["tool_uses"]]}
            return state

    def _node_run_tools(self, state: AgentState) -> AgentState:
        last = state["messages"][-1]
        tool_use_blocks = [
            b for b in last["content"] if b.get("type") == "tool_use"
        ]
        tool_result_blocks: list[dict[str, Any]] = []
        for tu in tool_use_blocks:
            tname = tu["name"]
            with trace_span(f"tool.{tname}", input=tu["input"]) as span:
                try:
                    tool = self.tools.get(tname)
                    result = tool.execute(**tu["input"])
                    span["output"] = result
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": json.dumps(result),
                    })
                    state["tool_results"].append({"tool": tname, "result": result})
                except Exception as e:
                    logger.exception(f"Tool {tname} failed")
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True,
                    })
        state["messages"].append({"role": "user", "content": tool_result_blocks})
        return state

    # ------------------------------------------------------------------
    # A2A methods (called via JSON-RPC)
    # ------------------------------------------------------------------
    async def handle_message_send(self, params: MessageSendParams) -> Task:
        msg = params.message
        # Extract user text from message parts
        user_text_parts = [p.text for p in msg.parts if isinstance(p, TextPart)]
        user_data_parts = [p.data for p in msg.parts if isinstance(p, DataPart)]
        combined_text = "\n".join(user_text_parts)
        if user_data_parts:
            combined_text += "\n\nStructured input:\n" + json.dumps(user_data_parts, indent=2)

        task = Task(
            id=str(uuid4()),
            context_id=msg.context_id or str(uuid4()),
            status=TaskStatus(state=TaskState.WORKING),
            history=[msg],
        )
        self._task_store[task.id] = task

        # Run the LangGraph loop
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": combined_text}],
            "tool_results": [],
            "final_text": "",
            "iteration": 0,
            "max_iterations": self.max_tool_iterations,
        }

        try:
            if self._graph is not None:
                final_state = self._graph.invoke(initial_state)
            else:
                final_state = await self._fallback_loop(initial_state)
            final_text = final_state["final_text"] or "(no response)"

            # Output guardrails
            grail = run_output_guardrails(final_text)
            if grail.has_blocking():
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text="Output blocked by safety guardrails.")],
                    ),
                )
                return task

            # Build artifact + agent message
            agent_msg = Message(
                role="agent",
                parts=[
                    TextPart(text=grail.sanitized_text or final_text),
                    DataPart(data={"tool_results": final_state["tool_results"]}),
                ],
                context_id=task.context_id,
                task_id=task.id,
            )
            task.history.append(agent_msg)
            task.artifacts.append(Artifact(
                name=f"{self.card.name}-response",
                parts=agent_msg.parts,
            ))
            task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg)
            return task

        except Exception as e:
            logger.exception("Agent task failed")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(role="agent", parts=[TextPart(text=f"Error: {e}")]),
            )
            return task

    async def _fallback_loop(self, state: AgentState) -> AgentState:
        """Plain loop when LangGraph isn't installed."""
        while state["iteration"] < state["max_iterations"]:
            state = self._node_call_llm(state)
            last = state["messages"][-1]
            if not any(
                b.get("type") == "tool_use"
                for b in (last.get("content") or [])
                if isinstance(b, dict)
            ):
                break
            state = self._node_run_tools(state)
        return state

    async def handle_task_get(self, task_id: str) -> Task:
        task = self._task_store.get(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        return task

    async def handle_task_cancel(self, task_id: str) -> Task:
        task = self._task_store.get(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        task.status = TaskStatus(state=TaskState.CANCELED)
        return task

    # ------------------------------------------------------------------
    # FastAPI app factory
    # ------------------------------------------------------------------
    def build_app(self) -> FastAPI:
        app = FastAPI(title=self.card.name, version=self.card.version)

        @app.get("/.well-known/agent-card.json")
        async def agent_card() -> dict[str, Any]:
            """A2A discovery endpoint — gateway reads this to know what we do."""
            return self.card.model_dump()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "agent": self.card.name}

        @app.post("/")
        async def jsonrpc_endpoint(request: Request) -> JSONResponse:
            agent_name_ctx.set(self.card.name)
            try:
                body = await request.json()
            except Exception:
                return self._rpc_err(None, RpcErrorCode.PARSE_ERROR, "Invalid JSON")

            try:
                rpc_req = JsonRpcRequest(**body)
            except Exception as e:
                return self._rpc_err(body.get("id"), RpcErrorCode.INVALID_REQUEST, str(e))

            trace_id_ctx.set(str(rpc_req.id))
            try:
                if rpc_req.method == "message/send":
                    params = MessageSendParams(**(rpc_req.params or {}))
                    task = await self.handle_message_send(params)
                    return self._rpc_ok(rpc_req.id, task.model_dump(mode="json"))
                if rpc_req.method == "tasks/get":
                    tid = (rpc_req.params or {}).get("id")
                    task = await self.handle_task_get(tid)
                    return self._rpc_ok(rpc_req.id, task.model_dump(mode="json"))
                if rpc_req.method == "tasks/cancel":
                    tid = (rpc_req.params or {}).get("id")
                    task = await self.handle_task_cancel(tid)
                    return self._rpc_ok(rpc_req.id, task.model_dump(mode="json"))
                return self._rpc_err(
                    rpc_req.id, RpcErrorCode.METHOD_NOT_FOUND,
                    f"Unknown method: {rpc_req.method}",
                )
            except KeyError as e:
                return self._rpc_err(rpc_req.id, RpcErrorCode.TASK_NOT_FOUND, str(e))
            except Exception as e:
                logger.exception("Unhandled agent error")
                return self._rpc_err(rpc_req.id, RpcErrorCode.INTERNAL_ERROR, str(e))

        return app

    @staticmethod
    def _rpc_ok(rid: Any, result: Any) -> JSONResponse:
        return JSONResponse(
            JsonRpcResponse(id=rid, result=result).model_dump(exclude_none=True)
        )

    @staticmethod
    def _rpc_err(rid: Any, code: int, msg: str, data: Any = None) -> JSONResponse:
        return JSONResponse(
            JsonRpcResponse(
                id=rid,
                error=JsonRpcError(code=code, message=msg, data=data),
            ).model_dump(exclude_none=True)
        )
