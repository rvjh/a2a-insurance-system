"""
JSON-RPC 2.0 and Google A2A Protocol Schemas.

The A2A (Agent-to-Agent) protocol from Google standardizes how AI agents
communicate. It builds on JSON-RPC 2.0 for transport and adds:
  - Agent Cards (capability discovery)
  - Tasks (units of work with lifecycle)
  - Messages (multi-part content: text, data, files)
  - Artifacts (results produced by agents)

Reference: https://a2a-protocol.org/latest/specification/
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 Core
# ---------------------------------------------------------------------------
# JSON-RPC 2.0 mandates: "jsonrpc": "2.0", "method", optional "params",
# optional "id". Responses MUST contain either "result" or "error", never both.

class JsonRpcRequest(BaseModel):
    """A JSON-RPC 2.0 request envelope."""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str = Field(..., description="The A2A method name, e.g. 'message/send'")
    params: Optional[dict[str, Any]] = None
    id: Optional[Union[str, int]] = Field(
        default_factory=lambda: str(uuid4()),
        description="Correlates request with response. None = notification (no reply)."
    )


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object. Codes -32000 to -32099 are reserved for app errors."""
    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcResponse(BaseModel):
    """A JSON-RPC 2.0 response envelope. Either result OR error, never both."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None

    model_config = ConfigDict(extra="forbid")


# Standard JSON-RPC error codes (per spec)
class RpcErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # A2A-specific (in -32000..-32099 reserved server range)
    TASK_NOT_FOUND = -32001
    AGENT_UNAVAILABLE = -32002
    GUARDRAIL_VIOLATION = -32003
    AUTHENTICATION_FAILED = -32004


# ---------------------------------------------------------------------------
# A2A Protocol Types
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    """Lifecycle states for an A2A task."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class DataPart(BaseModel):
    """Structured data — used for tool args, agent-to-agent JSON payloads."""
    kind: Literal["data"] = "data"
    data: dict[str, Any]


class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    name: str
    mime_type: str
    bytes_b64: Optional[str] = None
    uri: Optional[str] = None


# A Message contains 1+ Parts; this is what flows between agents
Part = Union[TextPart, DataPart, FilePart]


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[Part]
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: Optional[str] = None
    context_id: Optional[str] = None  # groups related tasks (e.g. one claim case)


class Artifact(BaseModel):
    """A durable output produced by an agent (report, decision, etc.)."""
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: list[Part]


class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[Message] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Task(BaseModel):
    """A unit of work inside the A2A protocol."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    context_id: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus
    history: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent Card — A2A discovery document (served at /.well-known/agent-card.json)
# ---------------------------------------------------------------------------

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True


class AgentCard(BaseModel):
    """Public description of what an agent can do. The gateway uses this for routing."""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text", "data"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text", "data"])
    skills: list[AgentSkill] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Method-specific param schemas
# ---------------------------------------------------------------------------

class MessageSendParams(BaseModel):
    """Params for the 'message/send' A2A method."""
    message: Message
    configuration: Optional[dict[str, Any]] = None


class TaskGetParams(BaseModel):
    id: str


class TaskCancelParams(BaseModel):
    id: str
