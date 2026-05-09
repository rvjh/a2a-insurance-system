"""
MCP (Model Context Protocol) tool abstraction.

ABOUT MCP: Anthropic's open standard for connecting LLMs to data sources and
tools. Each MCP tool exposes:
  - a name + description (LLM uses these to decide when to call it)
  - an input JSON schema
  - an execute() method

In production you'd run MCP tools as separate servers communicating over
stdio/SSE. For this learning project we keep them in-process for simplicity
but use the SAME schema shape — so migrating to a real MCP server is a
1-day refactor (just wrap each tool with mcp.server.Server).

If you want a real MCP server later: pip install mcp, then each tool becomes:
    @server.list_tools()
    @server.call_tool()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class MCPTool(ABC):
    """Base class — every MCP tool subclasses this."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema dict, passed to Claude as tool spec

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run the tool with given arguments. Returns JSON-serializable dict."""
        ...

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool-use schema for Claude."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Per-agent registry. Each agent owns a registry with its tools."""

    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> MCPTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())
