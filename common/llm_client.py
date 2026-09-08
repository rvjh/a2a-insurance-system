"""
Groq LLM client.

This wrapper keeps the existing Anthropic-style interface used by the
agent/tool loop, while translating requests and responses to/from
Groq's OpenAI-compatible Chat Completions API.

Reads GROQ_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from common.logging_utils import get_logger
from common.observability import trace_span

logger = get_logger(__name__)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_FAST = "openai/gpt-oss-20b"
MODEL_BALANCED = "openai/gpt-oss-120b"
MODEL_DEEP = "openai/gpt-oss-120b"


# ============================================================================
# LLM CLIENT
# ============================================================================


class LLMClient:
    """
    Groq wrapper exposing the interface expected by BaseA2AAgent.

    Repository interface:

        complete(
            system=...,
            messages=...,
            model=...,
            max_tokens=...,
            temperature=...,
            tools=...
        )

    Returns:

        {
            "text": "...",
            "tool_uses": [...],
            "stop_reason": "...",
            "usage": {
                "input_tokens": ...,
                "output_tokens": ...
            }
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_BALANCED,
        max_retries: int = 3,
    ) -> None:

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.default_model = default_model
        self.max_retries = max(1, max_retries)
        self._client: Any = None

        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY not set — LLM calls will fail until configured"
            )

    # ========================================================================
    # CLIENT
    # ========================================================================

    def _ensure_client(self) -> Any:
        """
        Lazily create the Groq client.
        """

        if self._client is None:

            try:
                from groq import Groq

                self._client = Groq(
                    api_key=self.api_key,
                    max_retries=self.max_retries,
                )

            except ImportError as exc:

                raise RuntimeError(
                    "Install `groq` package: pip install groq"
                ) from exc

        return self._client

    # ========================================================================
    # TOOL CONVERSION
    # ========================================================================

    @staticmethod
    def _convert_tools(
        tools: Optional[list[dict[str, Any]]],
    ) -> Optional[list[dict[str, Any]]]:
        """
        Convert repository Anthropic-style tools:

            {
                "name": "...",
                "description": "...",
                "input_schema": {...}
            }

        into Groq/OpenAI function tools:

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """

        if not tools:
            return None

        converted: list[dict[str, Any]] = []

        for tool in tools:

            if not isinstance(tool, dict):
                continue

            name = tool.get("name")

            if not name:
                logger.warning(
                    "Skipping tool without a name: %s",
                    tool,
                )
                continue

            input_schema = tool.get("input_schema")

            if not isinstance(input_schema, dict):
                input_schema = {
                    "type": "object",
                    "properties": {},
                }

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get(
                            "description",
                            "",
                        ),
                        "parameters": input_schema,
                    },
                }
            )

        return converted or None

    # ========================================================================
    # MESSAGE CONVERSION
    # ========================================================================

    @staticmethod
    def _convert_messages(
        system: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert repository Anthropic-style messages into
        OpenAI/Groq-compatible chat messages.

        Anthropic-style assistant:

            [
                {"type": "text", ...},
                {"type": "tool_use", ...}
            ]

        becomes:

            {
                "role": "assistant",
                "content": "...",
                "tool_calls": [...]
            }

        Anthropic-style tool result:

            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "...",
                        "content": "..."
                    }
                ]
            }

        becomes:

            {
                "role": "tool",
                "tool_call_id": "...",
                "content": "..."
            }
        """

        converted: list[dict[str, Any]] = []

        # --------------------------------------------------------------------
        # SYSTEM
        # --------------------------------------------------------------------

        converted.append(
            {
                "role": "system",
                "content": system or "",
            }
        )

        # --------------------------------------------------------------------
        # CONVERSATION
        # --------------------------------------------------------------------

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            # ---------------------------------------------------------------
            # SIMPLE STRING CONTENT
            # ---------------------------------------------------------------

            if isinstance(content, str):

                converted.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

                continue

            # ---------------------------------------------------------------
            # CONTENT BLOCKS
            # ---------------------------------------------------------------

            if isinstance(content, list):

                # ===========================================================
                # USER MESSAGE CONTAINING TOOL RESULTS
                # ===========================================================

                if role == "user":

                    tool_result_blocks = [
                        block
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "tool_result"
                    ]

                    if tool_result_blocks:

                        for block in tool_result_blocks:

                            tool_call_id = block.get(
                                "tool_use_id"
                            )

                            tool_content = block.get(
                                "content",
                                "",
                            )

                            if not isinstance(
                                tool_content,
                                str,
                            ):
                                try:
                                    tool_content = json.dumps(
                                        tool_content,
                                        ensure_ascii=False,
                                    )
                                except Exception:
                                    tool_content = str(
                                        tool_content
                                    )

                            tool_message: dict[str, Any] = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_content,
                            }

                            converted.append(tool_message)

                        # ---------------------------------------------------
                        # There may also be ordinary user text in the same
                        # content array.
                        # ---------------------------------------------------

                        normal_blocks = [
                            block
                            for block in content
                            if isinstance(block, dict)
                            and block.get("type")
                            != "tool_result"
                        ]

                        text_parts: list[str] = []

                        for block in normal_blocks:

                            if block.get("type") == "text":

                                text_parts.append(
                                    block.get(
                                        "text",
                                        "",
                                    )
                                )

                        if text_parts:

                            converted.append(
                                {
                                    "role": "user",
                                    "content": "".join(
                                        text_parts
                                    ),
                                }
                            )

                        continue

                # ===========================================================
                # ASSISTANT MESSAGE
                # ===========================================================

                if role == "assistant":

                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []

                    for block in content:

                        if not isinstance(block, dict):
                            continue

                        block_type = block.get("type")

                        # ---------------------------------------------------
                        # TEXT
                        # ---------------------------------------------------

                        if block_type == "text":

                            text_parts.append(
                                block.get(
                                    "text",
                                    "",
                                )
                            )

                        # ---------------------------------------------------
                        # TOOL USE
                        # ---------------------------------------------------

                        elif block_type == "tool_use":

                            tool_name = block.get("name")

                            if not tool_name:
                                logger.warning(
                                    "Ignoring tool_use without name: %s",
                                    block,
                                )
                                continue

                            tool_input = block.get(
                                "input",
                                {},
                            )

                            if not isinstance(
                                tool_input,
                                dict,
                            ):
                                tool_input = {}

                            tool_calls.append(
                                {
                                    "id": block.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(
                                            tool_input,
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            )

                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": (
                            "".join(text_parts)
                            if text_parts
                            else None
                        ),
                    }

                    if tool_calls:
                        assistant_message[
                            "tool_calls"
                        ] = tool_calls

                    converted.append(
                        assistant_message
                    )

                    continue

                # ===========================================================
                # NORMAL CONTENT BLOCKS
                # ===========================================================

                text_parts: list[str] = []

                for block in content:

                    if not isinstance(block, dict):
                        continue

                    if block.get("type") == "text":

                        text_parts.append(
                            block.get(
                                "text",
                                "",
                            )
                        )

                converted.append(
                    {
                        "role": role,
                        "content": "".join(
                            text_parts
                        ),
                    }
                )

                continue

            # ----------------------------------------------------------------
            # FALLBACK
            # ----------------------------------------------------------------

            converted.append(
                {
                    "role": role,
                    "content": str(content),
                }
            )

        return converted

    # ========================================================================
    # SAFE JSON PARSER
    # ========================================================================

    @staticmethod
    def _parse_tool_arguments(
        arguments: Any,
    ) -> dict[str, Any]:
        """
        Safely parse tool-call arguments.

        Groq normally returns arguments as a JSON string.
        """

        if arguments is None:
            return {}

        if isinstance(arguments, dict):
            return arguments

        if not isinstance(arguments, str):
            return {}

        arguments = arguments.strip()

        if not arguments:
            return {}

        try:

            parsed = json.loads(arguments)

            if isinstance(parsed, dict):
                return parsed

            logger.warning(
                "Tool arguments JSON was not an object: %r",
                parsed,
            )

            return {}

        except json.JSONDecodeError:

            logger.warning(
                "Could not parse tool arguments: %s",
                arguments,
            )

            return {}

    # ========================================================================
    # TOOL NAME VALIDATION
    # ========================================================================

    @staticmethod
    def _get_allowed_tool_names(
        tools: Optional[list[dict[str, Any]]],
    ) -> set[str]:
        """
        Return names of tools that the application actually exposed
        to the model.
        """

        if not tools:
            return set()

        names: set[str] = set()

        for tool in tools:

            if not isinstance(tool, dict):
                continue

            name = tool.get("name")

            if name:
                names.add(name)

        return names

    # ========================================================================
    # GROQ REQUEST
    # ========================================================================

    def _create_completion(
        self,
        *,
        client: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Execute a Groq request with retry handling.

        GPT-OSS can occasionally produce an invalid tool call such as:

            {"name": "JSON", ...}

        even though "JSON" was not supplied as a tool.

        Groq normally rejects this with HTTP 400.

        We retry the request with a slightly lower temperature.
        """

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):

            try:

                return client.chat.completions.create(
                    **kwargs
                )

            except Exception as exc:

                last_error = exc

                status_code = getattr(
                    exc,
                    "status_code",
                    None,
                )

                error_text = str(exc)

                is_tool_validation_error = (
                    status_code == 400
                    and (
                        "tool call validation failed"
                        in error_text.lower()
                        or "failed_generation"
                        in error_text.lower()
                        or "not in request.tools"
                        in error_text.lower()
                    )
                )

                if not is_tool_validation_error:
                    raise

                if attempt >= self.max_retries - 1:
                    raise

                logger.warning(
                    "Groq tool-call validation failed "
                    "(attempt %d/%d). Retrying with safer "
                    "generation settings.",
                    attempt + 1,
                    self.max_retries,
                )

                # Reduce randomness on retry.
                kwargs["temperature"] = max(
                    0.0,
                    float(
                        kwargs.get(
                            "temperature",
                            0.2,
                        )
                    )
                    - 0.1,
                )

                time.sleep(
                    0.25 * (attempt + 1)
                )

        if last_error:
            raise last_error

        raise RuntimeError(
            "Groq completion failed without an exception"
        )

    # ========================================================================
    # MAIN COMPLETION
    # ========================================================================

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Single-shot completion.

        Returns the structure expected by BaseA2AAgent:

            {
                "text": "...",
                "tool_uses": [...],
                "stop_reason": "...",
                "usage": {
                    "input_tokens": ...,
                    "output_tokens": ...
                }
            }
        """

        client = self._ensure_client()

        chosen_model = model or self.default_model

        # --------------------------------------------------------------------
        # CONVERT INPUT
        # --------------------------------------------------------------------

        groq_messages = self._convert_messages(
            system=system,
            messages=messages,
        )

        groq_tools = self._convert_tools(
            tools
        )

        allowed_tool_names = (
            self._get_allowed_tool_names(tools)
        )

        # --------------------------------------------------------------------
        # LOG REQUEST
        # --------------------------------------------------------------------

        logger.info(
            "Groq request model=%s tools=%s",
            chosen_model,
            sorted(
                allowed_tool_names
            ),
        )

        # --------------------------------------------------------------------
        # REQUEST PARAMETERS
        # --------------------------------------------------------------------

        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": groq_messages,

            # Groq recommends max_completion_tokens for newer API usage.
            "max_completion_tokens": max_tokens,

            "temperature": max(
                0.0,
                min(
                    2.0,
                    float(temperature),
                ),
            ),

            # GPT-OSS supports reasoning_effort.
            # "low" keeps the agent responsive while still allowing
            # reasoning for tool selection.
            "reasoning_effort": "low",

            # Do not expose internal reasoning.
            "include_reasoning": False,

            # GPT-OSS 120B currently does not support parallel tool use.
            "parallel_tool_calls": False,
        }

        # --------------------------------------------------------------------
        # TOOLS
        # --------------------------------------------------------------------

        if groq_tools:

            kwargs["tools"] = groq_tools

            # Let the model decide whether a tool is required.
            kwargs["tool_choice"] = "auto"

            # IMPORTANT:
            #
            # GPT-OSS occasionally generates a tool name such as "JSON"
            # when it is trying to format its final response.
            #
            # Groq's API normally validates that generated tool names
            # exist in request.tools and returns HTTP 400.
            #
            # We allow the API to return such a call so that this wrapper
            # can detect and safely ignore it instead of crashing the
            # entire A2A request.
            #
            # The application NEVER executes an unknown tool.
            kwargs["disable_tool_validation"] = True

        # --------------------------------------------------------------------
        # CALL GROQ
        # --------------------------------------------------------------------

        with trace_span(
            "llm.complete",
            input={
                "model": chosen_model,
                "system_preview": system[:120],
            },
            metadata={
                "temperature": kwargs["temperature"],
                "max_tokens": max_tokens,
                "tools": sorted(
                    allowed_tool_names
                ),
            },
        ) as span:

            resp = self._create_completion(
                client=client,
                kwargs=kwargs,
            )

            choice = resp.choices[0]
            message = choice.message

            # ----------------------------------------------------------------
            # TEXT
            # ----------------------------------------------------------------

            text = message.content or ""

            # ----------------------------------------------------------------
            # TOOL CALLS
            # ----------------------------------------------------------------

            tool_uses: list[dict[str, Any]] = []

            raw_tool_calls = (
                getattr(
                    message,
                    "tool_calls",
                    None,
                )
                or []
            )

            for tool_call in raw_tool_calls:

                function = getattr(
                    tool_call,
                    "function",
                    None,
                )

                if function is None:
                    logger.warning(
                        "Groq returned tool call without function: %r",
                        tool_call,
                    )
                    continue

                tool_name = getattr(
                    function,
                    "name",
                    None,
                )

                arguments = getattr(
                    function,
                    "arguments",
                    "{}",
                )

                # ------------------------------------------------------------
                # UNKNOWN TOOL
                # ------------------------------------------------------------

                if (
                    tool_name
                    and tool_name not in allowed_tool_names
                ):

                    logger.warning(
                        "Groq generated unknown tool '%s'. "
                        "Allowed tools=%s. Ignoring unknown "
                        "tool call instead of executing it.",
                        tool_name,
                        sorted(
                            allowed_tool_names
                        ),
                    )

                    # Do NOT pass this fake tool to BaseA2AAgent.
                    #
                    # This is important because "JSON" is not an
                    # application tool.
                    continue

                # ------------------------------------------------------------
                # VALID TOOL
                # ------------------------------------------------------------

                parsed_input = (
                    self._parse_tool_arguments(
                        arguments
                    )
                )

                logger.info(
                    "Groq tool call: %s(%s)",
                    tool_name,
                    parsed_input,
                )

                tool_uses.append(
                    {
                        "id": getattr(
                            tool_call,
                            "id",
                            None,
                        ),
                        "name": tool_name,
                        "input": parsed_input,
                    }
                )

            # ----------------------------------------------------------------
            # FINISH REASON
            # ----------------------------------------------------------------

            stop_reason = getattr(
                choice,
                "finish_reason",
                None,
            )

            # ----------------------------------------------------------------
            # USAGE
            # ----------------------------------------------------------------

            usage = getattr(
                resp,
                "usage",
                None,
            )

            input_tokens = 0
            output_tokens = 0

            if usage:

                input_tokens = getattr(
                    usage,
                    "prompt_tokens",
                    0,
                ) or 0

                output_tokens = getattr(
                    usage,
                    "completion_tokens",
                    0,
                ) or 0

            # ----------------------------------------------------------------
            # IMPORTANT FALLBACK
            # ----------------------------------------------------------------
            #
            # If Groq produced only an unknown tool such as "JSON", then
            # tool_uses is empty.
            #
            # In that situation BaseA2AAgent should see a normal response
            # rather than attempting to execute "JSON".
            #
            # If the model returned no visible text either, provide a
            # controlled fallback rather than crashing.
            # ----------------------------------------------------------------

            if raw_tool_calls and not tool_uses and not text.strip():

                logger.warning(
                    "Groq returned only unknown/invalid tool calls. "
                    "Returning controlled fallback."
                )

                text = (
                    "I retrieved the requested information, "
                    "but the final response could not be generated."
                )

                stop_reason = "stop"

            # ----------------------------------------------------------------
            # RESULT
            # ----------------------------------------------------------------

            result = {
                "text": text,
                "tool_uses": tool_uses,
                "stop_reason": stop_reason,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }

            # ----------------------------------------------------------------
            # TRACE OUTPUT
            # ----------------------------------------------------------------

            span["output"] = {
                "stop_reason": result[
                    "stop_reason"
                ],
                "usage": result[
                    "usage"
                ],
                "tool_count": len(
                    tool_uses
                ),
                "text_preview": (
                    result["text"][:160]
                ),
            }

            return result
