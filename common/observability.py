"""
Observability: Langfuse + LangSmith.

WHY BOTH:
  - Langfuse: self-hostable, open source. Best for prompt management,
    cost tracking, evals, and team-shared dashboards. Recommended for prod.
  - LangSmith: hosted by LangChain. Tightly integrated with LangChain/LangGraph,
    excellent for rapid debugging during development.

DESIGN: We expose a unified `Tracer` facade. Both backends become no-ops
if their env vars are missing — so local dev "just works" without setup.

The decorator @observe() from langfuse handles trace propagation automatically.
For LangSmith, setting the env vars below makes LangChain/LangGraph emit traces
without code changes. Our explicit spans are for non-LangChain code paths
(direct Anthropic SDK calls, MCP tool invocations).
"""
from __future__ import annotations

import os
import time
import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from common.logging_utils import get_logger, log_with_extra
import logging

logger = get_logger(__name__)

# --- Langfuse (lazy-imported so missing package won't break import) ---
_langfuse_client = None
_langfuse_enabled = False

def _init_langfuse() -> None:
    global _langfuse_client, _langfuse_enabled
    if _langfuse_client is not None:
        return
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not (pk and sk):
        logger.info("Langfuse disabled (LANGFUSE_PUBLIC_KEY/SECRET_KEY not set)")
        return
    try:
        from langfuse import Langfuse  # type: ignore
        _langfuse_client = Langfuse(public_key=pk, secret_key=sk, host=host)
        _langfuse_enabled = True
        logger.info(f"Langfuse enabled at {host}")
    except ImportError:
        logger.warning("langfuse package not installed; tracing disabled")
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")


def _init_langsmith() -> None:
    """LangSmith activates via environment variables — no client object needed.

    pydantic-settings loads .env into the Settings object but does NOT write
    back to os.environ. We must do that explicitly so LangGraph's auto-tracing
    (which reads os.environ directly) can see the credentials.
    """
    from config.settings import settings  # local import avoids circular deps at module load

    api_key = settings.langchain_api_key or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.info("LangSmith disabled (LANGCHAIN_API_KEY not set)")
        return

    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    logger.info(
        f"LangSmith enabled, project={settings.langchain_project}, "
        f"endpoint={settings.langchain_endpoint}"
    )


def init_observability() -> None:
    """Call once at process startup."""
    _init_langfuse()
    _init_langsmith()


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    """
    Generic span — works whether Langfuse is on or off.

    Usage:
        with trace_span("classify_claim", input={"claim_id": cid}) as span:
            result = do_work()
            span["output"] = result
    """
    span_data: dict[str, Any] = {"output": None, "metadata": metadata or {}}
    started = time.time()
    lf_span = None

    if _langfuse_enabled and _langfuse_client is not None:
        try:
            # Langfuse v2 API — adjust to v3 if needed
            lf_span = _langfuse_client.span(
                name=name, input=input, metadata=metadata, trace_id=trace_id
            )
        except Exception as e:
            logger.debug(f"Langfuse span creation failed: {e}")

    try:
        yield span_data
    except Exception as e:
        span_data["error"] = str(e)
        if lf_span:
            try:
                lf_span.end(output={"error": str(e)}, level="ERROR")
            except Exception:
                pass
        raise
    else:
        elapsed_ms = int((time.time() - started) * 1000)
        log_with_extra(
            logger, logging.INFO, f"span:{name}",
            duration_ms=elapsed_ms, span=name,
        )
        if lf_span:
            try:
                lf_span.end(output=span_data.get("output"))
            except Exception:
                pass


def observe(name: Optional[str] = None) -> Callable:
    """Decorator wrapping a function in a trace span."""
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        if functools.WRAPPER_ASSIGNMENTS and hasattr(fn, "__call__"):
            import asyncio
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    with trace_span(span_name, input={"args": str(args)[:200]}) as span:
                        result = await fn(*args, **kwargs)
                        span["output"] = result
                        return result
                return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, input={"args": str(args)[:200]}) as span:
                result = fn(*args, **kwargs)
                span["output"] = result
                return result
        return sync_wrapper
    return decorator
