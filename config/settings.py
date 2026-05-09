"""
Centralized settings.

WHY pydantic-settings:
  - Type-validated env vars (no more KeyError at runtime)
  - .env file support for local dev
  - Single source of truth — every component imports `settings` from here

PRINCIPLE: secrets via env, not files. Code never sees an api_key string literal.
"""
from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
    )

    # --- Service URLs ---------------------------------------------------
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    claims_agent_url: str = "http://localhost:8001"
    policy_agent_url: str = "http://localhost:8002"

    # --- LLM ------------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    default_model: str = "claude-sonnet-4-5"

    # --- Auth -----------------------------------------------------------
    # Bearer token clients use to call the gateway. Rotate via secret manager in prod.
    api_bearer_token: str = Field(default="dev-local-token-change-me")

    # --- Observability --------------------------------------------------
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langchain_api_key: Optional[str] = None  # LangSmith
    langchain_project: str = "a2a-insurance-system"
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # --- Behavior -------------------------------------------------------
    log_level: str = "INFO"
    request_timeout_s: float = 30.0
    max_concurrent_agent_calls: int = 10


settings = Settings()
