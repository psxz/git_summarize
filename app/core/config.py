"""
app/core/config.py
──────────────────
Centralised, type-safe settings loaded from environment variables or .env file.
Pydantic-settings v2 style.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    NEBIUS = "nebius"  # Nebius Token Factory — MiniMaxAI/MiniMax-M2.1 and others


class SummarizationStrategy(str, Enum):
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── GitHub ──────────────────────────────────────────────────────────
    github_token: str = Field(..., description="GitHub Personal Access Token")
    github_api_base: str = "https://api.github.com"
    github_api_version: str = "2022-11-28"

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    # Nebius Token Factory — OpenAI-compatible inference endpoint
    # Get your key at: https://studio.nebius.com/settings/api-keys
    nebius_api_key: str | None = None
    nebius_api_base: str = "https://api.studio.nebius.com/v1/"
    default_llm_provider: LLMProvider = LLMProvider.NEBIUS
    default_llm_model: str = "MiniMaxAI/MiniMax-M2.1"
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(2048, gt=0)

    # ── Summarization ────────────────────────────────────────────────────
    summarization_strategy: SummarizationStrategy = SummarizationStrategy.MAP_REDUCE
    chunk_size_tokens: int = Field(1500, gt=0)
    chunk_overlap_tokens: int = Field(150, ge=0)

    # ── Cache ─────────────────────────────────────────────────────────────
    redis_url: str | None = None
    cache_ttl_seconds: int = Field(3600, gt=0)

    # ── API Security ──────────────────────────────────────────────────────
    # Legacy static Bearer key — used only when LOGTO_ENDPOINT is not set.
    api_secret_key: str | None = None

    # ── Logto OIDC / JWT Auth ─────────────────────────────────────────────────
    # Set LOGTO_ENDPOINT to enable Logto JWT validation.
    # Leave blank to fall back to static API_SECRET_KEY (dev / open mode).
    logto_endpoint: str | None = Field(
        None,
        description=(
            "Logto tenant base URL, e.g. https://my-tenant.logto.app. "
            "JWKS URI -> {logto_endpoint}/oidc/jwks  "
            "Issuer   -> {logto_endpoint}/oidc"
        ),
    )
    logto_api_resource: str | None = Field(
        None,
        description=(
            "API resource indicator registered in Logto, "
            "e.g. https://api.yourapp.com. Used to verify the aud claim."
        ),
    )
    logto_required_scopes: list[str] = Field(
        default_factory=list,
        description=(
            "Scopes required in every access token, "
            "e.g. ['repo:summarize']. Empty list = no scope enforcement."
        ),
    )
    # Custom JWT claim populated by a Logto JWT-claims script, e.g.:
    #   { github_access_token: context.social?.github?.accessToken }
    logto_github_token_claim: str = Field(
        "github_access_token",
        description=(
            "Name of the custom JWT claim carrying the user's GitHub "
            "access token. Populated via Logto's JWT-claims script."
        ),
    )

    # ── Observability ─────────────────────────────────────────────────────
    langchain_tracing_v2: str | None = None
    langchain_api_key: str | None = None
    langchain_project: str = "github-summarizer"
    log_level: str = "INFO"

    # Arize Phoenix (self-hosted LLM observability)
    phoenix_collector_endpoint: str | None = Field(
        None,
        description=(
            "OTLP/gRPC endpoint of your Phoenix instance, "
            "e.g. 'http://localhost:4317'. Leave unset to disable."
        ),
    )

    # ── Runtime Limits ────────────────────────────────────────────────────
    max_files_to_scan: int = Field(20, gt=0)
    max_file_size_bytes: int = Field(102_400, gt=0)  # 100 KB
    max_total_tokens: int = Field(80_000, gt=0)

    @field_validator("github_token")
    @classmethod
    def token_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("GITHUB_TOKEN must not be empty")
        return v.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of settings."""
    return Settings()  # type: ignore[call-arg]
