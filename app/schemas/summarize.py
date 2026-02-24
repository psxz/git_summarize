"""
app/schemas/summarize.py
─────────────────────────
Pydantic v2 request and response schemas.
These power both FastAPI validation and the auto-generated Swagger/ReDoc docs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ── Request ───────────────────────────────────────────────────────────────────


class LLMProviderChoice(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    NEBIUS = "nebius"


class SummarizationStrategyChoice(str, Enum):
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"


class SummarizeRequest(BaseModel):
    """Request body for the summarization endpoint."""

    repo_url: str = Field(
        ...,
        description=(
            "Full GitHub repository URL. "
            "Accepts HTTPS (https://github.com/owner/repo) "
            "or SSH (git@github.com:owner/repo.git) formats."
        ),
        examples=["https://github.com/tiangolo/fastapi"],
    )
    llm_provider: LLMProviderChoice | None = Field(
        None,
        description="Override the default LLM provider for this request.",
    )
    llm_model: str | None = Field(
        None,
        description="Override the model name (e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022').",
    )
    strategy: SummarizationStrategyChoice | None = Field(
        None,
        description="Summarization strategy. Defaults to service config value.",
    )
    include_file_tree: bool = Field(
        False,
        description="Include a condensed file-tree section in the summary.",
    )
    max_files: int | None = Field(
        None,
        ge=1,
        le=50,
        description="Override maximum number of supplementary files to scan.",
    )

    @field_validator("repo_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        import re

        v = v.strip()
        # Accept HTTPS and SSH GitHub URLs
        patterns = [
            r"^https?://github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?$",
            r"^git@github\.com:[\w.\-]+/[\w.\-]+(\.git)?$",
        ]
        if not any(re.match(p, v) for p in patterns):
            raise ValueError(
                "repo_url must be a valid GitHub URL, e.g. https://github.com/owner/repo"
            )
        return v


# ── Sub-models ────────────────────────────────────────────────────────────────


class RepoMetadata(BaseModel):
    owner: str
    name: str
    full_name: str
    description: str | None = None
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    topics: list[str] = []
    default_branch: str = "main"
    language: str | None = None
    languages: dict[str, int] = {}
    is_archived: bool = False
    license_name: str | None = None
    homepage: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SummarizationMeta(BaseModel):
    strategy_used: str
    llm_provider: str
    llm_model: str
    total_input_tokens_estimated: int
    chunks_processed: int
    files_scanned: int
    cache_hit: bool = False


# ── Response ──────────────────────────────────────────────────────────────────


class SummarizeResponse(BaseModel):
    """Full summarization response returned by POST /summarize."""

    repo_url: str
    metadata: RepoMetadata
    summary: str = Field(
        ...,
        description="The generated technical summary of the repository.",
    )
    key_technologies: list[str] = Field(
        default_factory=list,
        description="Detected primary technologies and frameworks.",
    )
    file_tree_snippet: str | None = Field(
        None,
        description="Condensed file tree (only if include_file_tree=true).",
    )
    summarization_meta: SummarizationMeta
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during processing.",
    )


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    github_authenticated: bool
    llm_provider: str
    cache_connected: bool


class RateLimitResponse(BaseModel):
    core_limit: int
    core_remaining: int
    core_reset_utc: str
    search_limit: int
    search_remaining: int
