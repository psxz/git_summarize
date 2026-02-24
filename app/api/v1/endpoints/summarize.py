"""
app/api/v1/endpoints/summarize.py
──────────────────────────────────
FastAPI router defining all v1 endpoints.

POST /summarize         — full JSON summary (with caching)
POST /summarize/stream  — SSE streaming summary
GET  /health            — service health check
GET  /rate-limits       — GitHub rate limit status
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import AuthInfo, verify_api_key
from app.schemas.summarize import (
    HealthResponse,
    RateLimitResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.cache import (
    build_request_cache_key,
    cache_get,
    cache_set,
    is_cache_connected,
)
from app.services.github_client import GitHubClient
from app.services.summarizer import summarize_repo, summarize_repo_stream

logger = get_logger(__name__)
router = APIRouter()
_background_tasks: set[object] = set()  # prevent GC of fire-and-forget tasks


# ── POST /summarize ───────────────────────────────────────────────────────────


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarise a GitHub Repository",
    description=(
        "Accepts a GitHub repository URL and returns a structured technical "
        "intelligence report including a prose summary, detected technologies, "
        "repository metadata, and processing statistics. "
        "Responses are cached for efficiency."
    ),
    status_code=status.HTTP_200_OK,
)
async def summarize_endpoint(
    request: SummarizeRequest,
    auth: AuthInfo = Depends(verify_api_key),
) -> SummarizeResponse:
    """Full summarization with Redis-backed caching."""
    cache_key = build_request_cache_key(request.model_dump())

    # Cache read
    cached = await cache_get(cache_key)
    if cached:
        response = SummarizeResponse(**cached)
        response.summarization_meta.cache_hit = True
        return response

    try:
        result = await summarize_repo(request, github_token=auth.github_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("summarize_endpoint_error", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {exc}",
        )

    # Cache write (non-blocking)
    import asyncio

    task = asyncio.create_task(cache_set(cache_key, result.model_dump()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return result


# ── POST /summarize/stream ────────────────────────────────────────────────────


@router.post(
    "/summarize/stream",
    summary="Stream a Repository Summary via SSE",
    description=(
        "Returns a Server-Sent Events stream. "
        "Events: `status` (progress), `chunk` (summary tokens), "
        "`done` (final metadata), `error` (on failure)."
    ),
)
async def summarize_stream_endpoint(
    request: SummarizeRequest,
    auth: AuthInfo = Depends(verify_api_key),
) -> EventSourceResponse:
    """SSE streaming endpoint — yields incremental summary tokens."""

    async def _generator() -> AsyncIterator[dict]:
        try:
            async for raw_event in summarize_repo_stream(request, github_token=auth.github_token):
                event_data = json.loads(raw_event)
                yield {
                    "event": event_data["event"],
                    "data": json.dumps(event_data["data"]),
                }
        except Exception as exc:
            logger.error("stream_generator_error", error=str(exc))
            yield {"event": "error", "data": json.dumps(str(exc))}

    return EventSourceResponse(_generator())


# ── GET /health ───────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns service health status including GitHub auth and cache connectivity.",
)
async def health_check() -> HealthResponse:
    settings = get_settings()
    client = GitHubClient()

    # Check GitHub auth
    github_ok = False
    try:
        limits = await client.get_rate_limits()
        github_ok = limits.get("rate", {}).get("limit", 0) > 60
    except Exception:
        pass

    cache_ok = await is_cache_connected()

    return HealthResponse(
        status="ok" if github_ok else "degraded",
        github_authenticated=github_ok,
        llm_provider=settings.default_llm_provider.value,
        cache_connected=cache_ok,
    )


# ── GET /rate-limits ──────────────────────────────────────────────────────────


@router.get(
    "/rate-limits",
    response_model=RateLimitResponse,
    summary="GitHub Rate Limit Status",
    description="Returns current GitHub API rate limit state for the configured token.",
)
async def rate_limits_endpoint(
    auth: AuthInfo = Depends(verify_api_key),
) -> RateLimitResponse:
    client = GitHubClient(github_token=auth.github_token)
    try:
        data = await client.get_rate_limits()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch GitHub rate limits: {exc}",
        )

    from datetime import datetime

    core = data.get("resources", {}).get("core", {})
    search = data.get("resources", {}).get("search", {})

    reset_ts = core.get("reset", 0)
    reset_utc = datetime.fromtimestamp(reset_ts, tz=UTC).isoformat()

    return RateLimitResponse(
        core_limit=core.get("limit", 0),
        core_remaining=core.get("remaining", 0),
        core_reset_utc=reset_utc,
        search_limit=search.get("limit", 0),
        search_remaining=search.get("remaining", 0),
    )
