"""
app/main.py
────────────
FastAPI application factory.

Key features:
  • Lifespan handler (startup/shutdown) for logging init and observability hooks
  • CORS middleware
  • Request-ID middleware for distributed tracing
  • Global exception handlers
  • Versioned router mounting (/api/v1/...)
  • Interactive docs at /docs (Swagger) and /redoc (ReDoc)
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    get_uvicorn_log_config,
)

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs at startup and shutdown."""
    # Startup
    configure_logging()
    settings = get_settings()

    # Enable LangSmith tracing if configured
    if settings.langchain_api_key and settings.langchain_tracing_v2 == "true":
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("langsmith_tracing_enabled", project=settings.langchain_project)

    # Enable Arize Phoenix tracing if configured
    from app.core.observability import init_observability, shutdown_observability
    init_observability()

    logger.info(
        "app_startup",
        provider=settings.default_llm_provider.value,
        model=settings.default_llm_model,
        strategy=settings.summarization_strategy.value,
    )

    yield  # ← application runs here

    # Shutdown
    shutdown_observability()
    logger.info("app_shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="GitHub Repository Summarization API",
        description=(
            "Transform any GitHub repository URL into a structured technical "
            "intelligence report. Powered by LLMs with Map-Reduce and Iterative "
            "Refinement summarization strategies."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-ID & timing middleware ────────────────────────────────────
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        # Bind structured fields to this request's log context
        clear_request_context()
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = str(elapsed_ms)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )
        return response

    # ── Global exception handlers ─────────────────────────────────────────
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred. Please try again."},
        )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(api_router)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    return app


app = create_app()
