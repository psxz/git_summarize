"""
app/core/logging.py
────────────────────
Production-grade structured logging via structlog.

Design goals
────────────
  • One log format decision: JSON in production, coloured console in dev.
    "Production" is detected automatically — Render sets RENDER=true;
    any non-TTY stdout also triggers JSON mode.

  • Consistent structure everywhere: every log line carries at minimum
    { "timestamp", "level", "logger", "event", ...kwargs }.

  • Third-party libraries (uvicorn, httpx, langchain, redis) are silenced
    or redirected through structlog so their records share the same format.

  • High-volume events (cache hits, refine steps) are sampled so they
    don't dominate log streams under load.

  • Health-check polling (/api/v1/health) is filtered from access logs.

  • Request context (request_id, path, method) is injected once per
    request and merged automatically into every log record in that
    async context via structlog's contextvars support.

Public API
──────────
  configure_logging()           Call once at application startup (lifespan).
  get_logger(name)              Returns a bound structlog logger for a module.
  bind_request_context(**kw)    Adds keys to the current request's log context.
  clear_request_context()       Clears per-request context (end of request).
  get_uvicorn_log_config()      Returns a logging.config dict for Uvicorn.
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import get_settings

# ── Environment detection ─────────────────────────────────────────────────────


def _is_production() -> bool:
    """
    True when running on Render (RENDER=true) or when stdout is not a TTY.
    In both cases we emit newline-delimited JSON — no ANSI, no colours.
    Render's log aggregator parses JSON natively and surfaces structured fields
    in its dashboard.
    """
    if os.getenv("RENDER", "").lower() in {"true", "1", "yes"}:
        return True
    return not sys.stdout.isatty()


# ── Module-level log level overrides ─────────────────────────────────────────
# Quiet down chatty third-party libraries without touching our own loggers.

_THIRD_PARTY_LEVELS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "hpack": logging.WARNING,
    "openai": logging.WARNING,
    "anthropic": logging.WARNING,
    "langchain": logging.WARNING,
    "langchain_core": logging.WARNING,
    "langchain_openai": logging.WARNING,
    "langchain_anthropic": logging.WARNING,
    "langchain_google_genai": logging.WARNING,
    "redis": logging.WARNING,
    "duckdb": logging.WARNING,
    "urllib3": logging.WARNING,
    "charset_normalizer": logging.WARNING,
    # Uvicorn access log is handled by _HealthCheckFilter below;
    # the error/warning log stays at WARNING.
    "uvicorn.error": logging.WARNING,
    "uvicorn.access": logging.INFO,  # filtered, not silenced
}


# ── Custom processors ─────────────────────────────────────────────────────────


class _AddServiceContext:
    """
    Injects static service-level fields into every log record.
    Reads from config once and caches — safe for long-lived processes.
    """

    _fields: dict[str, Any] | None = None

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        if self._fields is None:
            try:
                settings = get_settings()
                type(self)._fields = {
                    "service": "github-summarizer",
                    "llm_provider": settings.default_llm_provider.value,
                    "env": "production" if _is_production() else "development",
                }
            except Exception:
                type(self)._fields = {"service": "github-summarizer"}
        event_dict.update(self._fields)
        return event_dict


class _SuppressHealthChecks:
    """
    Drops uvicorn access log records for the health-check endpoint.
    High-frequency polling (load balancers, Render's health probe) would
    otherwise flood the log stream with zero-value noise.
    """

    _SUPPRESSED_PATHS = {"/api/v1/health", "/health", "/"}

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        path = event_dict.get("path", "") or ""
        if path in self._SUPPRESSED_PATHS and event_dict.get("status", 0) == 200:
            raise structlog.DropEvent()
        return event_dict


class _SamplingFilter:
    """
    Rate-limits high-volume event names to 1-in-N to keep log volume
    manageable under sustained load while preserving observability.

    Sampled events still appear — they're just thinned.
    100 % of ERROR and WARNING records always pass through.
    """

    _HIGH_VOLUME = {
        "cache_hit_redis": 10,  # log 1 in 10
        "cache_hit_memory": 10,
        "cast_chunked": 5,
        "refine_step": 3,
    }
    _counters: dict[str, int] = {}

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        if method in {"error", "warning", "critical"}:
            return event_dict

        event = event_dict.get("event", "")
        rate = self._HIGH_VOLUME.get(event)
        if rate is None:
            return event_dict

        count = self._counters.get(event, 0) + 1
        self._counters[event] = count

        if count % rate != 0:
            raise structlog.DropEvent()

        event_dict["_sampled"] = f"1/{rate}"
        return event_dict


class _CallsiteInfo:
    """
    Adds { module, func_name, lineno } to every record in dev mode.
    Skipped in production to save bytes.
    """

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        frame = sys._getframe(6)  # walk past structlog internals
        event_dict["module"] = frame.f_globals.get("__name__", "unknown")
        event_dict["func_name"] = frame.f_code.co_name
        event_dict["lineno"] = frame.f_lineno
        return event_dict


# ── stdlib → structlog bridge ─────────────────────────────────────────────────


class _StructlogHandler(logging.Handler):
    """
    Redirects stdlib logging records (from uvicorn, httpx, langchain, etc.)
    through structlog so they share the same format and context variables.

    The stdlib record's `name` becomes the structlog `logger` field,
    and the original log level is preserved.
    """

    _LEVEL_TO_METHOD = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }

    def emit(self, record: logging.LogRecord) -> None:
        # Pull the structlog method name from the stdlib level
        method = self._LEVEL_TO_METHOD.get(record.levelno, "info")
        bound_logger = structlog.get_logger(record.name)

        kwargs: dict[str, Any] = {}
        if record.exc_info:
            kwargs["exc_info"] = record.exc_info

        getattr(bound_logger, method)(
            self.format(record),
            **kwargs,
        )


# ── Uvicorn access log reformatter ────────────────────────────────────────────


class _UvicornAccessReformatter(logging.Filter):
    """
    Parses uvicorn's access log string into structured fields so the
    middleware's X-Process-Time-ms header is preserved in logs.

    uvicorn emits: '127.0.0.1:52000 - "GET /api/v1/health HTTP/1.1" 200'
    We parse this into { client, method, path, http_version, status }.
    """

    import re

    _PATTERN = re.compile(
        r'(?P<client>\S+) - "(?P<method>\w+) (?P<path>\S+) (?P<proto>[^"]+)" (?P<status>\d+)'
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        m = self._PATTERN.match(msg)
        if m:
            record.path = m.group("path")
            record.method = m.group("method")
            record.status = int(m.group("status"))
            record.client = m.group("client")
            # Health check suppression at the stdlib level
            if record.path in {"/api/v1/health", "/health", "/"} and record.status == 200:
                return False
        return True


# ── Core configuration ────────────────────────────────────────────────────────


def _build_processors(*, production: bool, log_level: int) -> list:
    """
    Build the structlog processor chain appropriate for the environment.

    Production chain (JSON):
      contextvars → service fields → sampling → level → timestamp
      → stack info → exc info → unicode → JSON

    Development chain (coloured console):
      contextvars → callsite → service fields → level → timestamp
      → stack info → exc info → ConsoleRenderer
    """
    shared_pre: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _AddServiceContext(),
    ]

    if not production:
        # Callsite info only in dev — too expensive and verbose for prod
        shared_pre.insert(1, _CallsiteInfo())

    shared_post: list = [
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if production:
        renderer = structlog.processors.JSONRenderer()
        sampling = [_SamplingFilter()]  # only sample in prod
        health = [_SuppressHealthChecks()]
        return shared_pre + sampling + health + shared_post + [renderer]
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
            sort_keys=False,
        )
        return shared_pre + shared_post + [renderer]


def configure_logging() -> None:
    """
    Configure structlog + stdlib logging.  Call once at application startup
    (inside the FastAPI lifespan handler).  Safe to call multiple times —
    subsequent calls are no-ops after the first configuration is applied.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    production = _is_production()

    # ── 1. Configure structlog ─────────────────────────────────────────────
    structlog.configure(
        processors=_build_processors(production=production, log_level=log_level),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── 2. Root stdlib logger → structlog bridge ───────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any existing handlers to prevent duplicate output
    root.handlers.clear()

    bridge = _StructlogHandler()
    bridge.setLevel(log_level)
    # In prod the structlog chain renders JSON; the bridge's own formatter
    # is only invoked for the message string extraction, so keep it plain.
    bridge.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(bridge)

    # ── 3. Per-module level overrides ──────────────────────────────────────
    for module, level in _THIRD_PARTY_LEVELS.items():
        logging.getLogger(module).setLevel(level)

    # ── 4. Uvicorn access log reformatter ──────────────────────────────────
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(_UvicornAccessReformatter())

    # ── 5. Startup confirmation ────────────────────────────────────────────
    _startup_log = structlog.get_logger("app.core.logging")
    _startup_log.info(
        "logging_configured",
        level=settings.log_level.upper(),
        format="json" if production else "console",
        environment="production" if production else "development",
        render_detected=os.getenv("RENDER", "false"),
    )


# ── Request context helpers ───────────────────────────────────────────────────


def bind_request_context(**kwargs: Any) -> None:
    """
    Bind key-value pairs to the current async request context.
    Every subsequent log call in the same request will include these fields.

    Typical usage in middleware:
        bind_request_context(request_id="abc", method="POST", path="/api/v1/summarize")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """
    Clear all per-request context variables.
    Call at the end of each request (after response is sent).
    """
    structlog.contextvars.clear_contextvars()


# ── Public logger factory ─────────────────────────────────────────────────────


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    Return a structlog BoundLogger bound to the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("event_name", key="value", count=42)

    The name is attached as the "logger" field in every emitted record,
    enabling per-module filtering in log aggregators (Datadog, CloudWatch,
    Render's log dashboard).
    """
    return structlog.get_logger(name)


# ── Uvicorn log configuration ─────────────────────────────────────────────────


def get_uvicorn_log_config() -> dict:
    """
    Returns a logging.config dict suitable for passing to
    uvicorn.run(log_config=...) or the --log-config CLI flag.

    Disables uvicorn's default handlers so all output flows through our
    structlog bridge, preventing duplicate / inconsistently-formatted lines.

    Usage in start command (Dockerfile / render.yaml):
        uvicorn app.main:app --log-config log_config.json

    Or programmatically:
        uvicorn.run(app, log_config=get_uvicorn_log_config())
    """
    settings = get_settings()
    level = settings.log_level.upper()

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(message)s",
                "use_colors": False,
            },
        },
        "handlers": {
            # Null handler — our bridge on the root logger catches everything
            "null": {
                "class": "logging.NullHandler",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["null"], "level": level, "propagate": True},
            "uvicorn.error": {"handlers": ["null"], "level": "WARNING", "propagate": True},
            "uvicorn.access": {"handlers": ["null"], "level": level, "propagate": True},
        },
    }
