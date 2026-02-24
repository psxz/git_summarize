"""
app/core/observability.py
──────────────────────────
Arize Phoenix / OpenTelemetry tracing for LLM observability.

Automatically instruments all LangChain calls (ainvoke, astream, etc.)
via OpenInference and exports traces to a Phoenix instance over OTLP/gRPC.

Phoenix provides:
  • Full LLM trace visualization (prompts, completions, latencies, token counts)
  • Evaluation tools for LLM output quality
  • Dataset management and experiments
  • 100% self-hosted, zero-cost, data stays on your infrastructure

Setup:
  Set PHOENIX_COLLECTOR_ENDPOINT in .env (default: http://localhost:6006)
  Run Phoenix via Docker:  docker compose up phoenix

Public API:
  init_observability()   Call once at startup (lifespan handler).
  shutdown_observability()  Call at shutdown (flushes pending spans).
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_tracer_provider: object | None = None


def init_observability() -> bool:
    """
    Initialize OpenTelemetry tracing with Phoenix as the backend.

    Returns True if tracing was successfully enabled, False otherwise.
    Safe to call even if Phoenix is unreachable — traces will be dropped
    silently after the exporter timeout.
    """
    global _tracer_provider

    settings = get_settings()
    endpoint = settings.phoenix_collector_endpoint

    if not endpoint:
        logger.info(
            "phoenix_disabled",
            msg="PHOENIX_COLLECTOR_ENDPOINT not set. LLM tracing is off.",
        )
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )

        # ── Resource identifies this service in the Phoenix UI ────────────
        resource = Resource.create(
            {
                "service.name": "github-summarizer",
                "service.version": "1.0.0",
                "deployment.environment": "production",
            }
        )

        # ── OTLP exporter → Phoenix collector ─────────────────────────────
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)

        # ── Tracer provider ───────────────────────────────────────────────
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        # ── Auto-instrument LangChain ─────────────────────────────────────
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=provider)

        logger.info(
            "phoenix_tracing_enabled",
            endpoint=endpoint,
            msg="All LangChain calls are now traced to Phoenix.",
        )
        return True

    except ImportError as exc:
        logger.warning(
            "phoenix_import_error",
            error=str(exc),
            msg="OpenTelemetry/OpenInference packages not installed. Tracing disabled.",
        )
        return False
    except Exception as exc:
        logger.warning(
            "phoenix_init_error",
            error=str(exc),
            msg="Failed to initialize Phoenix tracing. Continuing without observability.",
        )
        return False


def shutdown_observability() -> None:
    """Flush pending spans and shut down the tracer provider."""
    global _tracer_provider

    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_millis=5000)  # type: ignore[union-attr]
            _tracer_provider.shutdown()  # type: ignore[union-attr]
            logger.info("phoenix_shutdown", msg="Tracer provider shut down cleanly.")
        except Exception as exc:
            logger.warning("phoenix_shutdown_error", error=str(exc))
        finally:
            _tracer_provider = None
