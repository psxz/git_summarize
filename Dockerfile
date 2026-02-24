# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build using uv for deterministic, fast dependency installation.
#
# Stage 1 (builder)  — install uv, sync dependencies into /app/.venv
# Stage 2 (runtime)  — copy only the venv + source; no build tools
#
# uv installs from uv.lock (committed to VCS) so the build is fully
# reproducible without a separate pip freeze step.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Copy uv from the official distroless image — fastest install method,
# no curl/shell required.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (separate layer → cached unless lock file changes)
COPY pyproject.toml uv.lock ./

# --frozen         → fail if uv.lock is out of sync with pyproject.toml
# --no-install-project → skip installing the project itself (app/ not copied yet)
# --no-dev         → exclude [dependency-groups.dev]
# --compile-bytecode → pre-compile .pyc files for faster cold starts
RUN uv sync \
        --frozen \
        --no-install-project \
        --no-dev \
        --compile-bytecode

# Now copy application source and install the project into the same venv
COPY app/ ./app/
RUN uv sync \
        --frozen \
        --no-dev \
        --compile-bytecode


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application source
COPY --chown=appuser:appuser app/ ./app/

# Activate the venv by prepending it to PATH — no `source activate` needed
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

# Uvicorn production flags:
#  --workers 4      → (2 × CPU cores) + 1 is the recommended formula
#  --loop uvloop    → fastest asyncio event loop
#  --http httptools → fast HTTP parser
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log"]
