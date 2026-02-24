# 🔍 GitHub Repository Summarization API

A production-grade FastAPI service that transforms any GitHub repository URL into a 
structured technical intelligence report using LLMs. Built on the architecture described
in the research document, implementing Map-Reduce/Iterative Refinement summarization,
rate-limit mitigation, markdown sanitization, streaming SSE, and multi-LLM support.

## Features
- ✅ GitHub REST API integration (metadata, README, languages, stats)
- ✅ Exponential backoff & rate-limit mitigation
- ✅ Markdown sanitization pipeline (badge/link stripping)
- ✅ Map-Reduce & Iterative Refinement summarization strategies
- ✅ Streaming responses via Server-Sent Events (SSE)
- ✅ Multi-LLM provider support (OpenAI, Anthropic, Google Gemini)
- ✅ Redis-backed caching
- ✅ Pydantic v2 schema validation
- ✅ Structured logging + LangSmith tracing hooks
- ✅ API key authentication & input sanitization
- ✅ Interactive Swagger/ReDoc docs

## Quickstart

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

cp .env.example .env       # fill in your keys
uv sync                    # creates .venv and installs all dependencies
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs

### Common uv commands

```bash
uv sync                    # install / update deps from uv.lock
uv sync --no-dev           # production-only deps
uv add <package>           # add a dependency to pyproject.toml + uv.lock
uv remove <package>        # remove a dependency
uv run pytest              # run tests inside the managed venv
uv run ruff check app/     # lint
uv run ruff format app/    # format
uv lock                    # regenerate uv.lock after manual pyproject.toml edits
```

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GITHUB_TOKEN` | GitHub PAT (classic or fine-grained) | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `GOOGLE_API_KEY` | Google Gemini API key | Optional |
| `DEFAULT_LLM_PROVIDER` | `openai` / `anthropic` / `google` | Yes |
| `REDIS_URL` | Redis connection URL | Optional |
| `API_SECRET_KEY` | Key for Bearer auth on this API | Optional |
| `LANGCHAIN_API_KEY` | LangSmith tracing | Optional |

## Architecture

```
POST /api/v1/summarize          → Full summary (JSON)
POST /api/v1/summarize/stream   → SSE streaming summary
GET  /api/v1/health             → Health check
GET  /api/v1/rate-limits        → GitHub rate limit status
```
