# GitHub Repository Summarizer

Takes a GitHub repo URL, pulls the README + key files, runs them through an LLM
(Nebius AI Studio by default), and returns a structured summary — what the project does, its technology stack, and how it's organized.
Built with FastAPI, LangChain, and a Map-Reduce chunking pipeline that keeps
things within context limits even for large repos.

## Setup
You'll need Python 3.12+ and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: pip install uv

# Clone and enter the project
git clone https://github.com/psxz/git_summarize.git
cd git_summarize

# Create .env from the template
cp .env.example .env
# Edit .env — you need at minimum:
#   GITHUB_TOKEN   - a GitHub personal access token (optional for public repos,
#                    but required in practice — unauthenticated API is capped at
#                    60 requests/hour, which a single large repo can exhaust)
#   NEBIUS_API_KEY - from https://studio.nebius.com/settings/api-keys

# Install everything and start the server
uv sync
uv run uvicorn app.main:app --reload
```
Server runs at http://localhost:8000. Swagger docs at http://localhost:8000/docs.

### Try it
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```
The field name `github_url` or `repo_url` both work (Pydantic alias).

## Model choice
We went with **MiniMax-M2.1** on Nebius AI Studio for the free credits. It's a reasoning model with a 200K context window and strong code comprehension — good enough for this use case at a fraction of the cost of GPT-4o or Claude. Since Nebius exposes an OpenAI-compatible API, LangChain's `ChatOpenAI` works with just a different `base_url`, so switching providers is a one-line env var change.

## How repository content is handled

### What gets fetched
The service grabs the README first — that's usually the most informative file.
Then it scans the root directory for supplementary files, prioritizing:
- **Documentation** — `.md`, `.rst`, `.txt`, plus `ARCHITECTURE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
- **Project manifests** — `pyproject.toml`, `package.json`, `Cargo.toml`, `Dockerfile`, `docker-compose.yml`, `go.mod`
- **Source code** — `.py`, `.js`, `.ts`, `.go`, `.rs`, etc. (budget permitting)

Fetching happens in parallel (asyncio tasks) to keep latency reasonable.

### What gets skipped
- Binary files (detected via null-byte heuristic in the first 100 bytes)
- Files over 100KB — diminishing returns for context budget
- Lock files, `node_modules/`, vendored directories — auto-generated noise
- Images, compiled assets, `.git/` metadata

### Staying within context limits
This was the interesting part to get right. The pipeline:
1. **Token counting** with tiktoken before any LLM call
2. **Markdown-aware chunking** — recursive splitter that respects heading boundaries
3. For code files, **ChunkHound's cAST algorithm** — tree-sitter based, respects function/class boundaries
4. **Hard budget cap** — `MAX_TOTAL_TOKENS` (120K default) prevents runaway costs
5. **File count limit** — max 30 files by default, docs prioritized over code

The Map-Reduce strategy summarizes each chunk independently (parallelized),
then combines partial summaries in a final "reduce" pass. There's also an
Iterative Refinement strategy (`SUMMARIZATION_STRATEGY=refine`) that builds a
running summary chunk-by-chunk — trades parallelism for coherence.

## Environment variables
| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Recommended | GitHub PAT — technically optional for public repos, but unauthenticated API caps at 60 req/hr (one large repo can exhaust this) |
| `NEBIUS_API_KEY` | Yes* | Nebius AI Studio key (*or set another provider's key) |
| `OPENAI_API_KEY` | No | If using `DEFAULT_LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | No | If using `DEFAULT_LLM_PROVIDER=anthropic` |
| `GOOGLE_API_KEY` | No | If using `DEFAULT_LLM_PROVIDER=google` |
| `DEFAULT_LLM_PROVIDER` | No | `nebius` (default), `openai`, `anthropic`, `google` |
| `DEFAULT_LLM_MODEL` | No | Defaults to `MiniMaxAI/MiniMax-M2.1` |
| `REDIS_URL` | No | For response caching (falls back to in-memory) |
| `API_SECRET_KEY` | No | Bearer token for API auth (unset = open access) |

All configuration via pydantic-settings — no hardcoded keys.

## API endpoints
| Method | Path | Description |
|---|---|---|
| POST | `/summarize` | Main endpoint (accepts `github_url` or `repo_url`) |
| POST | `/api/v1/summarize/stream` | SSE streaming variant |
| GET | `/health` | Health check |
| GET | `/api/v1/rate-limits` | GitHub API rate limit status |

### Example response
```json
{
  "repo_url": "https://github.com/psf/requests",
  "metadata": {
    "owner": "psf",
    "name": "requests",
    "stars": 52000,
    "language": "Python"
  },
  "summary": "# Requests Technical Summary\n\n...",
  "key_technologies": ["Python", "urllib3", "certifi", "pytest"],
  "summarization_meta": {
    "strategy_used": "map_reduce",
    "llm_provider": "nebius",
    "llm_model": "MiniMaxAI/MiniMax-M2.1",
    "cache_hit": false
  }
}
```

## Error handling
- **Invalid URL** → 422 with Pydantic validation details
- **Private/missing repo** → 404 propagated from GitHub API
- **Rate limited** → retries with exponential backoff, then 503
- **Missing API key** → 500 with descriptive message
- **Empty repo** → 200 with warning, summary based on metadata alone

## Authentication
Three modes, auto-selected based on which env vars are set:
1. **Open mode** (default for eval) — nothing set, all requests accepted
2. **Static key** — set `API_SECRET_KEY`, send `Authorization: Bearer <key>`
3. **Logto JWT** — set `LOGTO_ENDPOINT` for production OIDC validation

Logto support was added for production deployments where you want proper JWT auth with per-user GitHub tokens and scope-based access control. For evaluation, just leave both `API_SECRET_KEY` and `LOGTO_ENDPOINT` unset.

## Project structure
```
app/
├── main.py                    # App factory, middleware, lifespan
├── core/
│   ├── config.py              # Settings from env vars
│   ├── security.py            # Auth layer (Logto JWT / static key / open)
│   ├── logging.py             # Structured logging (structlog)
│   └── observability.py       # Phoenix OTLP tracing
├── api/v1/endpoints/summarize.py # Route handlers
├── schemas/summarize.py       # Pydantic models
└── services/
    ├── github_client.py       # Async GitHub API client
    ├── summarizer.py          # Map-Reduce / Refine pipeline
    ├── preprocessor.py        # Markdown cleanup, chunking
    ├── llm_factory.py         # Multi-provider LLM
    └── cache.py               # Redis + in-memory fallback
```

## Development
```bash
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pytest tests/ -v     # tests (63 passing)
```

## Deployment
Includes `render.yaml` for Render.com and `.github/workflows/ci.yml` for
GitHub Actions CI (lint → test → docker build).

## License
MIT
