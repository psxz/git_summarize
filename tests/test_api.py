"""
tests/test_api.py
──────────────────
Integration-style tests using pytest + httpx AsyncClient.
Tests cover URL validation, caching, error handling, and endpoint contracts.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.security import AuthInfo, verify_api_key
from app.schemas.summarize import (
    RepoMetadata,
    SummarizationMeta,
    SummarizeResponse,
)
from app.services.preprocessor import (
    chunk_text,
    estimate_tokens,
    sanitize_markdown,
)
from app.services.github_client import parse_repo_url


# ── Fixtures ──────────────────────────────────────────────────────────────────

_MOCK_AUTH = AuthInfo(
    sub="test-user",
    github_token="ghp_test_token",
    scopes=["repo:summarize"],
)

@pytest.fixture
def client():
    """TestClient with auth dependency overridden so tests pass regardless of .env auth config."""
    app.dependency_overrides[verify_api_key] = lambda: _MOCK_AUTH
    yield TestClient(app)
    app.dependency_overrides.clear()


_MOCK_RESPONSE = SummarizeResponse(
    repo_url="https://github.com/tiangolo/fastapi",
    metadata=RepoMetadata(
        owner="tiangolo",
        name="fastapi",
        full_name="tiangolo/fastapi",
        description="FastAPI framework",
        stars=75000,
        forks=6000,
        open_issues=500,
        topics=["python", "api", "fastapi"],
        default_branch="master",
        language="Python",
        languages={"Python": 5_000_000},
    ),
    summary="FastAPI is a modern, high-performance web framework for Python APIs.",
    key_technologies=["Python", "FastAPI", "Pydantic", "Starlette"],
    summarization_meta=SummarizationMeta(
        strategy_used="map_reduce",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        total_input_tokens_estimated=3000,
        chunks_processed=2,
        files_scanned=10,
    ),
)


# ── URL Parsing Tests ─────────────────────────────────────────────────────────

class TestParseRepoUrl:
    def test_https_url(self):
        owner, repo = parse_repo_url("https://github.com/tiangolo/fastapi")
        assert owner == "tiangolo"
        assert repo == "fastapi"

    def test_https_url_with_dot_git(self):
        owner, repo = parse_repo_url("https://github.com/tiangolo/fastapi.git")
        assert owner == "tiangolo"
        assert repo == "fastapi"

    def test_https_url_trailing_slash(self):
        owner, repo = parse_repo_url("https://github.com/tiangolo/fastapi/")
        assert owner == "tiangolo"
        assert repo == "fastapi"

    def test_ssh_url(self):
        owner, repo = parse_repo_url("git@github.com:tiangolo/fastapi.git")
        assert owner == "tiangolo"
        assert repo == "fastapi"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            parse_repo_url("https://gitlab.com/user/repo")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_repo_url("not-a-url")


# ── Preprocessor Tests ────────────────────────────────────────────────────────

class TestSanitizeMarkdown:
    def test_removes_shields_badge(self):
        md = "![Build](https://img.shields.io/badge/build-passing-green)"
        result = sanitize_markdown(md)
        assert "shields.io" not in result

    def test_removes_html_comments(self):
        md = "# Title\n<!-- this is a comment -->\nContent"
        result = sanitize_markdown(md)
        assert "this is a comment" not in result
        assert "Content" in result

    def test_preserves_headings(self):
        md = "# My Project\n\nThis is a great project."
        result = sanitize_markdown(md)
        assert "My Project" in result
        assert "great project" in result

    def test_empty_input(self):
        assert sanitize_markdown("") == ""
        assert sanitize_markdown("   ") == ""

    def test_collapses_blank_lines(self):
        md = "Paragraph 1\n\n\n\n\nParagraph 2"
        result = sanitize_markdown(md)
        assert "\n\n\n" not in result


class TestEstimateTokens:
    def test_non_zero(self):
        assert estimate_tokens("Hello world") > 0

    def test_longer_text_has_more_tokens(self):
        short = estimate_tokens("Hi")
        long_ = estimate_tokens("This is a much longer sentence with many more words.")
        assert long_ > short

    def test_empty_string(self):
        assert estimate_tokens("") >= 0


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short paragraph."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        # Create a long document
        text = "\n\n".join([f"## Section {i}\n" + ("word " * 300) for i in range(10)])
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) > 1

    def test_chunks_not_empty(self):
        text = "## Header\n\nSome content here."
        chunks = chunk_text(text, chunk_size=500)
        assert all(c.strip() for c in chunks)


# ── Request Schema Validation ─────────────────────────────────────────────────

class TestSummarizeRequest:
    def test_valid_github_url(self):
        from app.schemas.summarize import SummarizeRequest
        req = SummarizeRequest(repo_url="https://github.com/tiangolo/fastapi")
        assert req.repo_url == "https://github.com/tiangolo/fastapi"

    def test_invalid_url_raises_validation_error(self):
        from pydantic import ValidationError
        from app.schemas.summarize import SummarizeRequest
        with pytest.raises(ValidationError):
            SummarizeRequest(repo_url="https://evil.com/xss")

    def test_gitlab_url_rejected(self):
        from pydantic import ValidationError
        from app.schemas.summarize import SummarizeRequest
        with pytest.raises(ValidationError):
            SummarizeRequest(repo_url="https://gitlab.com/user/repo")


# ── API Endpoint Tests ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with patch(
            "app.api.v1.endpoints.summarize.GitHubClient.get_rate_limits",
            new_callable=AsyncMock,
            return_value={"rate": {"limit": 5000, "remaining": 4999, "reset": 0}},
        ), patch(
            "app.api.v1.endpoints.summarize.is_cache_connected",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "github_authenticated" in data


class TestSummarizeEndpoint:
    def test_invalid_url_returns_422(self, client):
        response = client.post(
            "/api/v1/summarize",
            json={"repo_url": "not-a-github-url"},
        )
        assert response.status_code == 422

    def test_valid_request_mocked(self, client):
        with patch(
            "app.api.v1.endpoints.summarize.summarize_repo",
            new_callable=AsyncMock,
            return_value=_MOCK_RESPONSE,
        ), patch(
            "app.api.v1.endpoints.summarize.cache_get",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.api.v1.endpoints.summarize.cache_set",
            new_callable=AsyncMock,
        ):
            response = client.post(
                "/api/v1/summarize",
                json={"repo_url": "https://github.com/tiangolo/fastapi"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "metadata" in data
        assert "key_technologies" in data

    def test_cache_hit_returns_fast(self, client):
        cached_data = _MOCK_RESPONSE.model_dump()
        with patch(
            "app.api.v1.endpoints.summarize.cache_get",
            new_callable=AsyncMock,
            return_value=cached_data,
        ):
            response = client.post(
                "/api/v1/summarize",
                json={"repo_url": "https://github.com/tiangolo/fastapi"},
            )
        assert response.status_code == 200
        assert response.json()["summarization_meta"]["cache_hit"] is True


# ── Cache Tests ───────────────────────────────────────────────────────────────

class TestCache:
    @pytest.mark.asyncio
    async def test_memory_cache_roundtrip(self):
        from app.services.cache import cache_get, cache_set, _make_cache_key
        key = _make_cache_key("https://github.com/test/repo", "openai", "gpt-4o", "map_reduce")
        payload = {"test": "data", "value": 42}
        await cache_set(key, payload)
        result = await cache_get(key)
        assert result == payload
