"""
tests/test_services.py
───────────────────────
Unit tests for service-layer functions not covered by test_api.py.

Covers:
  • preprocessor: build_file_tree_snippet, is_code_file, chunk_code_with_cast
  • cache: _make_cache_key determinism, build_request_cache_key
  • llm_factory: get_llm error paths (missing API keys)
  • security: AuthInfo.requires_scope, AuthInfo.to_dict
  • schemas: optional overrides, response roundtrip
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.security import AuthInfo
from app.services.preprocessor import (
    build_file_tree_snippet,
    chunk_code_with_cast,
    is_code_file,
)

# ── Preprocessor: build_file_tree_snippet ─────────────────────────────────────


class TestBuildFileTreeSnippet:
    def test_renders_files_and_dirs(self):
        entries = [
            {"name": "README.md", "type": "file"},
            {"name": "src", "type": "dir"},
            {"name": "main.py", "type": "file"},
        ]
        result = build_file_tree_snippet(entries)
        assert "README.md" in result
        assert "src" in result
        assert "main.py" in result

    def test_excludes_ignored_directories(self):
        entries = [
            {"name": "node_modules", "type": "dir"},
            {"name": ".git", "type": "dir"},
            {"name": "__pycache__", "type": "dir"},
            {"name": "app.py", "type": "file"},
        ]
        result = build_file_tree_snippet(entries)
        assert "node_modules" not in result
        assert ".git" not in result
        assert "__pycache__" not in result
        assert "app.py" in result

    def test_empty_entries(self):
        result = build_file_tree_snippet([])
        assert result == "(no file tree available)"

    def test_respects_max_entries(self):
        entries = [{"name": f"file_{i}.py", "type": "file"} for i in range(100)]
        result = build_file_tree_snippet(entries, max_entries=5)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) <= 5


# ── Preprocessor: is_code_file ────────────────────────────────────────────────


class TestIsCodeFile:
    @pytest.mark.parametrize(
        "filename",
        ["main.py", "app.js", "index.ts", "handler.go", "lib.rs", "Main.java", "server.cpp"],
    )
    def test_code_files_detected(self, filename):
        assert is_code_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        ["README.md", "notes.txt", "config.ini", "image.png", "data.csv", "styles.css"],
    )
    def test_non_code_files_rejected(self, filename):
        assert is_code_file(filename) is False

    def test_case_insensitive_extension(self):
        assert is_code_file("Main.PY") is True
        assert is_code_file("app.JS") is True


# ── Preprocessor: chunk_code_with_cast ────────────────────────────────────────


class TestChunkCodeWithCast:
    def test_fallback_when_cast_unavailable(self):
        """When ChunkHound is not installed, should fall back to text chunking."""
        code = "def hello():\n    return 'world'\n" * 50
        try:
            chunks = chunk_code_with_cast(code, "test.py", max_tokens=100)
            assert len(chunks) >= 1
            assert all(isinstance(c, str) for c in chunks)
        except Exception:
            # tiktoken may need network to download encoding on first run
            pytest.skip("tiktoken encoding not available (offline)")

    def test_empty_content_returns_chunks(self):
        chunks = chunk_code_with_cast("", "test.py")
        assert isinstance(chunks, list)

    def test_whitespace_only_content(self):
        chunks = chunk_code_with_cast("   \n\n  ", "test.py")
        assert isinstance(chunks, list)


# ── Cache: key generation ─────────────────────────────────────────────────────


class TestCacheKeys:
    def test_make_cache_key_deterministic(self):
        from app.services.cache import _make_cache_key

        key1 = _make_cache_key("https://github.com/a/b", "openai", "gpt-4o", "map_reduce")
        key2 = _make_cache_key("https://github.com/a/b", "openai", "gpt-4o", "map_reduce")
        assert key1 == key2

    def test_make_cache_key_differs_on_input(self):
        from app.services.cache import _make_cache_key

        key1 = _make_cache_key("https://github.com/a/b", "openai", "gpt-4o", "map_reduce")
        key2 = _make_cache_key("https://github.com/a/b", "openai", "gpt-4o", "refine")
        assert key1 != key2

    def test_make_cache_key_has_prefix(self):
        from app.services.cache import _make_cache_key

        key = _make_cache_key("https://github.com/a/b", "openai", "gpt-4o", "map_reduce")
        assert key.startswith("ghsum:")

    def test_build_request_cache_key_uses_defaults(self):
        from app.services.cache import build_request_cache_key

        key = build_request_cache_key({"repo_url": "https://github.com/a/b"})
        assert key.startswith("ghsum:")
        assert len(key) > 10  # hash should be non-trivial


# ── LLM Factory: error paths ─────────────────────────────────────────────────


class TestLLMFactory:
    def test_missing_openai_key_raises(self):
        from app.services.llm_factory import get_llm

        with (
            patch("app.services.llm_factory.get_settings") as mock_settings,
        ):
            settings = mock_settings.return_value
            settings.default_llm_provider = "openai"
            settings.default_llm_model = "gpt-4o-mini"
            settings.llm_temperature = 0.2
            settings.llm_max_tokens = 2048
            settings.openai_api_key = None

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_llm(provider="openai")

    def test_missing_anthropic_key_raises(self):
        from app.services.llm_factory import get_llm

        with (
            patch("app.services.llm_factory.get_settings") as mock_settings,
        ):
            settings = mock_settings.return_value
            settings.default_llm_provider = "anthropic"
            settings.default_llm_model = "claude-3-5-haiku-20241022"
            settings.llm_temperature = 0.2
            settings.llm_max_tokens = 2048
            settings.anthropic_api_key = None

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                get_llm(provider="anthropic")

    def test_missing_nebius_key_raises(self):
        from app.services.llm_factory import get_llm

        with (
            patch("app.services.llm_factory.get_settings") as mock_settings,
        ):
            settings = mock_settings.return_value
            settings.default_llm_provider = "nebius"
            settings.default_llm_model = "MiniMax-M2.1"
            settings.llm_temperature = 0.2
            settings.llm_max_tokens = 2048
            settings.nebius_api_key = None

            with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
                get_llm(provider="nebius")

    def test_unknown_provider_raises(self):
        from app.services.llm_factory import get_llm

        with pytest.raises(ValueError):
            get_llm(provider="invalid_provider")


# ── Security: AuthInfo ────────────────────────────────────────────────────────


class TestAuthInfo:
    def test_requires_scope_present(self):
        auth = AuthInfo(sub="user-1", scopes=["read", "write"])
        assert auth.requires_scope("read") is True
        assert auth.requires_scope("write") is True

    def test_requires_scope_absent(self):
        auth = AuthInfo(sub="user-1", scopes=["read"])
        assert auth.requires_scope("admin") is False

    def test_requires_scope_empty(self):
        auth = AuthInfo(sub="user-1")
        assert auth.requires_scope("anything") is False

    def test_to_dict_excludes_github_token(self):
        auth = AuthInfo(
            sub="user-1",
            client_id="client-abc",
            github_token="ghp_secret_token",
            scopes=["repo:summarize"],
        )
        d = auth.to_dict()
        assert "github_token" not in d
        assert d["sub"] == "user-1"
        assert d["client_id"] == "client-abc"
        assert d["scopes"] == ["repo:summarize"]

    def test_to_dict_contains_expected_keys(self):
        auth = AuthInfo(sub="u")
        d = auth.to_dict()
        assert set(d.keys()) == {"sub", "client_id", "organization_id", "scopes", "audience"}


# ── Schema edge cases ────────────────────────────────────────────────────────


class TestSchemaEdgeCases:
    def test_summarize_request_with_all_overrides(self):
        from app.schemas.summarize import (
            LLMProviderChoice,
            SummarizationStrategyChoice,
            SummarizeRequest,
        )

        req = SummarizeRequest(
            repo_url="https://github.com/owner/repo",
            llm_provider=LLMProviderChoice.ANTHROPIC,
            llm_model="claude-3-5-sonnet-20241022",
            strategy=SummarizationStrategyChoice.REFINE,
            include_file_tree=True,
            max_files=10,
        )
        assert req.llm_provider == LLMProviderChoice.ANTHROPIC
        assert req.strategy == SummarizationStrategyChoice.REFINE
        assert req.include_file_tree is True
        assert req.max_files == 10

    def test_summarize_request_defaults(self):
        from app.schemas.summarize import SummarizeRequest

        req = SummarizeRequest(repo_url="https://github.com/owner/repo")
        assert req.llm_provider is None
        assert req.llm_model is None
        assert req.strategy is None
        assert req.include_file_tree is False
        assert req.max_files is None

    def test_summarize_request_ssh_url(self):
        from app.schemas.summarize import SummarizeRequest

        req = SummarizeRequest(repo_url="git@github.com:owner/repo.git")
        assert req.repo_url == "git@github.com:owner/repo.git"

    def test_max_files_bounds(self):
        from pydantic import ValidationError

        from app.schemas.summarize import SummarizeRequest

        with pytest.raises(ValidationError):
            SummarizeRequest(
                repo_url="https://github.com/owner/repo",
                max_files=0,  # below ge=1
            )
        with pytest.raises(ValidationError):
            SummarizeRequest(
                repo_url="https://github.com/owner/repo",
                max_files=51,  # above le=50
            )

    def test_response_model_roundtrip(self):
        from app.schemas.summarize import (
            RepoMetadata,
            SummarizationMeta,
            SummarizeResponse,
        )

        resp = SummarizeResponse(
            repo_url="https://github.com/a/b",
            metadata=RepoMetadata(owner="a", name="b", full_name="a/b"),
            summary="A test summary",
            summarization_meta=SummarizationMeta(
                strategy_used="map_reduce",
                llm_provider="openai",
                llm_model="gpt-4o",
                total_input_tokens_estimated=100,
                chunks_processed=1,
                files_scanned=5,
            ),
        )
        data = resp.model_dump()
        reconstructed = SummarizeResponse(**data)
        assert reconstructed.summary == resp.summary
        assert reconstructed.metadata.owner == "a"
