"""
tests/conftest.py
──────────────────
Shared fixtures and environment setup for the test suite.

Sets dummy environment variables so tests work in CI (where no .env file exists).
Must be loaded BEFORE any module imports `get_settings()`.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Ensure required env vars are present for all tests."""
    # Only set if not already present (don't override real .env in local dev)
    defaults = {
        "GITHUB_TOKEN": "ghp_test_token_for_ci",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)

    # Clear the lru_cache on get_settings so each test gets fresh settings
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
