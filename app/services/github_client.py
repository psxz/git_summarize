"""
Async GitHub REST API client.

Handles authenticated requests with a PAT, exponential backoff on
rate limits (tenacity), and fetching repo metadata, READMEs, file
contents, and directory listings. Pinned to API version 2022-11-28.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"
_RATE_LIMIT_CODES = {403, 429}


def _is_rate_limited(exc: BaseException) -> bool:
    """tenacity predicate: retry only on rate-limit HTTP errors."""
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RATE_LIMIT_CODES


def _build_headers(token: str, api_version: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": api_version,
        "User-Agent": "github-summarizer-api/1.0",
    }


def parse_repo_url(url: str) -> tuple[str, str]:
    """
    Extract (owner, repo) from HTTPS or SSH GitHub URLs.
    Raises ValueError for unrecognised formats.
    """
    url = url.strip().rstrip("/").removesuffix(".git")

    # SSH: git@github.com:owner/repo
    ssh_match = re.match(r"^git@github\.com:([^/]+)/(.+)$", url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

    # HTTPS: https://github.com/owner/repo
    parsed = urlparse(url)
    if parsed.hostname in ("github.com", "www.github.com"):
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]

    raise ValueError(f"Cannot parse GitHub owner/repo from URL: {url!r}")


class GitHubClient:
    """
    Async wrapper around the GitHub REST API.

    Pass ``github_token`` to use a per-user token (from Logto AuthInfo).
    Falls back to the service-level GITHUB_TOKEN env var when omitted.
    Create one instance per request.
    """

    def __init__(
        self,
        github_token: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        settings = get_settings()
        self._settings = settings
        token = github_token or settings.github_token
        self._headers = _build_headers(token, settings.github_api_version)
        self._client = client  # allow injection for testing

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _get_with_retry(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET with tenacity exponential backoff on rate-limit errors."""
        for attempt in range(1, 6):  # max 5 attempts
            async with httpx.AsyncClient(headers=self._headers, timeout=30) as c:
                response = await c.get(url, **kwargs)

            if response.status_code not in _RATE_LIMIT_CODES:
                if response.status_code == 404:
                    return response
                response.raise_for_status()
                return response

            retry_after = int(response.headers.get("Retry-After", str(2**attempt)))
            logger.warning(
                "github_rate_limited",
                attempt=attempt,
                status=response.status_code,
                retry_after=retry_after,
                url=url,
            )
            import asyncio

            await asyncio.sleep(min(retry_after, 60))

        raise httpx.HTTPStatusError(
            "Exceeded retry limit due to GitHub rate limiting.",
            request=response.request,  # type: ignore
            response=response,  # type: ignore
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def get_repo_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        """GET /repos/{owner}/{repo} — primary metadata."""
        url = f"{GITHUB_API}/repos/{owner}/{repo}"
        response = await self._get_with_retry(url)
        if response.status_code == 404:
            raise ValueError(f"Repository '{owner}/{repo}' not found or is private.")
        return response.json()

    async def get_readme(self, owner: str, repo: str) -> str | None:
        """
        GET /repos/{owner}/{repo}/readme — raw README content.
        Returns None if the repo has no README.
        """
        url = f"{GITHUB_API}/repos/{owner}/{repo}/readme"
        # Request raw content directly
        headers = {**self._headers, "Accept": "application/vnd.github.raw+json"}
        async with httpx.AsyncClient(headers=headers, timeout=30) as c:
            response = await c.get(url)

        if response.status_code == 404:
            logger.info("readme_not_found", owner=owner, repo=repo)
            return None

        response.raise_for_status()
        return response.text

    async def get_languages(self, owner: str, repo: str) -> dict[str, int]:
        """GET /repos/{owner}/{repo}/languages — bytes by language."""
        url = f"{GITHUB_API}/repos/{owner}/{repo}/languages"
        response = await self._get_with_retry(url)
        return response.json()

    async def get_directory_contents(
        self, owner: str, repo: str, path: str = ""
    ) -> list[dict[str, Any]]:
        """
        GET /repos/{owner}/{repo}/contents/{path} — file listing.
        Returns empty list on 404 or oversized directories.
        """
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
        response = await self._get_with_retry(url)
        if response.status_code == 404:
            return []
        data = response.json()
        return data if isinstance(data, list) else []

    async def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """
        Fetch and decode a file's text content.
        Skips files > MAX_FILE_SIZE_BYTES or binary files.
        """
        settings = get_settings()
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
        # Use JSON (not raw) to check size before downloading
        json_headers = {**self._headers, "Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(headers=json_headers, timeout=30) as c:
            response = await c.get(url)

        if response.status_code == 404:
            return None
        response.raise_for_status()

        data = response.json()
        size = data.get("size", 0)
        if size > settings.max_file_size_bytes:
            logger.info(
                "file_skipped_too_large",
                path=path,
                size=size,
                limit=settings.max_file_size_bytes,
            )
            return None

        encoding = data.get("encoding")
        content_b64 = data.get("content", "")

        if encoding == "base64":
            try:
                raw_bytes = base64.b64decode(content_b64)
                return raw_bytes.decode("utf-8", errors="replace")
            except Exception:
                return None
        return None

    async def get_rate_limits(self) -> dict[str, Any]:
        """GET /rate_limit — current rate-limit status."""
        url = f"{GITHUB_API}/rate_limit"
        response = await self._get_with_retry(url)
        return response.json()

    async def get_commit_activity(self, owner: str, repo: str) -> list[dict[str, Any]]:
        """GET /repos/{owner}/{repo}/stats/commit_activity — weekly commits."""
        url = f"{GITHUB_API}/repos/{owner}/{repo}/stats/commit_activity"
        response = await self._get_with_retry(url)
        if response.status_code in (202, 204):
            return []  # GitHub is computing stats — return empty
        return response.json() or []
