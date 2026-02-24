"""
Redis-backed response cache with in-memory fallback.

Cache key is a SHA-256 hash of (repo_url + provider + model + strategy).
TTL configurable via CACHE_TTL_SECONDS (default 1h). If REDIS_URL isn't
set, falls back to a simple in-process dict (fine for dev, not for
multi-worker production).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── In-memory fallback (single-process only) ──────────────────────────────────
_MEM_CACHE: dict[str, str] = {}
_MAX_MEM_ENTRIES = 256


def _make_cache_key(
    repo_url: str,
    provider: str,
    model: str,
    strategy: str,
) -> str:
    raw = f"{repo_url}|{provider}|{model}|{strategy}"
    return "ghsum:" + hashlib.sha256(raw.encode()).hexdigest()


async def _get_redis():
    """Return an async Redis client or None if Redis is not configured."""
    settings = get_settings()
    if not settings.redis_url:
        return None
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as e:
        logger.warning("redis_unavailable", error=str(e))
        return None


async def cache_get(key: str) -> Any | None:
    """Return cached value or None."""
    redis = await _get_redis()
    if redis:
        try:
            raw = await redis.get(key)
            if raw:
                logger.info("cache_hit_redis", key=key)
                return json.loads(raw)
        except Exception as e:
            logger.warning("redis_get_error", error=str(e))
        finally:
            await redis.aclose()
    # Fallback
    if key in _MEM_CACHE:
        logger.info("cache_hit_memory", key=key)
        return json.loads(_MEM_CACHE[key])
    return None


async def cache_set(key: str, value: Any) -> None:
    """Persist a value to cache."""
    settings = get_settings()
    serialized = json.dumps(value)

    redis = await _get_redis()
    if redis:
        try:
            await redis.setex(key, settings.cache_ttl_seconds, serialized)
            logger.info("cache_set_redis", key=key, ttl=settings.cache_ttl_seconds)
            return
        except Exception as e:
            logger.warning("redis_set_error", error=str(e))
        finally:
            await redis.aclose()

    # Fallback to memory
    if len(_MEM_CACHE) >= _MAX_MEM_ENTRIES:
        # Evict oldest entry (FIFO via dict insertion order in Python 3.7+)
        oldest_key = next(iter(_MEM_CACHE))
        del _MEM_CACHE[oldest_key]
    _MEM_CACHE[key] = serialized
    logger.info("cache_set_memory", key=key)


async def is_cache_connected() -> bool:
    """Health check: returns True if Redis is reachable."""
    redis = await _get_redis()
    if not redis:
        return False
    try:
        await redis.ping()
        return True
    except Exception:
        return False
    finally:
        await redis.aclose()


def build_request_cache_key(request_data: dict) -> str:
    settings = get_settings()
    return _make_cache_key(
        repo_url=request_data.get("repo_url", ""),
        provider=request_data.get("llm_provider") or settings.default_llm_provider.value,
        model=request_data.get("llm_model") or settings.default_llm_model,
        strategy=request_data.get("strategy") or settings.summarization_strategy.value,
    )
