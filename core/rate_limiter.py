"""
Rate Limiter — Sliding Window per User/API Key
===============================================
Two-tier limit: per-minute and per-hour. Falls back to in-memory when Redis
is unavailable; production deployments should use Redis for cross-worker state.

In-memory implementation uses a deque of timestamps per key — O(1) amortized.

Usage:
    from core.rate_limiter import check_rate_limit, RateLimitExceeded

    allowed, retry_after = check_rate_limit(user_id="user123")
    if not allowed:
        raise HTTPException(429, detail=f"Rate limit exceeded. Retry after {retry_after}s")
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# In-memory sliding window storage: {key: deque[timestamp]}
_windows_minute: dict[str, deque] = defaultdict(deque)
_windows_hour: dict[str, deque] = defaultdict(deque)
_lock = Lock()

# Redis client (optional)
_redis_client = None
_redis_attempted = False


def _get_redis():
    global _redis_client, _redis_attempted
    if _redis_attempted:
        return _redis_client
    _redis_attempted = True
    try:
        import redis
        from config.settings import get_settings
        url = get_settings().cache.redis_url
        _redis_client = redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        _redis_client.ping()
        logger.info("Rate limiter using Redis", url=url)
    except Exception:
        _redis_client = None
    return _redis_client


def _check_in_memory(key: str, limit_per_min: int, limit_per_hour: int) -> tuple[bool, int]:
    """Sliding window check using in-memory deques. Thread-safe."""
    now = time.time()
    with _lock:
        # Minute window
        dq_min = _windows_minute[key]
        cutoff_min = now - 60
        while dq_min and dq_min[0] < cutoff_min:
            dq_min.popleft()
        if len(dq_min) >= limit_per_min:
            oldest = dq_min[0]
            retry_after = max(1, int(60 - (now - oldest)) + 1)
            return False, retry_after

        # Hour window
        dq_hr = _windows_hour[key]
        cutoff_hr = now - 3600
        while dq_hr and dq_hr[0] < cutoff_hr:
            dq_hr.popleft()
        if len(dq_hr) >= limit_per_hour:
            oldest = dq_hr[0]
            retry_after = max(1, int(3600 - (now - oldest)) + 1)
            return False, retry_after

        # Record this request
        dq_min.append(now)
        dq_hr.append(now)
        return True, 0


def _check_redis(key: str, limit_per_min: int, limit_per_hour: int) -> tuple[bool, int]:
    """Redis sliding window using sorted sets."""
    r = _get_redis()
    if r is None:
        return _check_in_memory(key, limit_per_min, limit_per_hour)

    now = time.time()
    pipe = r.pipeline()
    try:
        # Minute check
        min_key = f"rl:min:{key}"
        hr_key = f"rl:hr:{key}"
        cutoff_min = now - 60
        cutoff_hr = now - 3600

        pipe.zremrangebyscore(min_key, 0, cutoff_min)
        pipe.zcard(min_key)
        pipe.zremrangebyscore(hr_key, 0, cutoff_hr)
        pipe.zcard(hr_key)
        results = pipe.execute()

        count_min = results[1]
        count_hr = results[3]

        if count_min >= limit_per_min:
            return False, 61
        if count_hr >= limit_per_hour:
            return False, 3601

        # Record this request
        pipe2 = r.pipeline()
        pipe2.zadd(min_key, {str(now): now})
        pipe2.expire(min_key, 120)
        pipe2.zadd(hr_key, {str(now): now})
        pipe2.expire(hr_key, 7200)
        pipe2.execute()
        return True, 0
    except Exception as exc:
        logger.debug("Redis rate limit check failed, falling back", error=str(exc))
        return _check_in_memory(key, limit_per_min, limit_per_hour)


def check_rate_limit(
    user_id: str | None = None,
    api_key: str | None = None,
) -> tuple[bool, int]:
    """
    Check if the request is within rate limits.

    Uses the most specific available identifier: api_key > user_id > "anonymous".

    Returns:
        (allowed: bool, retry_after_seconds: int)
        retry_after is 0 when allowed, > 0 when rate limited.
    """
    from config.settings import get_settings
    settings = get_settings()

    if not settings.rate_limit.enabled:
        return True, 0

    key = api_key or user_id or "anonymous"
    limit_min = settings.rate_limit.requests_per_minute
    limit_hr = settings.rate_limit.requests_per_hour

    r = _get_redis()
    if r is not None:
        return _check_redis(key, limit_min, limit_hr)
    return _check_in_memory(key, limit_min, limit_hr)


def get_rate_limit_stats(user_id: str | None = None) -> dict[str, Any]:
    """Return current window counts for a user (for diagnostics)."""
    key = user_id or "anonymous"
    now = time.time()
    with _lock:
        dq_min = _windows_minute.get(key, deque())
        dq_hr = _windows_hour.get(key, deque())
        count_min = sum(1 for t in dq_min if t > now - 60)
        count_hr = sum(1 for t in dq_hr if t > now - 3600)

    from config.settings import get_settings
    settings = get_settings()
    return {
        "user_id": key,
        "requests_last_minute": count_min,
        "requests_last_hour": count_hr,
        "limit_per_minute": settings.rate_limit.requests_per_minute,
        "limit_per_hour": settings.rate_limit.requests_per_hour,
        "rate_limiting_enabled": settings.rate_limit.enabled,
    }
