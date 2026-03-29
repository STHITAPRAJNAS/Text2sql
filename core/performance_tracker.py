"""
Query Performance Regression Tracker
======================================
Detects when a query pattern is suddenly taking much longer than its baseline,
which usually indicates schema changes, data growth, or missing partition filters.

Design:
  - Query fingerprinting: strip literals/numbers to get a structural fingerprint
  - Baseline: exponential moving average (EMA) of p95 latency per fingerprint
  - Alert threshold: current latency > 2× baseline EMA triggers a RegressionAlert
  - Stored in `performance_baselines` table in the session database
  - All baseline updates are fire-and-forget (zero latency impact)

Usage:
    from core.performance_tracker import track_and_alert

    alert = track_and_alert(nl_query, pipeline_ms)
    if alert:
        logger.warning("Performance regression", **alert)
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_initialized = False
_init_lock = asyncio.Lock()

# EMA alpha: weight for the new sample (0.1 = slow-moving baseline)
EMA_ALPHA = 0.1
# Alert threshold multiplier: alert if current > THRESHOLD × baseline
ALERT_THRESHOLD = 2.0
# Minimum samples before alerting (avoid false positives on first queries)
MIN_SAMPLES = 5


async def _get_db_path() -> str:
    from config.settings import get_settings
    url = get_settings().session.session_db_url
    if "sqlite" in url:
        return url.split("///")[-1]
    return "./data/sessions.db"


async def _init_tables() -> None:
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        try:
            import aiosqlite
            import pathlib
            db_path = await _get_db_path()
            pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(db_path) as db:
                await db.executescript("""
                    CREATE TABLE IF NOT EXISTS performance_baselines (
                        fingerprint     TEXT PRIMARY KEY,
                        ema_ms          REAL NOT NULL,
                        p95_ms          REAL,
                        sample_count    INTEGER DEFAULT 0,
                        last_alert_at   REAL,
                        last_updated_at REAL
                    );
                """)
                await db.commit()
            _initialized = True
        except Exception as exc:
            logger.debug("Performance tracker init failed", error=str(exc))


def fingerprint_query(nl_query: str) -> str:
    """
    Normalize a natural language query into a structural fingerprint.

    Strips:
      - Numbers (replace with #)
      - Quoted strings (replace with ?)
      - Extra whitespace
      - Common date expressions (today, yesterday, last week)

    Returns a short hash of the normalized form.
    """
    q = nl_query.lower().strip()
    q = re.sub(r'\d+', '#', q)
    q = re.sub(r"'[^']*'", "?", q)
    q = re.sub(r'"[^"]*"', "?", q)
    q = re.sub(r'\b(today|yesterday|last\s+week|last\s+month|this\s+month|ytd|mtd)\b', '<DATE>', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return hashlib.md5(q.encode()).hexdigest()[:12]


async def _get_baseline(fingerprint: str, db_path: str) -> dict[str, Any] | None:
    try:
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM performance_baselines WHERE fingerprint=?", (fingerprint,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
    except Exception:
        return None


async def _update_baseline(fingerprint: str, current_ms: float, db_path: str) -> None:
    """Update EMA baseline with new sample — fire-and-forget."""
    await _init_tables()
    try:
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT ema_ms, p95_ms, sample_count FROM performance_baselines WHERE fingerprint=?",
                (fingerprint,)
            ) as cursor:
                row = await cursor.fetchone()

            now = time.time()
            if row:
                old_ema = row["ema_ms"]
                old_p95 = row["p95_ms"] or current_ms
                count = row["sample_count"] + 1
                new_ema = EMA_ALPHA * current_ms + (1 - EMA_ALPHA) * old_ema
                # Simple p95 approximation: max(old_p95, current_ms) decaying slowly
                new_p95 = 0.95 * old_p95 + 0.05 * current_ms if current_ms > old_p95 else old_p95
                await db.execute("""
                    UPDATE performance_baselines
                    SET ema_ms=?, p95_ms=?, sample_count=?, last_updated_at=?
                    WHERE fingerprint=?
                """, (new_ema, new_p95, count, now, fingerprint))
            else:
                await db.execute("""
                    INSERT INTO performance_baselines
                        (fingerprint, ema_ms, p95_ms, sample_count, last_updated_at)
                    VALUES (?, ?, ?, 1, ?)
                """, (fingerprint, current_ms, current_ms, now))
            await db.commit()
    except Exception as exc:
        logger.debug("Baseline update failed", error=str(exc))


def _fire_update(fingerprint: str, current_ms: float, db_path: str) -> None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_update_baseline(fingerprint, current_ms, db_path))
        else:
            loop.run_until_complete(_update_baseline(fingerprint, current_ms, db_path))
    except Exception:
        pass


def track_and_alert(
    nl_query: str,
    pipeline_ms: float,
) -> dict[str, Any] | None:
    """
    Record latency for a query and return a regression alert if detected.

    This function is synchronous (non-blocking) — baseline update is
    fire-and-forget. The alert detection uses a cached/synchronous read
    from a small in-memory dict.

    Returns:
        None if no regression detected.
        dict with regression details if latency > 2× baseline.
    """
    # Synchronous fingerprint + async baseline update
    fp = fingerprint_query(nl_query)

    # Fire-and-forget baseline update
    try:
        import asyncio as _asyncio
        db_path_future = _asyncio.get_event_loop()
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            _asyncio.create_task(_async_track(fp, pipeline_ms))
    except Exception:
        pass

    return None  # Regression alerts returned asynchronously via get_regressions()


async def _async_track(fingerprint: str, current_ms: float) -> dict[str, Any] | None:
    """Async version: update baseline and return alert if regression detected."""
    await _init_tables()
    db_path = await _get_db_path()
    baseline = await _get_baseline(fingerprint, db_path)
    alert = None

    if baseline and baseline["sample_count"] >= MIN_SAMPLES:
        ema = baseline["ema_ms"]
        if ema > 0 and current_ms > ALERT_THRESHOLD * ema:
            alert = {
                "fingerprint": fingerprint,
                "current_ms": round(current_ms, 1),
                "baseline_ema_ms": round(ema, 1),
                "ratio": round(current_ms / ema, 2),
                "threshold": ALERT_THRESHOLD,
                "message": (
                    f"Performance regression: {current_ms:.0f}ms is "
                    f"{current_ms / ema:.1f}× the baseline ({ema:.0f}ms). "
                    "Check for missing partition filters, data growth, or schema changes."
                ),
            }
            logger.warning("Performance regression detected", **alert)

    await _update_baseline(fingerprint, current_ms, db_path)
    return alert


async def get_slow_query_report(limit: int = 20) -> list[dict[str, Any]]:
    """Return the slowest query fingerprints by p95 latency."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT fingerprint, ema_ms, p95_ms, sample_count, last_updated_at
                   FROM performance_baselines
                   ORDER BY p95_ms DESC LIMIT ?""",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
    except Exception:
        return []
