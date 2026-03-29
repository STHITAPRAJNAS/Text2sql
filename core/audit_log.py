"""
Query Audit Log + Dead Letter Queue + Confidence Calibration
=============================================================
All stored in the session database as additional tables — no extra
infrastructure needed. All writes are fire-and-forget (asyncio tasks)
so they add zero latency to user responses.

Tables:
  query_audit_log      — every query with timing, SQL, success, cache status
  dead_letter_queries  — failed/low-confidence queries for human review
  confidence_calibration — predicted vs actual accuracy for threshold tuning

Design:
  - Writer: asyncio.create_task(log_query(...)) — non-blocking
  - Reader: query history / review queue API endpoints
  - Calibration: background task that re-computes threshold weekly
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_initialized = False
_init_lock = asyncio.Lock()


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
                    CREATE TABLE IF NOT EXISTS query_audit_log (
                        query_id        TEXT PRIMARY KEY,
                        user_id         TEXT,
                        session_id      TEXT,
                        nl_query        TEXT,
                        generated_sql   TEXT,
                        optimized_sql   TEXT,
                        database_name   TEXT,
                        success         INTEGER,
                        confidence      REAL,
                        execution_time_ms REAL,
                        pipeline_time_ms  REAL,
                        row_count       INTEGER,
                        cache_hit       INTEGER,
                        cache_source    TEXT,
                        pii_detected    INTEGER,
                        cost_warning    TEXT,
                        error           TEXT,
                        needs_clarification INTEGER,
                        created_at      REAL
                    );

                    CREATE INDEX IF NOT EXISTS idx_audit_session
                        ON query_audit_log(session_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_audit_user
                        ON query_audit_log(user_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS dead_letter_queries (
                        id              TEXT PRIMARY KEY,
                        query_id        TEXT,
                        nl_query        TEXT,
                        generated_sql   TEXT,
                        error           TEXT,
                        failure_reason  TEXT,
                        database_name   TEXT,
                        confidence      REAL,
                        reviewed        INTEGER DEFAULT 0,
                        corrected_sql   TEXT,
                        created_at      REAL
                    );

                    CREATE TABLE IF NOT EXISTS confidence_calibration (
                        id              TEXT PRIMARY KEY,
                        query_id        TEXT,
                        predicted_conf  REAL,
                        was_successful  INTEGER,
                        user_rating     REAL,
                        calibration_err REAL,
                        created_at      REAL
                    );
                """)
                await db.commit()
                # Migrate: add new columns if they don't exist yet
                for col_def in [
                    "correlation_id TEXT",
                    "token_input INTEGER",
                    "token_output INTEGER",
                ]:
                    try:
                        await db.execute(
                            f"ALTER TABLE query_audit_log ADD COLUMN {col_def}"
                        )
                        await db.commit()
                    except Exception:
                        pass  # Column already exists
            _initialized = True
            logger.info("Audit log tables initialized")
        except Exception as exc:
            logger.warning("Audit log init failed", error=str(exc))


# ------------------------------------------------------------------ #
# Fire-and-forget helpers                                              #
# ------------------------------------------------------------------ #

def _fire(coro) -> None:
    """Schedule a coroutine as a non-blocking background task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            loop.run_until_complete(coro)
    except Exception:
        pass


async def _write_audit(entry: dict[str, Any]) -> None:
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO query_audit_log
                (query_id, user_id, session_id, nl_query, generated_sql, optimized_sql,
                 database_name, success, confidence, execution_time_ms, pipeline_time_ms,
                 row_count, cache_hit, cache_source, pii_detected, cost_warning, error,
                 needs_clarification, created_at, correlation_id, token_input, token_output)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                entry["query_id"], entry.get("user_id", ""), entry.get("session_id", ""),
                entry.get("nl_query", ""), entry.get("generated_sql", ""), entry.get("optimized_sql", ""),
                entry.get("database_name", "default"), int(entry.get("success", False)),
                entry.get("confidence", 0.0), entry.get("execution_time_ms", 0.0),
                entry.get("pipeline_time_ms", 0.0), entry.get("row_count", 0),
                int(entry.get("cache_hit", False)), entry.get("cache_source"),
                int(entry.get("pii_detected", False)), entry.get("cost_warning"),
                entry.get("error"), int(entry.get("needs_clarification", False)),
                entry.get("created_at", time.time()),
                entry.get("correlation_id"), entry.get("token_input"), entry.get("token_output"),
            ))
            await db.commit()
    except Exception as exc:
        logger.debug("Audit write failed", error=str(exc))


async def _write_dead_letter(entry: dict[str, Any]) -> None:
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO dead_letter_queries
                (id, query_id, nl_query, generated_sql, error, failure_reason,
                 database_name, confidence, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                str(uuid.uuid4()), entry.get("query_id", ""),
                entry.get("nl_query", ""), entry.get("generated_sql", ""),
                entry.get("error", ""), entry.get("failure_reason", "unknown"),
                entry.get("database_name", "default"), entry.get("confidence", 0.0),
                time.time(),
            ))
            await db.commit()
    except Exception as exc:
        logger.debug("Dead letter write failed", error=str(exc))


async def _write_calibration(entry: dict[str, Any]) -> None:
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        predicted = entry.get("predicted_conf", 0.0)
        actual = 1.0 if entry.get("was_successful") else 0.0
        err = abs(predicted - actual)
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO confidence_calibration
                (id, query_id, predicted_conf, was_successful, user_rating, calibration_err, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                str(uuid.uuid4()), entry.get("query_id", ""),
                predicted, int(entry.get("was_successful", False)),
                entry.get("user_rating"), err, time.time(),
            ))
            await db.commit()
    except Exception as exc:
        logger.debug("Calibration write failed", error=str(exc))


# ------------------------------------------------------------------ #
# Public API — all writes are fire-and-forget                          #
# ------------------------------------------------------------------ #

def log_query(
    result: dict[str, Any],
    session_id: str = "",
    user_id: str = "",
    correlation_id: str | None = None,
    token_input: int | None = None,
    token_output: int | None = None,
) -> str:
    """
    Record a query in the audit log. Non-blocking — returns immediately.
    Returns the generated query_id.
    """
    query_id = result.get("query_id") or str(uuid.uuid4())
    entry = {
        **result,
        "query_id": query_id,
        "session_id": session_id,
        "user_id": user_id,
        "correlation_id": correlation_id or result.get("correlation_id"),
        "token_input": token_input or result.get("token_input"),
        "token_output": token_output or result.get("token_output"),
    }
    _fire(_write_audit(entry))

    # Also record calibration data
    _fire(_write_calibration({
        "query_id": query_id,
        "predicted_conf": result.get("confidence", 0.0),
        "was_successful": result.get("success", False),
    }))

    # Track performance regression (fire-and-forget)
    pipeline_ms = result.get("pipeline_time_ms", 0.0)
    if pipeline_ms and result.get("nl_query"):
        try:
            from core.performance_tracker import track_and_alert
            track_and_alert(result["nl_query"], pipeline_ms)
        except Exception:
            pass

    return query_id


def log_dead_letter(
    nl_query: str,
    generated_sql: str | None,
    error: str,
    failure_reason: str,
    database_name: str = "default",
    confidence: float = 0.0,
    query_id: str = "",
) -> None:
    """Record a failed query for human review. Non-blocking."""
    _fire(_write_dead_letter({
        "query_id": query_id or str(uuid.uuid4()),
        "nl_query": nl_query,
        "generated_sql": generated_sql or "",
        "error": error,
        "failure_reason": failure_reason,
        "database_name": database_name,
        "confidence": confidence,
    }))


# ------------------------------------------------------------------ #
# Readers (for API endpoints)                                          #
# ------------------------------------------------------------------ #

async def get_query_history(
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return query history for a user or session."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            if user_id:
                sql = "SELECT * FROM query_audit_log WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params = (user_id, limit, offset)
            elif session_id:
                sql = "SELECT * FROM query_audit_log WHERE session_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params = (session_id, limit, offset)
            else:
                sql = "SELECT * FROM query_audit_log ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params = (limit, offset)
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("History query failed", error=str(exc))
        return []


async def get_dead_letters(
    reviewed: bool | None = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return failed queries for the review queue."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            if reviewed is None:
                sql = "SELECT * FROM dead_letter_queries ORDER BY created_at DESC LIMIT ?"
                params = (limit,)
            else:
                sql = "SELECT * FROM dead_letter_queries WHERE reviewed = ? ORDER BY created_at DESC LIMIT ?"
                params = (int(reviewed), limit)
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Dead letter query failed", error=str(exc))
        return []


async def resolve_dead_letter(dead_letter_id: str, corrected_sql: str) -> bool:
    """Mark a dead letter as reviewed with a corrected SQL."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE dead_letter_queries SET reviewed = 1, corrected_sql = ? WHERE id = ?",
                (corrected_sql, dead_letter_id),
            )
            await db.commit()
        return True
    except Exception:
        return False


async def get_calibration_stats() -> dict[str, Any]:
    """Return confidence calibration statistics."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(predicted_conf) as avg_predicted,
                    AVG(was_successful) as actual_success_rate,
                    AVG(calibration_err) as avg_calibration_err,
                    AVG(CASE WHEN predicted_conf >= 0.85 THEN was_successful ELSE NULL END) as high_conf_success_rate
                FROM confidence_calibration
                WHERE created_at > ?
            """, (time.time() - 30 * 86400,)) as cur:  # last 30 days
                row = await cur.fetchone()
        if row and row[0]:
            return {
                "total_queries": row[0],
                "avg_predicted_confidence": round(float(row[1] or 0), 3),
                "actual_success_rate": round(float(row[2] or 0), 3),
                "avg_calibration_error": round(float(row[3] or 0), 3),
                "high_confidence_success_rate": round(float(row[4] or 0), 3),
                "suggested_threshold": _suggest_threshold(float(row[2] or 0.85)),
            }
    except Exception as exc:
        logger.warning("Calibration stats failed", error=str(exc))
    return {}


def _suggest_threshold(actual_success_rate: float) -> float:
    """Suggest a confidence threshold based on observed success rates."""
    # Simple heuristic: target 90% success rate at threshold
    # If actual success rate at 0.85 is > 90%, we can lower threshold
    if actual_success_rate > 0.92:
        return 0.80
    elif actual_success_rate > 0.88:
        return 0.85
    elif actual_success_rate > 0.80:
        return 0.88
    return 0.90
