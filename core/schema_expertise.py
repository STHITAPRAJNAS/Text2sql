"""
Schema Expertise Graph
=======================
Tracks which table join paths reliably produce high-rated results,
per skill category.  When the SQL Generator encounters a query with
a known skill type, it can look up "proven" join paths that have
historically worked well for that skill — surfacing them as soft hints
before generation.

Data model:
  join_path  — e.g. "orders→order_items→products"
  skill_tag  — taxonomy tag from skill_classifier
  success_count, total_count — usage statistics
  avg_rating  — running average of user ratings (0-5)

All writes are fire-and-forget.  Reads are synchronous (SQLite, fast).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DB_PATH = Path("./data/schema_expertise.db")
_initialized = False


async def _init_db() -> None:
    global _initialized
    if _initialized:
        return

    import aiosqlite

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schema_expertise (
                join_path     TEXT NOT NULL,
                skill_tag     TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                total_count   INTEGER DEFAULT 0,
                rating_sum    REAL    DEFAULT 0.0,
                avg_rating    REAL    DEFAULT 0.0,
                last_used     INTEGER DEFAULT 0,
                PRIMARY KEY (join_path, skill_tag)
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_expertise_skill ON schema_expertise(skill_tag)"
        )
        await db.commit()

    _initialized = True


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result(timeout=5)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Public API ─────────────────────────────────────────────────────────────────

def record_join_result(
    join_paths: list[str],
    skill_tags: list[str],
    rating: float,
) -> None:
    """
    Record that a set of join paths was used for a skill-tagged query,
    with the resulting user rating.  Fire-and-forget.

    Args:
        join_paths: List of "tableA→tableB→tableC" strings
        skill_tags: Taxonomy tags for the query
        rating:     User rating (0.0 = no feedback, 1-5 = user rating)
    """
    if not join_paths or not skill_tags:
        return

    is_success = rating >= 4.0

    async def _write() -> None:
        await _init_db()
        import aiosqlite

        async with aiosqlite.connect(_DB_PATH) as db:
            for path in join_paths:
                for skill in skill_tags:
                    await db.execute("""
                        INSERT INTO schema_expertise
                            (join_path, skill_tag, success_count, total_count,
                             rating_sum, avg_rating, last_used)
                        VALUES (?, ?, ?, 1, ?, ?, ?)
                        ON CONFLICT(join_path, skill_tag) DO UPDATE SET
                            success_count = success_count + ?,
                            total_count   = total_count + 1,
                            rating_sum    = rating_sum + ?,
                            avg_rating    = (rating_sum + ?) / (total_count + 1),
                            last_used     = ?
                    """, (
                        path, skill,
                        1 if is_success else 0,
                        rating, rating,
                        int(time.time()),
                        # ON CONFLICT values:
                        1 if is_success else 0,
                        rating, rating,
                        int(time.time()),
                    ))
            await db.commit()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_write())
        else:
            loop.run_until_complete(_write())
    except RuntimeError:
        asyncio.run(_write())


def get_proven_joins(
    skill_tags: list[str],
    min_successes: int = 3,
    min_avg_rating: float = 4.0,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Return join paths that have historically worked well for the given skills.

    Args:
        skill_tags:     Taxonomy tags for the current query
        min_successes:  Minimum number of successful uses required
        min_avg_rating: Minimum average user rating
        top_k:          Maximum paths to return

    Returns:
        List of dicts: {join_path, skill_tag, success_count, avg_rating}
    """
    if not skill_tags:
        return []

    async def _query() -> list[dict]:
        await _init_db()
        import aiosqlite

        placeholders = ",".join("?" for _ in skill_tags)
        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"""
                SELECT join_path, skill_tag, success_count, total_count, avg_rating
                FROM schema_expertise
                WHERE skill_tag IN ({placeholders})
                  AND success_count >= ?
                  AND avg_rating >= ?
                ORDER BY avg_rating DESC, success_count DESC
                LIMIT ?
            """, (*skill_tags, min_successes, min_avg_rating, top_k)) as cursor:
                rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    try:
        return _run_async(_query())
    except Exception as exc:
        logger.warning("get_proven_joins failed", error=str(exc))
        return []


def get_expertise_report(top_k: int = 10) -> dict[str, Any]:
    """Return the top join paths by average rating for the metrics endpoint."""
    async def _query() -> dict:
        await _init_db()
        import aiosqlite

        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT join_path, skill_tag, success_count, total_count, avg_rating
                FROM schema_expertise
                ORDER BY avg_rating DESC, success_count DESC
                LIMIT ?
            """, (top_k,)) as cursor:
                rows = await cursor.fetchall()

            async with db.execute(
                "SELECT COUNT(*) FROM schema_expertise"
            ) as c:
                total = (await c.fetchone())[0]

        return {
            "total_paths": total,
            "top_paths": [dict(r) for r in rows],
        }

    try:
        return _run_async(_query())
    except Exception:
        return {"total_paths": 0, "top_paths": []}
