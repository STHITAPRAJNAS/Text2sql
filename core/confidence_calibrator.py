"""
Adaptive Confidence Calibrator
================================
Tracks the relationship between the pipeline's predicted confidence scores
and actual user satisfaction ratings, per skill category.

Over time this builds a picture of how well-calibrated the pipeline is
for each query type.  If the pipeline consistently over-estimates confidence
on "window_function" queries (predicts 0.9 but users only rate 3/5), the
effective threshold is raised so those queries trigger a retry loop more
aggressively.

Algorithm:
  - Rolling window of last MAX_SAMPLES (predicted, actual_rating) pairs per skill
  - actual_rating is normalised to [0, 1] (divide by 5.0)
  - Effective threshold = base_threshold * (base_accuracy / observed_accuracy)
    where accuracy = % of queries where actual_rating ≥ 4.0 at predicted ≥ threshold
  - Clamped to [base * 0.7, base * 1.2] to avoid extreme drift
  - Falls back to settings threshold when fewer than MIN_SAMPLES observations exist

All writes fire-and-forget; reads are synchronous.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DB_PATH = Path("./data/confidence_calibration.db")
_initialized = False

MAX_SAMPLES = 200   # Per-skill rolling window size
MIN_SAMPLES = 10    # Minimum before calibration kicks in


async def _init_db() -> None:
    global _initialized
    if _initialized:
        return

    import aiosqlite

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_tag    TEXT    NOT NULL,
                predicted    REAL    NOT NULL,
                actual_rating REAL   NOT NULL,
                created_at   INTEGER NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_calib_skill ON confidence_calibration(skill_tag)"
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


def _base_threshold() -> float:
    try:
        from config.settings import get_settings
        return get_settings().deep_think.deep_think_confidence_threshold
    except Exception:
        return 0.85


# ── Public API ─────────────────────────────────────────────────────────────────

def record_calibration(
    skill_tags: list[str],
    predicted_confidence: float,
    actual_rating: float,
) -> None:
    """
    Record an observation for calibration.  Fire-and-forget.

    Args:
        skill_tags:           Taxonomy tags from the query
        predicted_confidence: Confidence score the pipeline reported (0-1)
        actual_rating:        User rating normalised to 0-1 (raw_rating / 5.0)
    """
    if not skill_tags or predicted_confidence <= 0:
        return

    async def _write() -> None:
        await _init_db()
        import aiosqlite

        now = int(time.time())
        async with aiosqlite.connect(_DB_PATH) as db:
            for skill in skill_tags:
                await db.execute("""
                    INSERT INTO confidence_calibration
                        (skill_tag, predicted, actual_rating, created_at)
                    VALUES (?, ?, ?, ?)
                """, (skill, predicted_confidence, actual_rating, now))

                # Trim to rolling window
                await db.execute("""
                    DELETE FROM confidence_calibration
                    WHERE skill_tag = ?
                      AND id NOT IN (
                          SELECT id FROM confidence_calibration
                          WHERE skill_tag = ?
                          ORDER BY created_at DESC
                          LIMIT ?
                      )
                """, (skill, skill, MAX_SAMPLES))

            await db.commit()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_write())
        else:
            loop.run_until_complete(_write())
    except RuntimeError:
        asyncio.run(_write())


def get_calibrated_threshold(skill_tags: list[str]) -> float:
    """
    Return the effective confidence threshold for a skill combination.

    When sufficient observations exist, adjusts the base threshold up or
    down based on observed accuracy.  Returns the base threshold when data
    is insufficient.

    Args:
        skill_tags: Taxonomy tags for the current query

    Returns:
        Effective confidence threshold (0.0 – 1.0)
    """
    if not skill_tags:
        return _base_threshold()

    base = _base_threshold()

    async def _compute() -> float:
        await _init_db()
        import aiosqlite

        # Use the most specific (complex) skill in the list
        skill = skill_tags[0]

        async with aiosqlite.connect(_DB_PATH) as db:
            async with db.execute("""
                SELECT predicted, actual_rating
                FROM confidence_calibration
                WHERE skill_tag = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (skill, MAX_SAMPLES)) as cursor:
                rows = await cursor.fetchall()

        if len(rows) < MIN_SAMPLES:
            return base

        # Accuracy = % of observations where actual was "good" (≥ 0.8 = 4/5)
        # when predicted was at or above base threshold
        qualified = [(p, a) for p, a in rows if p >= base]
        if not qualified:
            return base

        observed_accuracy = sum(1 for _, a in qualified if a >= 0.8) / len(qualified)

        # Adjustment: if accuracy is lower than ideal, raise threshold
        # Clamp to ±20% of base
        ideal_accuracy = 0.80   # We want 80% of queries at threshold to succeed
        if observed_accuracy < 0.01:
            return min(base * 1.2, 0.99)

        adjusted = base * (ideal_accuracy / observed_accuracy)
        return round(max(base * 0.7, min(base * 1.2, adjusted)), 3)

    try:
        return _run_async(_compute())
    except Exception as exc:
        logger.warning("get_calibrated_threshold failed", error=str(exc))
        return base


def get_calibration_report() -> dict[str, Any]:
    """Return per-skill calibration statistics for the metrics endpoint."""
    async def _report() -> dict:
        await _init_db()
        import aiosqlite

        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT
                    skill_tag,
                    COUNT(*) as samples,
                    AVG(predicted) as avg_predicted,
                    AVG(actual_rating) as avg_actual,
                    SUM(CASE WHEN actual_rating >= 0.8 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy
                FROM confidence_calibration
                GROUP BY skill_tag
                ORDER BY samples DESC
            """) as cursor:
                rows = await cursor.fetchall()

        base = _base_threshold()
        skills = []
        for row in rows:
            d = dict(row)
            d["effective_threshold"] = get_calibrated_threshold([d["skill_tag"]])
            d["base_threshold"] = base
            d["avg_predicted"] = round(d["avg_predicted"] or 0, 3)
            d["avg_actual"] = round(d["avg_actual"] or 0, 3)
            d["accuracy"] = round(d["accuracy"] or 0, 3)
            skills.append(d)

        return {"base_threshold": base, "skills": skills}

    try:
        return _run_async(_report())
    except Exception:
        return {"base_threshold": _base_threshold(), "skills": []}
