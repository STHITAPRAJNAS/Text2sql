"""
SQL Correction Store
=====================
Persists (wrong_sql → corrected_sql) triples so agents can learn from
user-supplied corrections and avoid repeating known mistakes.

Every time a user submits a corrected SQL via the feedback endpoint,
the triple is stored here.  Before generating SQL, the SQL Generator
queries this store for similar corrections to use as negative examples
("avoid patterns like X, which previously failed — correct form is Y").

Storage: SQLite (same pattern as audit_log, glossary)
All writes are fire-and-forget (asyncio.create_task) — zero latency impact.

Similarity matching uses simple word-overlap scoring for speed.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DB_PATH = Path("./data/corrections.db")
_initialized = False


async def _init_db() -> None:
    global _initialized
    if _initialized:
        return

    import aiosqlite

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sql_corrections (
                id          TEXT PRIMARY KEY,
                query_hash  TEXT NOT NULL,
                nl_query    TEXT NOT NULL,
                wrong_sql   TEXT NOT NULL,
                correct_sql TEXT NOT NULL,
                skill_tags  TEXT DEFAULT '[]',
                user_id     TEXT DEFAULT 'anonymous',
                tenant_id   TEXT DEFAULT 'default',
                use_count   INTEGER DEFAULT 0,
                created_at  INTEGER NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_corrections_hash ON sql_corrections(query_hash)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_corrections_tenant ON sql_corrections(tenant_id)"
        )
        await db.commit()

    _initialized = True


def _fingerprint(text: str) -> str:
    """12-char hex fingerprint of normalised text."""
    normalised = re.sub(r"[\d'\"]+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()[:12]


def _keyword_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word sets."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Public API ─────────────────────────────────────────────────────────────────

def store_correction(
    nl_query: str,
    wrong_sql: str,
    correct_sql: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
    skill_tags: list[str] | None = None,
) -> None:
    """
    Persist a user-supplied correction.  Fire-and-forget — returns immediately.

    Args:
        nl_query:   Original natural language question
        wrong_sql:  SQL that was generated incorrectly
        correct_sql: SQL the user provided as correction
        user_id:    User identifier for attribution
        tenant_id:  Tenant scope
        skill_tags: Skill taxonomy tags for this query
    """
    import json

    async def _write() -> None:
        await _init_db()
        import aiosqlite

        entry_id = hashlib.md5(
            f"{nl_query}{wrong_sql}{time.time()}".encode()
        ).hexdigest()[:16]

        async with aiosqlite.connect(_DB_PATH) as db:
            await db.execute("""
                INSERT OR IGNORE INTO sql_corrections
                    (id, query_hash, nl_query, wrong_sql, correct_sql,
                     skill_tags, user_id, tenant_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                _fingerprint(nl_query),
                nl_query[:1000],
                wrong_sql[:5000],
                correct_sql[:5000],
                json.dumps(skill_tags or []),
                user_id,
                tenant_id,
                int(time.time()),
            ))
            await db.commit()

        logger.info("Correction stored", user_id=user_id, tenant_id=tenant_id)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_write())
        else:
            loop.run_until_complete(_write())
    except RuntimeError:
        asyncio.run(_write())


def find_similar_corrections(
    nl_query: str,
    tenant_id: str = "default",
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Return the top-k corrections most similar to nl_query.

    Uses word-overlap scoring — fast, synchronous.

    Returns:
        List of dicts: {nl_query, wrong_sql, correct_sql, skill_tags, similarity}
    """
    import json

    async def _query() -> list[dict]:
        await _init_db()
        import aiosqlite

        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT nl_query, wrong_sql, correct_sql, skill_tags
                FROM sql_corrections
                WHERE tenant_id IN (?, 'default')
                ORDER BY created_at DESC
                LIMIT 200
            """, (tenant_id,)) as cursor:
                rows = await cursor.fetchall()

        scored = []
        for row in rows:
            score = _keyword_overlap(nl_query, row["nl_query"])
            if score > 0.1:
                scored.append((score, {
                    "nl_query": row["nl_query"],
                    "wrong_sql": row["wrong_sql"],
                    "correct_sql": row["correct_sql"],
                    "skill_tags": json.loads(row["skill_tags"] or "[]"),
                    "similarity": round(score, 3),
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _query()).result(timeout=5)
        return loop.run_until_complete(_query())
    except Exception as exc:
        logger.warning("find_similar_corrections failed", error=str(exc))
        return []


def get_correction_stats() -> dict[str, Any]:
    """Return aggregate stats about the correction store."""
    async def _stats() -> dict:
        await _init_db()
        import aiosqlite

        async with aiosqlite.connect(_DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM sql_corrections") as c:
                total = (await c.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(DISTINCT tenant_id) FROM sql_corrections"
            ) as c:
                tenants = (await c.fetchone())[0]

        return {"total_corrections": total, "tenants": tenants}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _stats()).result(timeout=5)
        return loop.run_until_complete(_stats())
    except Exception:
        return {"total_corrections": 0, "tenants": 0}
