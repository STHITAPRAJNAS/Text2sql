"""
Clarification Memory
=====================
When agents ask a clarifying question and the user provides an answer,
that (question, answer) pair is stored keyed by a fingerprint of the
ambiguous query.  On future similar queries, the agent checks here first
before generating a clarification request — potentially auto-applying
the known answer and skipping the round-trip entirely.

This directly reduces user friction for recurring ambiguous patterns such as:
  "Did you mean fiscal year or calendar year?"
  "Should I include cancelled orders?"
  "Do you want gross or net revenue?"

Storage: SQLite (same pattern as other core stores)
All writes fire-and-forget; reads synchronous.
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

_DB_PATH = Path("./data/clarification_memory.db")
_initialized = False


async def _init_db() -> None:
    global _initialized
    if _initialized:
        return

    import aiosqlite

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS clarification_memory (
                query_fingerprint TEXT NOT NULL,
                tenant_id         TEXT NOT NULL DEFAULT 'default',
                nl_query          TEXT NOT NULL,
                question          TEXT NOT NULL,
                answer            TEXT NOT NULL,
                use_count         INTEGER DEFAULT 0,
                created_at        INTEGER NOT NULL,
                last_used         INTEGER NOT NULL,
                PRIMARY KEY (query_fingerprint, tenant_id)
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_clarif_tenant "
            "ON clarification_memory(tenant_id)"
        )
        await db.commit()

    _initialized = True


def _fingerprint(text: str) -> str:
    """12-char fingerprint of normalised query (strips numbers, literals)."""
    normalised = re.sub(r"[\d'\"]+", " ", text.lower().strip())
    normalised = re.sub(r"\s+", " ", normalised)
    return hashlib.md5(normalised.encode()).hexdigest()[:12]


def _keyword_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


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

def store_clarification(
    nl_query: str,
    question: str,
    answer: str,
    tenant_id: str = "default",
) -> None:
    """
    Store a (question, answer) pair for a query fingerprint.  Fire-and-forget.

    Args:
        nl_query:  The ambiguous natural language query
        question:  The clarifying question the agent asked
        answer:    The user's answer
        tenant_id: Tenant scope for multi-tenant isolation
    """
    fp = _fingerprint(nl_query)

    async def _write() -> None:
        await _init_db()
        import aiosqlite

        now = int(time.time())
        async with aiosqlite.connect(_DB_PATH) as db:
            await db.execute("""
                INSERT INTO clarification_memory
                    (query_fingerprint, tenant_id, nl_query, question, answer,
                     use_count, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(query_fingerprint, tenant_id) DO UPDATE SET
                    question  = excluded.question,
                    answer    = excluded.answer,
                    last_used = excluded.last_used
            """, (fp, tenant_id, nl_query[:500], question[:500], answer[:1000], now, now))
            await db.commit()

        logger.info(
            "Clarification stored",
            fingerprint=fp,
            tenant_id=tenant_id,
        )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_write())
        else:
            loop.run_until_complete(_write())
    except RuntimeError:
        asyncio.run(_write())


def find_known_clarification(
    nl_query: str,
    tenant_id: str = "default",
    similarity_threshold: float = 0.5,
) -> dict[str, Any] | None:
    """
    Check whether we already know how to handle this ambiguous query.

    Returns None if no sufficiently similar past clarification exists.

    Returns:
        {
            "question": str,    # The clarifying question previously asked
            "answer": str,      # The user's answer
            "similarity": float,
            "nl_query": str,    # The original stored query
            "use_count": int,
        }
        or None if nothing found.
    """
    fp = _fingerprint(nl_query)

    async def _query() -> dict | None:
        await _init_db()
        import aiosqlite

        # Exact fingerprint match first
        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT nl_query, question, answer, use_count
                FROM clarification_memory
                WHERE query_fingerprint = ?
                  AND tenant_id IN (?, 'default')
                ORDER BY tenant_id DESC
                LIMIT 1
            """, (fp, tenant_id)) as cursor:
                row = await cursor.fetchone()

            if row:
                # Bump use_count async (fire-and-forget within the existing async context)
                await db.execute("""
                    UPDATE clarification_memory
                    SET use_count = use_count + 1, last_used = ?
                    WHERE query_fingerprint = ? AND tenant_id = ?
                """, (int(time.time()), fp, tenant_id))
                await db.commit()
                return {
                    "question": row["question"],
                    "answer": row["answer"],
                    "similarity": 1.0,
                    "nl_query": row["nl_query"],
                    "use_count": row["use_count"] + 1,
                }

            # Fallback: keyword similarity search over recent entries
            async with db.execute("""
                SELECT nl_query, question, answer, use_count
                FROM clarification_memory
                WHERE tenant_id IN (?, 'default')
                ORDER BY last_used DESC
                LIMIT 100
            """, (tenant_id,)) as cursor:
                rows = await cursor.fetchall()

        best_score = 0.0
        best_row = None
        for r in rows:
            score = _keyword_overlap(nl_query, r["nl_query"])
            if score > best_score:
                best_score = score
                best_row = r

        if best_row and best_score >= similarity_threshold:
            return {
                "question": best_row["question"],
                "answer": best_row["answer"],
                "similarity": round(best_score, 3),
                "nl_query": best_row["nl_query"],
                "use_count": best_row["use_count"],
            }

        return None

    try:
        return _run_async(_query())
    except Exception as exc:
        logger.warning("find_known_clarification failed", error=str(exc))
        return None


def get_clarification_stats() -> dict[str, Any]:
    """Return stats about the clarification memory store."""
    async def _stats() -> dict:
        await _init_db()
        import aiosqlite

        async with aiosqlite.connect(_DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM clarification_memory") as c:
                total = (await c.fetchone())[0]
            async with db.execute(
                "SELECT SUM(use_count) FROM clarification_memory"
            ) as c:
                total_uses = (await c.fetchone())[0] or 0

        return {
            "stored_clarifications": total,
            "total_auto_applied": total_uses,
        }

    try:
        return _run_async(_stats())
    except Exception:
        return {"stored_clarifications": 0, "total_auto_applied": 0}
