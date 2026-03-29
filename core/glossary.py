"""
Business Glossary Store
=======================
Maps enterprise business terms to exact database table/column/filter
references. Stored as a table in the same session database so it persists
across restarts without extra infrastructure.

Table: glossary_terms
  term          TEXT PRIMARY KEY
  table_name    TEXT
  column_name   TEXT
  filter_sql    TEXT    -- e.g. "status = 'active'"
  description   TEXT
  example_sql   TEXT    -- example usage
  created_at    REAL
  updated_at    REAL

The in-memory dict cache makes lookups O(1) for the common path (agent tool
call during Phase R). Writes invalidate the cache entry.

Agents use `lookup_glossary(term)` to resolve business terms before schema
linking. Example resolutions:
  "active customers" → users WHERE status = 'active'
  "revenue"          → SUM(order_items.unit_price * quantity)
  "churned users"    → users WHERE last_login < NOW() - INTERVAL 90 DAYS
  "YTD"             → WHERE YEAR(created_at) = YEAR(CURRENT_DATE)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# In-memory cache: {term_lower: GlossaryEntry}
_cache: dict[str, dict[str, Any]] = {}
_db_initialized = False
_db_lock = asyncio.Lock()


# ------------------------------------------------------------------ #
# DB initialization                                                    #
# ------------------------------------------------------------------ #

async def _get_db_path() -> str:
    from config.settings import get_settings
    settings = get_settings()
    url = settings.session.session_db_url
    # Extract file path from sqlite URL
    if "sqlite" in url:
        path = url.split("///")[-1]
        return path
    return "./data/glossary.db"


async def _init_db() -> None:
    global _db_initialized
    if _db_initialized:
        return
    async with _db_lock:
        if _db_initialized:
            return
        try:
            import aiosqlite
            db_path = await _get_db_path()
            import pathlib
            pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS glossary_terms (
                        term        TEXT PRIMARY KEY COLLATE NOCASE,
                        table_name  TEXT,
                        column_name TEXT,
                        filter_sql  TEXT,
                        description TEXT,
                        example_sql TEXT,
                        created_at  REAL,
                        updated_at  REAL
                    )
                """)
                await db.commit()
            _db_initialized = True
            logger.info("Glossary DB initialized")
        except Exception as exc:
            logger.warning("Glossary DB init failed", error=str(exc))


async def _load_cache() -> None:
    """Load all glossary terms into the in-memory cache."""
    global _cache
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM glossary_terms") as cur:
                rows = await cur.fetchall()
        _cache = {row["term"].lower(): dict(row) for row in rows}
        logger.debug("Glossary cache loaded", entries=len(_cache))
    except Exception as exc:
        logger.warning("Glossary cache load failed", error=str(exc))


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

async def lookup(term: str) -> dict[str, Any] | None:
    """
    Look up a business term. Returns the glossary entry or None.
    Fast O(1) from in-memory cache after first load.
    """
    await _init_db()
    if not _cache:
        await _load_cache()
    return _cache.get(term.lower().strip())


async def upsert(
    term: str,
    table_name: str = "",
    column_name: str = "",
    filter_sql: str = "",
    description: str = "",
    example_sql: str = "",
) -> dict[str, Any]:
    """Insert or update a glossary term."""
    await _init_db()
    now = time.time()
    entry = {
        "term": term,
        "table_name": table_name,
        "column_name": column_name,
        "filter_sql": filter_sql,
        "description": description,
        "example_sql": example_sql,
        "created_at": _cache.get(term.lower(), {}).get("created_at", now),
        "updated_at": now,
    }
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO glossary_terms
                    (term, table_name, column_name, filter_sql, description, example_sql, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    table_name  = excluded.table_name,
                    column_name = excluded.column_name,
                    filter_sql  = excluded.filter_sql,
                    description = excluded.description,
                    example_sql = excluded.example_sql,
                    updated_at  = excluded.updated_at
            """, (
                entry["term"], entry["table_name"], entry["column_name"],
                entry["filter_sql"], entry["description"], entry["example_sql"],
                entry["created_at"], entry["updated_at"],
            ))
            await db.commit()
        _cache[term.lower()] = entry
        logger.info("Glossary term upserted", term=term)
        return {"status": "ok", "term": term}
    except Exception as exc:
        logger.warning("Glossary upsert failed", term=term, error=str(exc))
        return {"status": "error", "error": str(exc)}


async def delete(term: str) -> bool:
    """Delete a glossary term. Returns True if deleted."""
    await _init_db()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("DELETE FROM glossary_terms WHERE term = ? COLLATE NOCASE", (term,))
            await db.commit()
        _cache.pop(term.lower(), None)
        return True
    except Exception:
        return False


async def list_all(limit: int = 200) -> list[dict[str, Any]]:
    """Return all glossary terms."""
    await _init_db()
    if not _cache:
        await _load_cache()
    items = list(_cache.values())
    items.sort(key=lambda x: x.get("term", ""))
    return items[:limit]


async def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Simple keyword search over terms and descriptions."""
    await _init_db()
    if not _cache:
        await _load_cache()
    q = query.lower()
    results = [
        entry for entry in _cache.values()
        if q in entry.get("term", "").lower()
        or q in entry.get("description", "").lower()
    ]
    return results[:limit]


# Sync wrapper for agent tools (tools can't be async in some ADK versions)
def lookup_sync(term: str) -> dict[str, Any] | None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, lookup(term)).result(timeout=3)
        return loop.run_until_complete(lookup(term))
    except Exception:
        return _cache.get(term.lower().strip())
