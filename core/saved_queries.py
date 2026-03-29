"""
Saved / Named Queries Store
============================
Lets users save frequently-used NL queries with optional parameter templates.
Stored in the session database (no extra infrastructure).

Table: saved_queries
  id           TEXT PRIMARY KEY
  name         TEXT UNIQUE per (user_id, tenant_id)
  nl_query     TEXT     -- the query template, may contain {param} placeholders
  description  TEXT
  user_id      TEXT
  tenant_id    TEXT
  tags         TEXT     -- JSON list
  last_used_at REAL
  use_count    INTEGER
  created_at   REAL

Example:
    await save_query(
        name="top_customers",
        nl_query="Top {n} customers by revenue in {region} this month",
        description="Leaderboard query parameterized by region and count",
        user_id="analyst1",
    )

    entry = await get_saved_query("top_customers", user_id="analyst1")
    populated = populate_params(entry["nl_query"], {"n": "10", "region": "West"})
"""
from __future__ import annotations

import asyncio
import json
import re
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
                    CREATE TABLE IF NOT EXISTS saved_queries (
                        id           TEXT PRIMARY KEY,
                        name         TEXT NOT NULL,
                        nl_query     TEXT NOT NULL,
                        description  TEXT,
                        user_id      TEXT NOT NULL DEFAULT 'default',
                        tenant_id    TEXT NOT NULL DEFAULT 'default',
                        tags         TEXT DEFAULT '[]',
                        last_used_at REAL,
                        use_count    INTEGER DEFAULT 0,
                        created_at   REAL
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_name
                        ON saved_queries(name, user_id, tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_saved_user
                        ON saved_queries(user_id, tenant_id, created_at DESC);
                """)
                await db.commit()
            _initialized = True
            logger.info("Saved queries table initialized")
        except Exception as exc:
            logger.warning("Saved queries init failed", error=str(exc))


async def save_query(
    name: str,
    nl_query: str,
    description: str = "",
    user_id: str = "default",
    tenant_id: str = "default",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create or replace a saved query."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        entry_id = str(uuid.uuid4())
        now = time.time()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO saved_queries
                    (id, name, nl_query, description, user_id, tenant_id, tags, created_at, use_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(name, user_id, tenant_id) DO UPDATE SET
                    nl_query=excluded.nl_query,
                    description=excluded.description,
                    tags=excluded.tags
            """, (
                entry_id, name, nl_query, description, user_id, tenant_id,
                json.dumps(tags or []), now,
            ))
            await db.commit()
        return {"success": True, "id": entry_id, "name": name}
    except Exception as exc:
        logger.warning("Save query failed", error=str(exc))
        return {"success": False, "error": str(exc)}


async def get_saved_query(
    name: str,
    user_id: str = "default",
    tenant_id: str = "default",
) -> dict[str, Any] | None:
    """Retrieve a saved query by name."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM saved_queries WHERE name=? AND user_id=? AND tenant_id=?",
                (name, user_id, tenant_id),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                result = dict(row)
                result["tags"] = json.loads(result.get("tags") or "[]")
                # Bump use_count fire-and-forget
                asyncio.create_task(_bump_use_count(result["id"], db_path))
                return result
    except Exception as exc:
        logger.warning("Get saved query failed", error=str(exc))
        return None


async def _bump_use_count(query_id: str, db_path: str) -> None:
    try:
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE saved_queries SET use_count=use_count+1, last_used_at=? WHERE id=?",
                (time.time(), query_id),
            )
            await db.commit()
    except Exception:
        pass


async def list_saved_queries(
    user_id: str = "default",
    tenant_id: str = "default",
    limit: int = 100,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    """List saved queries for a user/tenant, ordered by use_count desc."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM saved_queries
                   WHERE user_id=? AND tenant_id=?
                   ORDER BY use_count DESC, created_at DESC
                   LIMIT ?""",
                (user_id, tenant_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    r = dict(row)
                    r["tags"] = json.loads(r.get("tags") or "[]")
                    if tag and tag not in r["tags"]:
                        continue
                    results.append(r)
                return results
    except Exception as exc:
        logger.warning("List saved queries failed", error=str(exc))
        return []


async def delete_saved_query(
    name: str,
    user_id: str = "default",
    tenant_id: str = "default",
) -> bool:
    """Delete a saved query. Returns True if deleted."""
    await _init_tables()
    try:
        import aiosqlite
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "DELETE FROM saved_queries WHERE name=? AND user_id=? AND tenant_id=?",
                (name, user_id, tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0
    except Exception as exc:
        logger.warning("Delete saved query failed", error=str(exc))
        return False


def populate_params(query_template: str, params: dict[str, str]) -> str:
    """
    Replace {param} placeholders in a saved query template.

    Example:
        populate_params("Top {n} customers in {region}", {"n": "10", "region": "West"})
        → "Top 10 customers in West"
    """
    result = query_template
    for key, value in params.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def extract_params(query_template: str) -> list[str]:
    """Return the list of parameter names in a template string."""
    return re.findall(r"\{(\w+)\}", query_template)
