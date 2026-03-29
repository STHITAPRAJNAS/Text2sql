"""
Schema Change Detector
======================
Detects when a Delta Lake table has been modified since it was last indexed
in the vector store and triggers automatic re-indexing.

Detection method:
  1. Query the schema index for the table's `indexed_at` timestamp.
  2. Run DESCRIBE HISTORY on the Delta table and find the latest operation.
  3. If the latest operation timestamp > indexed_at, the table has changed.
  4. Call refresh_table_index to update the vector store entry.

Supported for Databricks (Delta DESCRIBE HISTORY).
For non-Databricks DBs, falls back to a simpler information_schema check.

Usage (called by schema_agent when it encounters a table):
    from core.schema_change_detector import check_and_refresh_if_changed
    refreshed = await check_and_refresh_if_changed("main.sales.orders")
"""
from __future__ import annotations

import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def check_and_refresh_if_changed(table_id: str) -> bool:
    """
    Check if a table has changed since it was indexed and refresh if needed.

    Returns True if the table was refreshed, False if up-to-date or not indexed.
    """
    from core.schema_index import get_schema_index

    index = get_schema_index()
    stored = index.get_by_id(table_id)
    if not stored:
        # Not indexed yet — nothing to refresh here; index_table_if_new handles first index
        return False

    indexed_at = float(stored.get("indexed_at", 0))
    if not indexed_at:
        return False

    # Check for changes
    changed = await _has_table_changed(table_id, indexed_at)
    if not changed:
        return False

    logger.info("Schema change detected — refreshing index", table=table_id)
    try:
        from agents.tools.indexing_tools import refresh_table_index
        # refresh_table_index is a sync tool — run directly
        result = refresh_table_index(table_id=table_id)
        refreshed = result.get("status") == "refreshed"
        if refreshed:
            logger.info("Table re-indexed after schema change", table=table_id)
        return refreshed
    except Exception as exc:
        logger.warning("Failed to refresh changed table", table=table_id, error=str(exc))
        return False


async def _has_table_changed(table_id: str, indexed_at: float) -> bool:
    """
    Return True if the table was modified after `indexed_at` epoch seconds.
    Tries Databricks DESCRIBE HISTORY first, then information_schema.
    """
    # Try Databricks DESCRIBE HISTORY (most accurate for Delta tables)
    try:
        from core.databricks import get_databricks_connector, UCTableRef
        connector = get_databricks_connector()
        if connector is not None:
            ref = UCTableRef.parse(table_id)
            return _check_delta_history(connector, ref, indexed_at)
    except Exception:
        pass

    # Fallback: information_schema last_altered (PostgreSQL/Snowflake/BigQuery)
    try:
        from core.database import get_db_manager
        db = get_db_manager()
        if db is not None:
            return await _check_information_schema(db, table_id, indexed_at)
    except Exception:
        pass

    # Can't determine — assume not changed to avoid over-indexing
    return False


def _check_delta_history(connector: Any, ref: Any, indexed_at: float) -> bool:
    """Check Delta DESCRIBE HISTORY for changes after indexed_at."""
    try:
        result = connector.execute_query(
            f"DESCRIBE HISTORY {ref} LIMIT 1"
        )
        if not result.rows:
            return False

        last_op = result.rows[0]
        # timestamp field is a datetime object or ISO string
        ts = last_op.get("timestamp")
        if ts is None:
            return False

        # Convert to epoch seconds
        import datetime
        if isinstance(ts, (int, float)):
            op_epoch = float(ts)
        elif isinstance(ts, datetime.datetime):
            op_epoch = ts.timestamp()
        else:
            # Parse ISO string
            ts_str = str(ts).replace("Z", "+00:00")
            try:
                dt = datetime.datetime.fromisoformat(ts_str)
                op_epoch = dt.timestamp()
            except ValueError:
                return False

        return op_epoch > indexed_at

    except Exception as exc:
        logger.debug("DESCRIBE HISTORY failed", table=str(ref), error=str(exc))
        return False


async def _check_information_schema(db: Any, table_id: str, indexed_at: float) -> bool:
    """Fallback: check information_schema.tables last_altered."""
    try:
        parts = table_id.split(".")
        table_name = parts[-1]
        schema_name = parts[-2] if len(parts) >= 2 else "public"
        sql = f"""
        SELECT last_altered
        FROM information_schema.tables
        WHERE table_name = '{table_name}'
          AND table_schema = '{schema_name}'
        LIMIT 1
        """
        rows = await db.execute_safe_query(sql)
        if not rows:
            return False
        ts = rows[0].get("last_altered")
        if not ts:
            return False
        import datetime
        if isinstance(ts, datetime.datetime):
            return ts.timestamp() > indexed_at
    except Exception:
        pass
    return False


def get_stale_tables(max_age_seconds: int = 3600) -> list[str]:
    """
    Return table IDs that haven't been re-indexed in max_age_seconds.
    Useful for a background refresh job.
    """
    from core.schema_index import get_schema_index
    index = get_schema_index()
    now = time.time()
    stale = []
    for table_meta in index.list_indexed_tables(limit=10000):
        table_id = table_meta.get("id", "")
        stored = index.get_by_id(table_id)
        if not stored:
            continue
        indexed_at = float(stored.get("indexed_at", 0))
        if indexed_at and (now - indexed_at) > max_age_seconds:
            stale.append(table_id)
    return stale
