"""
Database Tools for PRISM Agents
Google ADK-compatible tool functions for database operations.
These are used by SQL Validator, Optimizer, and Response Formatter agents.
"""
from __future__ import annotations

import json
import time
from typing import Any

import structlog

from core.database import get_db_manager

logger = structlog.get_logger(__name__)


async def execute_sql_query(
    sql: str,
    max_rows: int = 100,
) -> dict[str, Any]:
    """
    Execute a SQL SELECT query and return the results.

    This tool executes the provided SQL query against the configured database.
    Only SELECT queries are allowed for safety. Results are automatically
    limited to prevent memory issues.

    Args:
        sql: The SQL SELECT query to execute
        max_rows: Maximum number of rows to return (default: 100, max: 1000)

    Returns:
        dict with keys:
            - success (bool): Whether execution succeeded
            - columns (list[str]): Column names
            - rows (list[dict]): Query results as list of row dicts
            - row_count (int): Number of rows returned
            - execution_time_ms (float): Query execution time
            - truncated (bool): Whether results were truncated
            - error (str | None): Error message if failed
    """
    try:
        db = await get_db_manager()
        result = await db.execute_safe_query(sql, timeout_ms=30000)

        # Apply additional row limit
        max_rows = min(max_rows, 1000)
        rows = result.rows[:max_rows]
        truncated = result.truncated or len(result.rows) > max_rows

        return {
            "success": True,
            "columns": result.columns,
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": result.execution_time_ms,
            "truncated": truncated,
            "error": None,
        }
    except Exception as e:
        logger.error("Query execution failed", error=str(e))
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "execution_time_ms": 0,
            "truncated": False,
            "error": str(e),
        }


async def get_table_row_count(table_name: str) -> dict[str, Any]:
    """
    Get the approximate row count for a database table.

    Args:
        table_name: Name of the table to count rows for

    Returns:
        dict with keys:
            - table_name (str): The table name
            - row_count (int): Approximate row count
            - error (str | None): Error if any
    """
    try:
        db = await get_db_manager()
        count = await db.get_table_row_count(table_name)
        return {"table_name": table_name, "row_count": count, "error": None}
    except Exception as e:
        return {"table_name": table_name, "row_count": -1, "error": str(e)}


async def get_query_explain(sql: str) -> dict[str, Any]:
    """
    Get the query execution plan (EXPLAIN) for a SQL query.

    Use this to understand query performance characteristics before execution.
    Helps identify missing indexes, sequential scans, or expensive operations.

    Args:
        sql: The SQL query to explain

    Returns:
        dict with keys:
            - plan (str): The query execution plan
            - error (str | None): Error if any
    """
    try:
        db = await get_db_manager()
        plan = await db.explain_query(sql)
        return {"plan": plan, "error": None}
    except Exception as e:
        return {"plan": "", "error": str(e)}
