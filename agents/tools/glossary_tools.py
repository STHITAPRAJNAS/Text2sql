"""
Business Glossary Tools
=======================
Agent-facing tools for business term resolution during Phase R reasoning.

The Deep Think Query Analyzer calls `lookup_glossary` for every business
term it identifies before schema linking. This converts ambiguous natural
language ("active customers", "revenue", "YTD") into precise SQL fragments
before any schema search happens.

Latency: O(1) from in-memory cache — no DB call on hot path.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def lookup_glossary(term: str) -> dict[str, Any]:
    """
    Look up a business term in the enterprise glossary.

    Call this for any business/domain-specific term before schema linking.
    Returns the exact table, column, or filter SQL to use.

    Examples:
      lookup_glossary("active customers")
      → {"term": "active customers", "table_name": "users",
         "filter_sql": "status = 'active'", "description": "..."}

      lookup_glossary("revenue")
      → {"term": "revenue", "column_name": "order_items.amount",
         "example_sql": "SUM(order_items.unit_price * quantity)"}

    Args:
        term: Business term to resolve (case-insensitive)

    Returns:
        Glossary entry dict or {"term": term, "found": false} if not found
    """
    from core.glossary import lookup_sync
    result = lookup_sync(term)
    if result:
        return {**result, "found": True}
    return {
        "term": term,
        "found": False,
        "message": (
            f"Term '{term}' not in glossary. "
            "Proceed with schema search and document assumptions. "
            "Consider adding this term via POST /api/v1/glossary."
        ),
    }


def search_glossary(query: str) -> dict[str, Any]:
    """
    Search the glossary for terms matching a keyword.
    Use when the exact term name is unknown.

    Args:
        query: Search keyword (e.g., "customer", "order")

    Returns:
        {"results": [...], "count": N}
    """
    import asyncio
    from core.glossary import search

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(asyncio.run, search(query, limit=10)).result(timeout=3)
        else:
            results = loop.run_until_complete(search(query, limit=10))
    except Exception as exc:
        logger.warning("Glossary search failed", error=str(exc))
        results = []

    return {"results": results, "count": len(results)}


def add_glossary_term(
    term: str,
    description: str,
    table_name: str = "",
    column_name: str = "",
    filter_sql: str = "",
    example_sql: str = "",
) -> dict[str, Any]:
    """
    Add or update a business term in the glossary.

    Call this when you discover a recurring business term that should be
    standardized for future queries. This improves resolution quality for
    all future users.

    Args:
        term: The business term (e.g., "churned customers")
        description: Plain-English explanation
        table_name: Primary table this term refers to
        column_name: Specific column if term maps to one column
        filter_sql: WHERE clause fragment (e.g., "status = 'inactive'")
        example_sql: Example SQL using this term

    Returns:
        {"status": "ok", "term": term}
    """
    import asyncio
    from core.glossary import upsert

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, upsert(
                    term=term, table_name=table_name, column_name=column_name,
                    filter_sql=filter_sql, description=description, example_sql=example_sql,
                )).result(timeout=3)
        else:
            result = loop.run_until_complete(upsert(
                term=term, table_name=table_name, column_name=column_name,
                filter_sql=filter_sql, description=description, example_sql=example_sql,
            ))
        return result
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
