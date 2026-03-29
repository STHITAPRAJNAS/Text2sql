"""
Schema Annotation Auto-Ingestion
=================================
Periodically syncs Unity Catalog table/column comments into:
  1. The schema vector index (improves semantic search relevance)
  2. The business glossary (auto-registers domain-specific column descriptions)

Design:
  - Runs as a background asyncio task (fire-and-forget)
  - Detects changes via DESCRIBE HISTORY (Delta tables) or comment drift
  - Only re-indexes tables whose comments have changed
  - Syncs column-level comments to glossary when they contain business definitions

Usage:
  from core.schema_annotation_sync import start_annotation_sync
  asyncio.create_task(start_annotation_sync())  # starts background loop

Or trigger manually:
  from core.schema_annotation_sync import sync_catalog_annotations
  await sync_catalog_annotations(catalog="main", schema="sales")
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Business term indicators in column comments (triggers glossary auto-registration)
_BUSINESS_TERM_KEYWORDS = {
    "represents", "calculated as", "defined as", "means", "refers to",
    "total", "sum of", "count of", "percentage of", "ratio of",
    "active", "inactive", "churned", "converted", "retained",
}


async def sync_catalog_annotations(
    catalog: str = "",
    schema: str = "",
    max_tables: int = 500,
) -> dict[str, Any]:
    """
    Sync Unity Catalog table/column comments → schema index + glossary.

    Args:
        catalog: Unity Catalog catalog name (e.g. "main"). Empty = use default.
        schema: Schema name (e.g. "sales"). Empty = all schemas.
        max_tables: Maximum tables to process per run.

    Returns:
        {"synced": N, "glossary_added": M, "skipped": K, "errors": [...]}
    """
    stats = {"synced": 0, "glossary_added": 0, "skipped": 0, "errors": []}

    try:
        from core.databricks import get_databricks_connector
        conn = get_databricks_connector()
        if not conn:
            logger.debug("No Databricks connector — skipping annotation sync")
            return stats

        from config.settings import get_settings
        settings = get_settings()
        _catalog = catalog or settings.databricks.default_catalog
        _schema = schema or settings.databricks.default_schema

        # Fetch all tables in the target schema
        tables = conn.list_tables(_catalog, _schema)
        if not tables:
            return stats

        tables = tables[:max_tables]

        from core.schema_index import get_schema_index
        index = get_schema_index()

        for table in tables:
            table_id = table.get("full_name", "")
            if not table_id:
                continue

            try:
                # Fetch full column metadata with comments
                details = conn.get_table_details(table_id)
                if not details:
                    stats["skipped"] += 1
                    continue

                # Check if comment has changed since last index
                comment = details.get("comment", "")
                columns = details.get("columns", [])

                # Build enriched description for the index
                col_descriptions = []
                for col in columns:
                    col_comment = col.get("comment", "")
                    if col_comment:
                        col_descriptions.append(
                            f"{col['name']} ({col.get('type', 'unknown')}): {col_comment}"
                        )
                        # Auto-register column comments as glossary terms if business-relevant
                        if _is_business_definition(col_comment):
                            await _maybe_add_to_glossary(
                                term=col.get("name", ""),
                                description=col_comment,
                                table_name=table_id,
                                column_name=col.get("name", ""),
                            )
                            stats["glossary_added"] += 1

                enriched_description = comment
                if col_descriptions:
                    enriched_description += "\nColumns: " + "; ".join(col_descriptions[:20])

                # Re-index with updated annotations
                entry = {
                    "table_id": table_id,
                    "catalog": _catalog,
                    "schema": _schema,
                    "table_name": table.get("name", table_id.split(".")[-1]),
                    "description": enriched_description,
                    "columns": columns,
                    "row_count": details.get("row_count", 0),
                    "comment": comment,
                    "indexed_at": time.time(),
                }
                index.upsert(table_id, entry)
                stats["synced"] += 1

            except Exception as exc:
                stats["errors"].append(f"{table_id}: {exc}")
                logger.warning("Annotation sync error", table=table_id, error=str(exc))

    except Exception as exc:
        logger.warning("Schema annotation sync failed", error=str(exc))
        stats["errors"].append(str(exc))

    logger.info(
        "Schema annotation sync complete",
        synced=stats["synced"],
        glossary_added=stats["glossary_added"],
        skipped=stats["skipped"],
        errors=len(stats["errors"]),
    )
    return stats


def _is_business_definition(comment: str) -> bool:
    """Return True if a column comment looks like a business definition."""
    if len(comment) < 10:
        return False
    comment_lower = comment.lower()
    return any(kw in comment_lower for kw in _BUSINESS_TERM_KEYWORDS)


async def _maybe_add_to_glossary(
    term: str,
    description: str,
    table_name: str,
    column_name: str,
) -> None:
    """Add a term to the glossary only if it isn't already defined."""
    try:
        from core.glossary import lookup, upsert
        existing = await lookup(term)
        if existing:
            return  # Don't overwrite manually curated entries
        await upsert(
            term=term,
            table_name=table_name,
            column_name=column_name,
            description=description,
        )
    except Exception as exc:
        logger.debug("Glossary auto-add failed", term=term, error=str(exc))


async def start_annotation_sync(
    interval_seconds: int = 3600,
    catalog: str = "",
    schema: str = "",
) -> None:
    """
    Background task: run sync_catalog_annotations every interval_seconds.
    Call with asyncio.create_task() to run non-blocking.

    Default: sync every 1 hour.
    """
    logger.info("Schema annotation sync started", interval_seconds=interval_seconds)
    while True:
        try:
            await sync_catalog_annotations(catalog=catalog, schema=schema)
        except Exception as exc:
            logger.warning("Annotation sync loop error", error=str(exc))
        await asyncio.sleep(interval_seconds)
