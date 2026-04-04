"""
Agent-Driven Schema Indexing Tools
====================================
These tools are called by PRISM agents during query processing to:

  1. search_relevant_tables   — semantic vector search across all indexed tables
  2. index_table_if_new       — fetch + embed + store a table on first encounter
  3. get_indexed_table        — retrieve stored metadata without re-fetching
  4. bulk_index_schema        — index all tables in a catalog.schema (background)
  5. list_indexed_tables      — audit what's already in the index
  6. refresh_table_index      — force re-fetch of a table (schema changed)

How agent-driven indexing works:
  - Schema Discovery Agent calls search_relevant_tables("revenue by category")
  - Returns top-15 table matches from the vector store
  - For any table NOT yet indexed, agents call index_table_if_new(table_id)
  - That triggers: Databricks SDK fetch → embed → upsert into ChromaDB/pgvector
  - Next time the same table is encountered: instant cache hit, no API call

This makes the index grow organically through query traffic, without a
separate batch pipeline. High-traffic tables stay fresh; rare tables are
indexed on first access.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from core.schema_index import get_schema_index

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------ #
# Tool 1: Semantic Search                                              #
# ------------------------------------------------------------------ #

def search_relevant_tables(
    query: str,
    top_k: int = 15,
    catalog: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """
    Find the most relevant database tables for a natural language query using
    semantic vector search over the schema index.

    This is the PRIMARY table discovery tool for large databases (1000+ tables).
    Always call this FIRST before fetching individual table details.
    It searches table names, column names, descriptions, and comments using
    semantic similarity — not just keyword matching.

    Args:
        query: Natural language description of what you're looking for.
               e.g. "revenue by product category over time"
               e.g. "customer churn and subscription status"
        top_k: Number of top matching tables to return (default 15, max 30)
        catalog: Optional Unity Catalog catalog name to restrict search
        schema:  Optional schema name to restrict search

    Returns:
        dict with:
            - results (list): Top matching tables with scores and metadata
            - total_indexed (int): Total tables in the index
            - query (str): The search query used
            - note (str): Guidance if index is empty
    """
    index = get_schema_index()
    top_k = min(top_k, 30)

    total = index.get_indexed_count()
    if total == 0:
        return {
            "results": [],
            "total_indexed": 0,
            "query": query,
            "note": (
                "Schema index is empty. Call bulk_index_schema(catalog, schema) "
                "to populate it, or call index_table_if_new(table_id) for specific tables."
            ),
        }

    results = index.search(
        query=query,
        top_k=top_k,
        catalog_filter=catalog,
        schema_filter=schema,
    )

    formatted = []
    for r in results:
        meta = r.metadata
        # Deserialize columns summary
        try:
            columns = json.loads(meta.get("columns_json", "[]"))
        except Exception:
            columns = []

        col_names = [c.get("name", "") for c in columns[:20]]

        formatted.append({
            "table_id": r.table_id,
            "catalog": meta.get("catalog", ""),
            "schema": meta.get("schema", ""),
            "table": meta.get("table", ""),
            "description": meta.get("comment", ""),
            "similarity_score": r.score,
            "column_count": len(columns),
            "key_columns": col_names[:10],
            "row_count": int(meta.get("row_count", -1)),
            "indexed_at": meta.get("indexed_at", ""),
        })

    return {
        "results": formatted,
        "total_indexed": total,
        "query": query,
        "note": f"Found {len(formatted)} relevant tables out of {total} indexed.",
    }


# ------------------------------------------------------------------ #
# Tool 2: Index a table on first encounter                            #
# ------------------------------------------------------------------ #

def index_table_if_new(
    table_id: str,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Fetch, embed, and store a table's metadata in the schema index IF it
    isn't already indexed. Idempotent — safe to call multiple times.

    Call this whenever you encounter a table name (from a query result,
    foreign key, or user mention) that you need more detail on.
    If the table is already indexed, this returns the cached metadata
    immediately without any API calls.

    Args:
        table_id: Fully-qualified table ID: "catalog.schema.table"
                  e.g. "main.sales.order_items"
                  e.g. "hive_metastore.default.customers"
        force_refresh: Set True to force re-fetch even if already indexed
                       (use when you suspect the schema has changed)

    Returns:
        dict with:
            - indexed (bool): True if newly indexed, False if already existed
            - table_id (str): The table ID
            - metadata (dict): Full table metadata (columns, types, comments)
            - from_cache (bool): True if returned from index without fetching
            - error (str | None): Error message if indexing failed
    """
    index = get_schema_index()

    # Cache hit — return immediately
    if not force_refresh and index.is_indexed(table_id):
        cached = index.get_by_id(table_id)
        return {
            "indexed": False,
            "table_id": table_id,
            "metadata": cached,
            "from_cache": True,
            "error": None,
        }

    # Need to fetch — try Databricks first, then SQLAlchemy
    metadata = _fetch_table_metadata(table_id)
    if metadata.get("error"):
        return {
            "indexed": False,
            "table_id": table_id,
            "metadata": None,
            "from_cache": False,
            "error": metadata["error"],
        }

    # Upsert into vector index
    try:
        index.upsert(metadata)
        logger.info("Table indexed", table_id=table_id)
        return {
            "indexed": True,
            "table_id": table_id,
            "metadata": metadata,
            "from_cache": False,
            "error": None,
        }
    except Exception as e:
        logger.error("Failed to index table", table_id=table_id, error=str(e))
        return {
            "indexed": False,
            "table_id": table_id,
            "metadata": metadata,
            "from_cache": False,
            "error": f"Indexing failed: {e}. Metadata still returned.",
        }


def _fetch_table_metadata(table_id: str) -> dict[str, Any]:
    """
    Fetch table metadata using the best available source.

    Priority:
      1. Databricks MCP server (if MCP_ENABLED=true and client is connected)
      2. Databricks native SDK (if DATABRICKS_HOST is configured)
      3. SQLAlchemy (fallback for non-Databricks databases)
    """
    # ── 1. Try MCP first ──────────────────────────────────────────────
    try:
        from core.mcp_client import get_mcp_client
        mcp_client = get_mcp_client()
        if mcp_client is not None and mcp_client.has_tool("get_table"):
            from agents.tools.mcp_tools import mcp_get_table_metadata
            result = mcp_get_table_metadata(table_id)
            if result.get("success") and result.get("columns") is not None:
                # Normalise to standard metadata shape
                metadata = {
                    "full_name": table_id,
                    "catalog": result.get("catalog", ""),
                    "schema": result.get("schema", ""),
                    "table": result.get("table", table_id.split(".")[-1]),
                    "columns": result.get("columns", []),
                    "comment": result.get("comment", ""),
                    "row_count": result.get("row_count", -1),
                    "table_type": result.get("table_type", "TABLE"),
                    "source": "mcp",
                }
                return metadata
            elif not result.get("success"):
                logger.debug("MCP get_table failed, trying native SDK", error=result.get("error"))
    except Exception as exc:
        logger.debug("MCP metadata fetch error", table_id=table_id, error=str(exc))

    # ── 2. Native Databricks SDK ───────────────────────────────────────
    from core.databricks import get_databricks_connector, UCTableRef

    db_connector = get_databricks_connector()
    if db_connector is not None:
        try:
            ref = UCTableRef.parse(table_id)
            metadata = db_connector.get_table_metadata(ref)
            stats = db_connector.get_table_stats(ref)
            metadata["row_count"] = stats.get("row_count", -1)
            metadata["size_bytes"] = stats.get("size_bytes", -1)
            metadata["partitioning"] = stats.get("partitioning", [])
            metadata["clustering_columns"] = stats.get("clustering_columns", [])
            metadata["full_name"] = table_id
            metadata["source"] = "databricks_sdk"
            return metadata
        except Exception as e:
            return {"error": f"Databricks fetch failed for {table_id}: {e}"}

    # ── 3. SQLAlchemy fallback ─────────────────────────────────────────
    try:
        import asyncio
        from core.schema_manager import SchemaManager
        from core.database import get_db_manager

        async def _async_fetch():
            db = await get_db_manager()
            manager = SchemaManager(db)
            schema = await manager.get_full_schema(include_samples=True)
            # table_id may be just the table name for non-UC databases
            table_name = table_id.split(".")[-1]
            table = schema.get_table(table_name)
            if not table:
                return {"error": f"Table '{table_name}' not found in schema"}
            result = table.to_dict()
            result["full_name"] = table_id
            result["source"] = "sqlalchemy"
            return result

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_async_fetch())
        finally:
            loop.close()
    except Exception as e:
        return {"error": f"SQLAlchemy fallback failed: {e}"}


# ------------------------------------------------------------------ #
# Tool 3: Get indexed table metadata                                   #
# ------------------------------------------------------------------ #

def get_indexed_table(table_id: str) -> dict[str, Any]:
    """
    Retrieve full metadata for a table from the schema index.
    Does NOT fetch from database — only returns what's already indexed.

    Use this after confirming a table is indexed (search returned it)
    to get full column-level detail for SQL generation.

    Args:
        table_id: Fully-qualified table ID: "catalog.schema.table"

    Returns:
        dict with:
            - found (bool): Whether the table is in the index
            - table_id (str)
            - metadata (dict): Full metadata including columns list
            - columns (list): Parsed column list for convenience
    """
    index = get_schema_index()
    metadata = index.get_by_id(table_id)

    if not metadata:
        return {
            "found": False,
            "table_id": table_id,
            "metadata": None,
            "columns": [],
        }

    columns = metadata.get("columns", [])
    if not columns and "columns_json" in metadata:
        try:
            columns = json.loads(metadata["columns_json"])
        except Exception:
            columns = []

    return {
        "found": True,
        "table_id": table_id,
        "metadata": metadata,
        "columns": columns,
    }


# ------------------------------------------------------------------ #
# Tool 4: Bulk index a schema (all tables in a catalog.schema)        #
# ------------------------------------------------------------------ #

def bulk_index_schema(
    catalog: str,
    schema: str,
    max_tables: int = 500,
) -> dict[str, Any]:
    """
    Index all tables in a Unity Catalog schema in one call.

    Use this for initial bootstrapping of a new catalog/schema, or to
    refresh an entire schema after major changes. Processes up to
    max_tables tables — for very large schemas, call in batches.

    This operation may take several minutes for large schemas. It runs
    synchronously but can be called in a background task.

    Args:
        catalog: Unity Catalog catalog name (e.g. "main", "hive_metastore")
        schema:  Schema/database name (e.g. "sales", "analytics")
        max_tables: Maximum tables to index in this call (default 500)

    Returns:
        dict with:
            - indexed (int): New tables indexed
            - skipped (int): Already-indexed tables skipped
            - failed (int): Tables that failed to index
            - errors (list): Error details for failed tables
            - total_in_schema (int): Total tables discovered in schema
    """
    from core.databricks import get_databricks_connector

    db_connector = get_databricks_connector()
    if not db_connector:
        return {
            "indexed": 0, "skipped": 0, "failed": 0,
            "errors": ["No Databricks connector configured"],
            "total_in_schema": 0,
        }

    index = get_schema_index()

    # List tables in the schema
    try:
        tables = db_connector.list_tables(catalog, schema)
    except Exception as e:
        return {
            "indexed": 0, "skipped": 0, "failed": 1,
            "errors": [f"Failed to list tables in {catalog}.{schema}: {e}"],
            "total_in_schema": 0,
        }

    total = len(tables)
    to_process = tables[:max_tables]

    indexed = skipped = failed = 0
    errors = []

    for table_info in to_process:
        table_id = table_info.get("full_name", "")
        if not table_id:
            failed += 1
            continue

        if index.is_indexed(table_id):
            skipped += 1
            continue

        result = index_table_if_new(table_id)
        if result.get("error"):
            failed += 1
            errors.append(f"{table_id}: {result['error']}")
        else:
            indexed += 1

    logger.info(
        "Bulk index complete",
        catalog=catalog, schema=schema,
        indexed=indexed, skipped=skipped, failed=failed,
    )
    return {
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors[:10],
        "total_in_schema": total,
        "processed": len(to_process),
        "truncated": total > max_tables,
    }


# ------------------------------------------------------------------ #
# Tool 5: List indexed tables                                          #
# ------------------------------------------------------------------ #

def list_indexed_tables(
    catalog: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    List tables currently in the schema index.

    Use this to check what's already indexed before deciding whether
    to call bulk_index_schema or index_table_if_new.

    Args:
        catalog: Optional filter by Unity Catalog catalog name
        limit:   Max entries to return (default 50)

    Returns:
        dict with:
            - tables (list): Indexed table summaries
            - total_indexed (int): Total tables in the entire index
    """
    index = get_schema_index()
    tables = index.list_indexed_tables(catalog_filter=catalog, limit=limit)
    return {
        "tables": tables,
        "total_indexed": index.get_indexed_count(),
        "catalog_filter": catalog,
    }


# ------------------------------------------------------------------ #
# Tool 6: Refresh a table's index entry                               #
# ------------------------------------------------------------------ #

def refresh_table_index(table_id: str) -> dict[str, Any]:
    """
    Force re-fetch and re-index a table's metadata.

    Call this when you know a table's schema has changed (new columns,
    changed types, updated descriptions) and the cached version is stale.

    Args:
        table_id: Fully-qualified table ID: "catalog.schema.table"

    Returns:
        Same structure as index_table_if_new with force_refresh=True
    """
    return index_table_if_new(table_id, force_refresh=True)
