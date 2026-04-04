"""
Databricks MCP Server — ADK Tool Wrappers
==========================================
Exposes the Databricks MCP server's tools as plain Python functions that
Google ADK agents can call directly.

All functions follow the same contract as the rest of ``agents/tools/``:
  - Synchronous (ADK wraps them as FunctionTool automatically)
  - Return ``dict[str, Any]`` — never raise
  - Gracefully degrade when MCP is not configured

Standard Databricks MCP tool mapping (configurable via MCPSettings):
    mcp_execute_query      → execute_statement
    mcp_list_catalogs      → list_catalogs
    mcp_list_schemas       → list_schemas
    mcp_list_tables        → list_tables
    mcp_get_table_metadata → get_table
    mcp_search_tables      → search_tables
    mcp_discover_tools     → list_tools (meta, no server round-trip)

The Schema Discovery Agent calls these tools alongside the existing
``indexing_tools`` — MCP tools are tried first, then native SDK, then
SQLAlchemy (see ``indexing_tools._fetch_table_metadata``).
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── Lazy import helpers ────────────────────────────────────────────────────────

def _client():
    """Return the singleton MCP client, or None."""
    from core.mcp_client import get_mcp_client
    return get_mcp_client()


def _not_configured(tool_name: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": f"MCP client not configured — cannot call {tool_name}",
        "mcp_available": False,
    }


def _no_tool(logical_name: str, actual_name: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": (
            f"MCP server does not expose tool {actual_name!r} "
            f"(logical: {logical_name!r})"
        ),
        "available_tools": _client().list_tools() if _client() else [],
    }


# ------------------------------------------------------------------ #
# Tool 1: Execute SQL query                                            #
# ------------------------------------------------------------------ #

def mcp_execute_query(
    sql: str,
    catalog: str = "",
    schema: str = "",
    max_rows: int = 1000,
    warehouse_id: str = "",
) -> dict[str, Any]:
    """
    Execute a SQL statement on a Databricks SQL warehouse via the MCP server.

    Prefer this over the native Databricks connector when MCP is configured —
    it uses the MCP server's credential chain and respects Unity Catalog
    permissions transparently.

    Args:
        sql:          SQL statement to execute (SELECT / WITH / EXPLAIN)
        catalog:      Unity Catalog catalog name (overrides workspace default)
        schema:       Schema name (overrides workspace default)
        max_rows:     Maximum rows to return (default 1000)
        warehouse_id: SQL warehouse ID; leave empty to use server default

    Returns:
        dict with:
            - success (bool)
            - columns (list[str]): Column names
            - rows (list[dict]): Result rows
            - row_count (int)
            - truncated (bool): True if more rows exist
            - execution_time_ms (float)
            - error (str | None)
    """
    client = _client()
    if not client:
        return _not_configured("mcp_execute_query")

    tool = client.resolve_tool_name("execute_query")
    if not client.has_tool("execute_query"):
        return _no_tool("execute_query", tool)

    args: dict[str, Any] = {"statement": sql, "max_rows": max_rows}
    if catalog:
        args["catalog"] = catalog
    if schema:
        args["schema"] = schema
    if warehouse_id:
        args["warehouse_id"] = warehouse_id

    raw = client.call_tool_json(tool, args)
    if not raw["success"]:
        return {"success": False, "error": raw["error"], "columns": [], "rows": [], "row_count": 0}

    content = raw["content"]

    # Normalise: MCP servers return varying shapes
    if isinstance(content, dict):
        columns = content.get("columns") or content.get("schema", {}).get("columns", [])
        # columns may be list[str] or list[{"name": ...}]
        if columns and isinstance(columns[0], dict):
            columns = [c.get("name", str(c)) for c in columns]
        rows = content.get("rows") or content.get("data", [])
        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "row_count": content.get("row_count", len(rows)),
            "truncated": content.get("truncated", False),
            "execution_time_ms": content.get("execution_time_ms", 0.0),
            "error": None,
        }

    # Fallback: plain text response
    return {
        "success": True,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "execution_time_ms": 0.0,
        "raw_response": str(content),
        "error": None,
    }


# ------------------------------------------------------------------ #
# Tool 2: List catalogs                                                #
# ------------------------------------------------------------------ #

def mcp_list_catalogs() -> dict[str, Any]:
    """
    List all Unity Catalog catalogs accessible to the MCP server's credentials.

    Returns:
        dict with:
            - success (bool)
            - catalogs (list[str]): Catalog names
            - count (int)
    """
    client = _client()
    if not client:
        return _not_configured("mcp_list_catalogs")

    tool = client.resolve_tool_name("list_catalogs")
    if not client.has_tool("list_catalogs"):
        return _no_tool("list_catalogs", tool)

    raw = client.call_tool_json(tool, {})
    if not raw["success"]:
        return {"success": False, "error": raw["error"], "catalogs": [], "count": 0}

    content = raw["content"]
    if isinstance(content, dict):
        catalogs = content.get("catalogs", [])
    elif isinstance(content, list):
        catalogs = content
    else:
        # Try to extract from text
        catalogs = [line.strip() for line in str(content).splitlines() if line.strip()]

    # Normalise to list of strings
    names = []
    for c in catalogs:
        names.append(c.get("name", str(c)) if isinstance(c, dict) else str(c))

    return {"success": True, "catalogs": names, "count": len(names)}


# ------------------------------------------------------------------ #
# Tool 3: List schemas                                                 #
# ------------------------------------------------------------------ #

def mcp_list_schemas(catalog: str) -> dict[str, Any]:
    """
    List all schemas in a Unity Catalog catalog.

    Args:
        catalog: Catalog name (e.g. "main", "hive_metastore")

    Returns:
        dict with:
            - success (bool)
            - catalog (str)
            - schemas (list[str]): Schema names
            - count (int)
    """
    client = _client()
    if not client:
        return _not_configured("mcp_list_schemas")

    tool = client.resolve_tool_name("list_schemas")
    if not client.has_tool("list_schemas"):
        return _no_tool("list_schemas", tool)

    raw = client.call_tool_json(tool, {"catalog_name": catalog})
    if not raw["success"]:
        return {"success": False, "error": raw["error"], "catalog": catalog, "schemas": [], "count": 0}

    content = raw["content"]
    if isinstance(content, dict):
        schemas = content.get("schemas", [])
    elif isinstance(content, list):
        schemas = content
    else:
        schemas = [line.strip() for line in str(content).splitlines() if line.strip()]

    names = []
    for s in schemas:
        names.append(s.get("name", str(s)) if isinstance(s, dict) else str(s))

    return {"success": True, "catalog": catalog, "schemas": names, "count": len(names)}


# ------------------------------------------------------------------ #
# Tool 4: List tables                                                  #
# ------------------------------------------------------------------ #

def mcp_list_tables(catalog: str, schema: str) -> dict[str, Any]:
    """
    List all tables in a Unity Catalog catalog.schema.

    Args:
        catalog: Catalog name
        schema:  Schema name

    Returns:
        dict with:
            - success (bool)
            - catalog (str)
            - schema (str)
            - tables (list[dict]): Table summaries with name, comment, table_type
            - count (int)
    """
    client = _client()
    if not client:
        return _not_configured("mcp_list_tables")

    tool = client.resolve_tool_name("list_tables")
    if not client.has_tool("list_tables"):
        return _no_tool("list_tables", tool)

    raw = client.call_tool_json(tool, {"catalog_name": catalog, "schema_name": schema})
    if not raw["success"]:
        return {
            "success": False, "error": raw["error"],
            "catalog": catalog, "schema": schema, "tables": [], "count": 0,
        }

    content = raw["content"]
    if isinstance(content, dict):
        tables = content.get("tables", [])
    elif isinstance(content, list):
        tables = content
    else:
        tables = [{"name": line.strip()} for line in str(content).splitlines() if line.strip()]

    # Normalise to list[dict] with at least {"name": ..., "full_name": ...}
    normalised = []
    for t in tables:
        if isinstance(t, str):
            t = {"name": t}
        name = t.get("name", "")
        full_name = t.get("full_name") or f"{catalog}.{schema}.{name}"
        normalised.append({
            "name": name,
            "full_name": full_name,
            "catalog": catalog,
            "schema": schema,
            "table_type": t.get("table_type", "TABLE"),
            "comment": t.get("comment", ""),
        })

    return {
        "success": True,
        "catalog": catalog,
        "schema": schema,
        "tables": normalised,
        "count": len(normalised),
    }


# ------------------------------------------------------------------ #
# Tool 5: Get table metadata                                           #
# ------------------------------------------------------------------ #

def mcp_get_table_metadata(table_id: str) -> dict[str, Any]:
    """
    Get full column metadata for a table via the Databricks MCP server.

    Args:
        table_id: Three-part table reference: "catalog.schema.table"
                  e.g. "main.sales.order_items"

    Returns:
        dict with:
            - success (bool)
            - table_id (str)
            - catalog, schema, table (str)
            - columns (list[dict]): Column definitions (name, type, comment, nullable)
            - comment (str): Table-level description
            - row_count (int)
            - error (str | None)
    """
    client = _client()
    if not client:
        return _not_configured("mcp_get_table_metadata")

    tool = client.resolve_tool_name("get_table")
    if not client.has_tool("get_table"):
        return _no_tool("get_table", tool)

    # Parse three-part name
    parts = table_id.split(".")
    args: dict[str, Any] = {"full_name": table_id}
    if len(parts) == 3:
        args["catalog_name"] = parts[0]
        args["schema_name"] = parts[1]
        args["table_name"] = parts[2]
    elif len(parts) == 2:
        args["schema_name"] = parts[0]
        args["table_name"] = parts[1]
    else:
        args["table_name"] = table_id

    raw = client.call_tool_json(tool, args)
    if not raw["success"]:
        return {"success": False, "error": raw["error"], "table_id": table_id, "columns": []}

    content = raw["content"]
    if not isinstance(content, dict):
        return {
            "success": True,
            "table_id": table_id,
            "columns": [],
            "comment": str(content),
            "row_count": -1,
            "raw": str(content),
            "error": None,
        }

    # Normalise columns
    raw_cols = content.get("columns", [])
    columns = []
    for col in raw_cols:
        if isinstance(col, dict):
            columns.append({
                "name": col.get("name", ""),
                "type": col.get("type_text") or col.get("type", "STRING"),
                "comment": col.get("comment", ""),
                "nullable": col.get("nullable", True),
            })

    table_parts = table_id.split(".")
    return {
        "success": True,
        "table_id": table_id,
        "catalog": table_parts[0] if len(table_parts) > 2 else "",
        "schema": table_parts[1] if len(table_parts) > 2 else (table_parts[0] if len(table_parts) > 1 else ""),
        "table": table_parts[-1],
        "columns": columns,
        "comment": content.get("comment", ""),
        "row_count": content.get("row_count", -1),
        "table_type": content.get("table_type", "TABLE"),
        "storage_location": content.get("storage_location", ""),
        "full_name": table_id,
        "error": None,
    }


# ------------------------------------------------------------------ #
# Tool 6: Search tables                                                #
# ------------------------------------------------------------------ #

def mcp_search_tables(
    keyword: str,
    catalog: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search Unity Catalog tables by name or description via the MCP server.

    Useful as a fallback when ``search_relevant_tables`` (vector search)
    returns no results — keyword search covers exact matches.

    Args:
        keyword: Search term (matched against table name and comment)
        catalog: Optional catalog scope filter
        limit:   Max results to return (default 20)

    Returns:
        dict with:
            - success (bool)
            - results (list[dict]): Matching tables
            - count (int)
            - keyword (str)
    """
    client = _client()
    if not client:
        return _not_configured("mcp_search_tables")

    if not client.has_tool("search_tables"):
        # Graceful degradation: list all tables and filter client-side
        return _keyword_search_fallback(keyword, catalog, limit)

    tool = client.resolve_tool_name("search_tables")
    args: dict[str, Any] = {"query": keyword, "max_results": limit}
    if catalog:
        args["catalog_name"] = catalog

    raw = client.call_tool_json(tool, args)
    if not raw["success"]:
        return {"success": False, "error": raw["error"], "results": [], "count": 0, "keyword": keyword}

    content = raw["content"]
    if isinstance(content, dict):
        tables = content.get("tables", content.get("results", []))
    elif isinstance(content, list):
        tables = content
    else:
        tables = []

    results = []
    for t in tables:
        if isinstance(t, str):
            results.append({"full_name": t, "comment": ""})
        elif isinstance(t, dict):
            results.append({
                "full_name": t.get("full_name", t.get("name", "")),
                "comment": t.get("comment", ""),
                "table_type": t.get("table_type", "TABLE"),
            })

    return {"success": True, "results": results, "count": len(results), "keyword": keyword}


def _keyword_search_fallback(keyword: str, catalog: str, limit: int) -> dict[str, Any]:
    """
    Client-side keyword filter when the MCP server has no search_tables tool.
    Calls mcp_list_catalogs → mcp_list_schemas → mcp_list_tables internally.
    """
    client = _client()
    if not client:
        return {"success": False, "error": "MCP not configured", "results": [], "count": 0, "keyword": keyword}

    kw_lower = keyword.lower()
    results = []

    # Resolve catalog list
    cat_result = mcp_list_catalogs()
    catalogs = [catalog] if catalog else cat_result.get("catalogs", [])

    for cat in catalogs[:3]:  # limit scan to avoid slowness
        schema_result = mcp_list_schemas(cat)
        for schema in schema_result.get("schemas", []):
            table_result = mcp_list_tables(cat, schema)
            for tbl in table_result.get("tables", []):
                name = tbl.get("name", "")
                comment = tbl.get("comment", "")
                if kw_lower in name.lower() or kw_lower in comment.lower():
                    results.append(tbl)
                    if len(results) >= limit:
                        return {"success": True, "results": results, "count": len(results), "keyword": keyword}

    return {"success": True, "results": results, "count": len(results), "keyword": keyword}


# ------------------------------------------------------------------ #
# Tool 7: Discover tools (meta)                                        #
# ------------------------------------------------------------------ #

def mcp_discover_tools() -> dict[str, Any]:
    """
    Return the list of tools advertised by the connected MCP server.

    Use this to understand what capabilities are available before calling
    other mcp_* tools.  This does NOT make a network call — the tool list
    is cached at connection time.

    Returns:
        dict with:
            - success (bool)
            - tools (list[str]): All available tool names
            - count (int)
            - logical_mapping (dict): PRISM logical name → actual MCP tool name
    """
    client = _client()
    if not client:
        return {
            "success": False,
            "error": "MCP client not configured",
            "tools": [],
            "count": 0,
            "logical_mapping": {},
        }

    tools = client.list_tools()
    logical = {
        op: client.resolve_tool_name(op)
        for op in [
            "execute_query", "list_catalogs", "list_schemas",
            "list_tables", "get_table", "search_tables",
        ]
    }

    return {
        "success": True,
        "tools": tools,
        "count": len(tools),
        "logical_mapping": logical,
    }
