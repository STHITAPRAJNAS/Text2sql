"""
PRISM Schema Discovery & Metadata Enrichment Agents
Phase P of the PRISM pipeline — run in parallel.

For large databases (Databricks Unity Catalog with thousands of tables):
  - Schema Discovery Agent uses search_relevant_tables FIRST (vector search)
    to find the 10–15 most relevant tables for the user's query.
  - It then calls index_table_if_new for any tables it encounters that
    aren't yet in the index, so they're available for future queries.
  - For previously indexed tables, it checks for schema changes via
    Delta DESCRIBE HISTORY and refreshes stale entries automatically.
  - Full table details are fetched ONLY for the relevant subset.

This design means the schema context passed to Phase R is always focused
(~3K tokens) regardless of whether the catalog has 50 or 50,000 tables.
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.schema_tools import (
    get_database_schema,
    get_table_details,
    find_related_tables,
    get_sample_values,
    search_schema_by_keyword,
)
from agents.tools.indexing_tools import (
    search_relevant_tables,
    index_table_if_new,
    get_indexed_table,
    list_indexed_tables,
    bulk_index_schema,
    refresh_table_index,
)
from agents.tools.mcp_tools import (
    mcp_execute_query,
    mcp_list_catalogs,
    mcp_list_schemas,
    mcp_list_tables,
    mcp_get_table_metadata,
    mcp_search_tables,
    mcp_discover_tools,
)


def _make_check_schema_change_tool():
    """
    Create a schema-change-check tool as a plain function.
    Wraps the async check in a sync ADK-compatible tool.
    """
    def check_table_for_changes(table_id: str) -> dict:
        """
        Check if a table has changed since it was last indexed.
        If changed, automatically refreshes the index entry.

        Use this after retrieving a table from the index to ensure
        the schema metadata is up-to-date.

        Args:
            table_id: Full table reference e.g. "main.sales.orders"

        Returns:
            {"changed": bool, "refreshed": bool, "table_id": table_id}
        """
        import asyncio
        from core.schema_change_detector import check_and_refresh_if_changed
        try:
            # Run the async function in the current event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule as a task and get result
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run, check_and_refresh_if_changed(table_id)
                        )
                        refreshed = future.result(timeout=10)
                else:
                    refreshed = loop.run_until_complete(
                        check_and_refresh_if_changed(table_id)
                    )
            except RuntimeError:
                refreshed = asyncio.run(check_and_refresh_if_changed(table_id))

            return {
                "table_id": table_id,
                "changed": refreshed,
                "refreshed": refreshed,
                "status": "refreshed" if refreshed else "up_to_date",
            }
        except Exception as exc:
            return {
                "table_id": table_id,
                "changed": False,
                "refreshed": False,
                "status": f"check_failed: {exc}",
            }

    return check_table_for_changes


# Create the tool once at module load time
check_table_for_changes = _make_check_schema_change_tool()


def create_schema_discovery_agent() -> Agent:
    """
    Create the Schema Discovery Agent.

    Tool priority for large databases:
      1. search_relevant_tables    → semantic vector search (fast, always first)
      2. index_table_if_new        → auto-index tables not yet in the store
      3. check_table_for_changes   → detect Delta table schema changes, auto-refresh
      4. get_indexed_table         → retrieve cached column-level details
      5. get_table_details         → direct DB fetch (when not indexed)
      6. find_related_tables       → discover join paths
      7. get_sample_values         → understand filter column values
      8. search_schema_by_keyword  → keyword fallback search
      9. bulk_index_schema         → bootstrap a new catalog/schema

    When MCP is enabled (MCP_ENABLED=true), the following MCP tools are also
    registered and take priority for Databricks operations:
      mcp_execute_query            → execute SQL via MCP server
      mcp_get_table_metadata       → column metadata via MCP server
      mcp_list_catalogs/schemas/tables → Unity Catalog navigation via MCP
      mcp_search_tables            → full-text table search via MCP
      mcp_discover_tools           → introspect available MCP tools
    """
    settings = get_settings()

    # Core tool list (always present)
    tools = [
        # Vector index tools (primary for large databases)
        search_relevant_tables,
        index_table_if_new,
        check_table_for_changes,
        get_indexed_table,
        list_indexed_tables,
        bulk_index_schema,
        refresh_table_index,
        # Direct DB tools (fallback / supplementary)
        get_database_schema,
        get_table_details,
        find_related_tables,
        get_sample_values,
        search_schema_by_keyword,
    ]

    # Prepend MCP tools when MCP is configured
    if settings.mcp.is_configured:
        mcp_tools = [
            mcp_discover_tools,       # Introspect first — agents can plan tool use
            mcp_list_catalogs,
            mcp_list_schemas,
            mcp_list_tables,
            mcp_get_table_metadata,
            mcp_search_tables,
            mcp_execute_query,        # Also available for quick validation queries
        ]
        tools = mcp_tools + tools

    return Agent(
        name="schema_discovery_agent",
        model=settings.llm.schema_agent_model,
        description=(
            "Database schema expert. For large databases (Unity Catalog), uses semantic "
            "vector search to find relevant tables among thousands, auto-indexes newly "
            "encountered tables, detects schema changes via Delta DESCRIBE HISTORY, "
            "and builds focused schema context for SQL generation. "
            + ("MCP server connected — can navigate Unity Catalog natively via MCP tools. "
               if settings.mcp.is_configured else "")
        ),
        instruction=PromptLibrary.SCHEMA_DISCOVERY_AGENT,
        tools=tools,
    )


def create_metadata_enrichment_agent() -> Agent:
    """
    Create the Metadata Enrichment Agent.

    For Unity Catalog: retrieves table comments, column descriptions,
    and owner information from the indexed metadata. Maps business
    terminology to schema elements using the business glossary.
    """
    settings = get_settings()

    return Agent(
        name="metadata_enrichment_agent",
        model=settings.llm.schema_agent_model,
        description=(
            "Business metadata specialist. Enriches schema context with business "
            "terminology, column descriptions from Unity Catalog, and glossary mappings."
        ),
        instruction=PromptLibrary.METADATA_ENRICHMENT_AGENT,
        tools=[
            search_relevant_tables,
            get_indexed_table,
            check_table_for_changes,
            search_schema_by_keyword,
            get_database_schema,
        ],
    )
