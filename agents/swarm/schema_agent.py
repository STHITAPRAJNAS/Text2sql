"""
PRISM Schema Discovery & Metadata Enrichment Agents
Phase P of the PRISM pipeline — run in parallel.

For large databases (Databricks Unity Catalog with thousands of tables):
  - Schema Discovery Agent uses search_relevant_tables FIRST (vector search)
    to find the 10–15 most relevant tables for the user's query.
  - It then calls index_table_if_new for any tables it encounters that
    aren't yet in the index, so they're available for future queries.
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


def create_schema_discovery_agent() -> Agent:
    """
    Create the Schema Discovery Agent.

    Tool priority for large databases:
      1. search_relevant_tables  → semantic vector search (fast, always first)
      2. index_table_if_new      → auto-index tables not yet in the store
      3. get_indexed_table       → retrieve cached column-level details
      4. get_table_details       → direct DB fetch (when not indexed)
      5. find_related_tables     → discover join paths
      6. get_sample_values       → understand filter column values
      7. search_schema_by_keyword → keyword fallback search
      8. bulk_index_schema       → bootstrap a new catalog/schema
    """
    settings = get_settings()

    return Agent(
        name="schema_discovery_agent",
        model=settings.llm.schema_agent_model,
        description=(
            "Database schema expert. For large databases (Unity Catalog), uses semantic "
            "vector search to find relevant tables among thousands, auto-indexes newly "
            "encountered tables, and builds focused schema context for SQL generation."
        ),
        instruction=PromptLibrary.SCHEMA_DISCOVERY_AGENT,
        tools=[
            # Vector index tools (primary for large databases)
            search_relevant_tables,
            index_table_if_new,
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
        ],
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
            search_schema_by_keyword,
            get_database_schema,
        ],
    )
