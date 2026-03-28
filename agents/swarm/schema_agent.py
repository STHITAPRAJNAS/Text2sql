"""
PRISM Schema Discovery & Metadata Enrichment Agents
Phase P of the PRISM pipeline — runs in parallel to discover schema and business context.
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


def create_schema_discovery_agent() -> Agent:
    """
    Create the Schema Discovery Agent.

    Responsibilities:
    - Fetch complete database schema
    - Identify relevant tables for the query
    - Discover relationships and join paths
    - Collect sample values for filter columns

    Returns an initialized Google ADK Agent.
    """
    settings = get_settings()

    return Agent(
        name="schema_discovery_agent",
        model=settings.llm.schema_agent_model,
        description=(
            "Database schema expert that discovers and analyzes the database structure, "
            "tables, columns, relationships, and sample data to provide rich schema context."
        ),
        instruction=PromptLibrary.SCHEMA_DISCOVERY_AGENT,
        tools=[
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

    Responsibilities:
    - Map business terms to schema elements
    - Identify calculated/derived metrics
    - Apply enterprise data dictionary context
    - Resolve ambiguous business terminology

    Returns an initialized Google ADK Agent.
    """
    settings = get_settings()

    return Agent(
        name="metadata_enrichment_agent",
        model=settings.llm.schema_agent_model,
        description=(
            "Business metadata specialist that enriches raw schema information with "
            "business context, synonyms, calculated metrics, and enterprise terminology."
        ),
        instruction=PromptLibrary.METADATA_ENRICHMENT_AGENT,
        tools=[
            search_schema_by_keyword,
            get_database_schema,
        ],
    )
