"""
PRISM SQL Generator Agent
Phase I of the PRISM pipeline — generates SQL from structured query analysis.

Uses the query analysis + schema context + few-shot examples to generate
production-quality, dialect-aware SQL queries.
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.schema_tools import (
    get_table_details,
    find_related_tables,
    get_sample_values,
    search_schema_by_keyword,
)
from agents.tools.few_shot_tools import get_similar_examples


def create_sql_generator_agent() -> Agent:
    """
    Create the SQL Generator Agent.

    This agent:
    1. Receives the structured QueryAnalysis from the Deep Think Analyzer
    2. Retrieves similar few-shot examples
    3. Generates production-quality SQL following best practices
    4. Provides confidence scoring and explains key decisions
    5. Handles multi-dialect SQL generation

    SQL Generation Principles:
    - Use CTEs for complex multi-step queries
    - Explicit column lists (no SELECT *)
    - Proper NULL handling with COALESCE
    - Appropriate LIMIT clauses
    - Dialect-correct syntax and functions

    Uses the Pro model for maximum accuracy.
    """
    settings = get_settings()

    return Agent(
        name="sql_generator_agent",
        model=settings.llm.sql_generator_model,
        description=(
            "Expert SQL generator that produces production-quality, optimized SQL queries "
            "from structured query analysis using few-shot examples and schema context."
        ),
        instruction=PromptLibrary.SQL_GENERATOR_AGENT,
        tools=[
            get_similar_examples,
            get_table_details,
            find_related_tables,
            get_sample_values,
            search_schema_by_keyword,
        ],
    )
