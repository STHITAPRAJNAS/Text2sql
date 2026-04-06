"""
PRISM SQL Generator Agent
Phase I of the PRISM pipeline — generates SQL from structured query analysis.

Uses the query analysis + schema context + few-shot examples + semantic memory
to generate production-quality, dialect-aware SQL queries.
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
from agents.tools.memory_tools import store_successful_query, get_memory_context
from agents.tools.indexing_tools import search_relevant_tables
from agents.tools.learning_tools import (
    classify_query_skills,
    find_correction_examples,
    get_proven_join_paths,
    get_effective_confidence_threshold,
    find_known_clarification,
)


def create_sql_generator_agent() -> Agent:
    """
    Create the SQL Generator Agent.

    This agent:
    1. Calls LoadMemoryTool (ADK built-in) to retrieve semantically similar
       past queries as dynamic few-shot context (self-improvement)
    2. Calls get_similar_examples for curated examples from the few-shot store
    3. Receives the structured QueryAnalysis from the Deep Think Analyzer
    4. Generates production-quality SQL following dialect-specific best practices
    5. Stores high-confidence results via store_successful_query for future learning

    SQL Generation Principles:
    - Use CTEs for complex multi-step queries
    - Explicit column lists (no SELECT *)
    - Proper NULL handling with COALESCE / TRY_CAST (Spark SQL)
    - Appropriate LIMIT clauses
    - Dialect-correct syntax: Spark SQL, PostgreSQL, MySQL, SQLite

    Spark SQL extras (Databricks): QUALIFY, PIVOT, VARIANT, TRY_CAST,
    cross-catalog JOINs, named WINDOW, ILIKE, TRY_DIVIDE, higher-order
    functions (TRANSFORM, FILTER, AGGREGATE), STRUCT fields.

    Uses the Pro model for maximum accuracy.
    """
    settings = get_settings()

    # Core tools
    tools = [
        # ── Progressive learning tools (consult before generating SQL) ──
        # 1. Classify skills → writes to session state for other tools
        classify_query_skills,
        # 2. Retrieve past corrections (avoid known mistakes)
        find_correction_examples,
        # 3. Retrieve proven join paths for current skill type
        get_proven_join_paths,
        # 4. Check if a clarification is already known (skip asking user)
        find_known_clarification,
        # 5. Get calibrated confidence threshold for this skill type
        get_effective_confidence_threshold,
        # ── Few-shot & memory retrieval ─────────────────────────────────
        get_similar_examples,
        get_memory_context,
        # ── Schema context tools ────────────────────────────────────────
        search_relevant_tables,
        get_table_details,
        find_related_tables,
        get_sample_values,
        search_schema_by_keyword,
        # ── Post-generation storage ─────────────────────────────────────
        store_successful_query,
    ]

    # Add ADK LoadMemoryTool for cross-session semantic memory search
    try:
        from google.adk.tools.load_memory_tool import LoadMemoryTool
        tools.insert(0, LoadMemoryTool())
    except ImportError:
        pass  # ADK not installed or memory tool unavailable

    return Agent(
        name="sql_generator_agent",
        model=settings.llm.sql_generator_model,
        description=(
            "Expert SQL generator that produces production-quality, optimized SQL queries "
            "from structured query analysis using few-shot examples, semantic memory, and schema context."
        ),
        instruction=PromptLibrary.SQL_GENERATOR_AGENT,
        tools=tools,
    )
