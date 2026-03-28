"""
PRISM Query Optimizer Agent
Phase S (Synthesis) of the PRISM pipeline — performance optimization.

Applies database-specific query optimization strategies:
- Predicate pushdown
- Join reordering
- CTE vs subquery decisions
- Index-aware rewriting
- Aggregation optimization
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.database_tools import get_query_explain, get_table_row_count
from agents.tools.schema_tools import get_table_details, search_schema_by_keyword
from agents.tools.validation_tools import check_performance_safety


def create_query_optimizer_agent() -> Agent:
    """
    Create the Query Optimizer Agent.

    Optimization strategies:
    1. Structural: predicate pushdown, join reordering, subquery elimination
    2. Index-aware: avoid functions on indexed columns, use covering indexes
    3. Result set: apply LIMIT, approximate functions for large data
    4. Plan analysis: identify sequential scans and suggest fixes

    The optimizer explains every change applied for transparency.
    """
    settings = get_settings()

    return Agent(
        name="query_optimizer_agent",
        model=settings.llm.optimizer_model,
        description=(
            "SQL performance optimizer that applies database-specific optimization strategies "
            "including predicate pushdown, join reordering, and index-aware query rewriting."
        ),
        instruction=PromptLibrary.QUERY_OPTIMIZER_AGENT,
        tools=[
            get_query_explain,
            get_table_row_count,
            get_table_details,
            search_schema_by_keyword,
            check_performance_safety,
        ],
    )
