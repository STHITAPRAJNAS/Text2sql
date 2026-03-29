"""
PRISM Query Optimizer Agent
Phase S (Synthesis) of the PRISM pipeline — performance optimization.

Applies database-specific query optimization strategies:
- Predicate pushdown and partition filter enforcement
- Join reordering for Spark SQL broadcast hints
- CTE vs subquery decisions
- Index-aware rewriting
- Query cost estimation and scan size warnings
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.database_tools import get_query_explain, get_table_row_count
from agents.tools.schema_tools import get_table_details, search_schema_by_keyword
from agents.tools.validation_tools import check_performance_safety
from agents.tools.cost_tools import estimate_query_cost, check_partition_coverage


def create_query_optimizer_agent() -> Agent:
    """
    Create the Query Optimizer Agent.

    Optimization strategies:
    1. Structural: predicate pushdown, join reordering, subquery elimination
    2. Index-aware: avoid functions on indexed columns, use covering indexes
    3. Result set: apply LIMIT, approximate functions for large data
    4. Plan analysis: identify sequential scans and suggest fixes
    5. Cost estimation: warn if query scans > N GB, suggest partition filters
    6. Spark SQL: add BROADCAST hints for small tables, ZORDER recommendations

    The optimizer explains every change applied for transparency.
    """
    settings = get_settings()

    return Agent(
        name="query_optimizer_agent",
        model=settings.llm.optimizer_model,
        description=(
            "SQL performance optimizer that applies database-specific optimization strategies "
            "including predicate pushdown, join reordering, cost estimation, and partition filter "
            "recommendations for Databricks/Spark SQL."
        ),
        instruction=PromptLibrary.QUERY_OPTIMIZER_AGENT,
        tools=[
            estimate_query_cost,
            check_partition_coverage,
            get_query_explain,
            get_table_row_count,
            get_table_details,
            search_schema_by_keyword,
            check_performance_safety,
        ],
    )
