"""
PRISM SQL Validator Agent
Phase S (Synthesis) of the PRISM pipeline — multi-layer SQL validation.

Performs five validation layers:
1. Complexity budget (pre-flight — blocks ADVANCED queries if configured)
2. Syntax validation (balanced parens, proper keywords)
3. Schema compliance (tables/columns exist)
4. Security validation (no DDL/DML, no injection)
5. Performance safety (no full scans without LIMIT on large tables)

Can auto-correct fixable issues and returns to the generator for unfixable ones.
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.validation_tools import (
    validate_sql_syntax,
    check_schema_compliance,
    check_sql_security,
    check_performance_safety,
    check_query_complexity,
)
from agents.tools.schema_tools import get_database_schema
from agents.tools.loop_tools import exit_validation_loop


def create_sql_validator_agent() -> Agent:
    """
    Create the SQL Validator Agent.

    Validation layers:
    0. Complexity budget: enforce join/subquery/window function limits
    1. Syntax: balanced parens, proper keywords, SQL grammar
    2. Schema: all tables/columns exist, types are compatible
    3. Security: no DDL/DML, no dangerous functions, no injection patterns
    4. Performance: LIMIT on large tables, no full scans, no Cartesian products

    If validation fails but the error is fixable, the agent corrects the SQL.
    If the error is unfixable, it signals the orchestrator for regeneration.
    """
    settings = get_settings()

    return Agent(
        name="sql_validator_agent",
        model=settings.llm.sql_validator_model,
        description=(
            "Multi-layer SQL validation specialist that checks complexity budget, syntax, "
            "schema compliance, security constraints, and performance safety."
        ),
        instruction=PromptLibrary.SQL_VALIDATOR_AGENT,
        tools=[
            check_query_complexity,    # Layer 0: complexity budget
            validate_sql_syntax,       # Layer 1: syntax
            check_schema_compliance,   # Layer 2: schema
            check_sql_security,        # Layer 3: security
            check_performance_safety,  # Layer 4: performance
            get_database_schema,
            exit_validation_loop,      # Signals LoopAgent to stop when all checks pass
        ],
    )
