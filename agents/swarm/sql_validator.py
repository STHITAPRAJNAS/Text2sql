"""
PRISM SQL Validator Agent
Phase S (Synthesis) of the PRISM pipeline — multi-layer SQL validation.

Performs four validation layers:
1. Syntax validation (balanced parens, proper keywords)
2. Schema compliance (tables/columns exist)
3. Security validation (no DDL/DML, no injection)
4. Performance safety (no full scans without LIMIT on large tables)

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
)
from agents.tools.schema_tools import get_database_schema


def create_sql_validator_agent() -> Agent:
    """
    Create the SQL Validator Agent.

    Validation layers:
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
            "Multi-layer SQL validation specialist that checks syntax, schema compliance, "
            "security constraints, and performance safety of generated SQL queries."
        ),
        instruction=PromptLibrary.SQL_VALIDATOR_AGENT,
        tools=[
            validate_sql_syntax,
            check_schema_compliance,
            check_sql_security,
            check_performance_safety,
            get_database_schema,
        ],
    )
