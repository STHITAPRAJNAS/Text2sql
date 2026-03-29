"""
PRISM Deep Think Query Analyzer & Schema Linker Agents
Phase R of the PRISM pipeline — the reasoning core using chain-of-thought.

The Deep Think approach:
1. Decompose the query into atomic components
2. Extract business entities with NER-like analysis
3. Assess ambiguities and request clarification when needed
4. Score confidence and flag complex cases
5. Produce a structured QueryAnalysis for downstream agents
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.schema_tools import (
    search_schema_by_keyword,
    get_table_details,
    find_related_tables,
    get_sample_values,
)
from agents.tools.clarification_tools import request_clarification
from agents.tools.indexing_tools import search_relevant_tables


def create_deep_think_query_analyzer() -> Agent:
    """
    Create the Deep Think Query Analyzer Agent.

    This is the reasoning core of the PRISM system. It uses step-by-step
    chain-of-thought analysis to deeply understand the user's intent before
    any SQL is generated.

    Deep Think Process:
    1. Query decomposition
    2. Entity extraction
    3. Ambiguity resolution — calls request_clarification if unresolvable
    4. Complexity assessment
    5. Mental execution plan
    6. Confidence scoring

    If confidence < deep_think_confidence_threshold, the agent MUST call
    request_clarification before proceeding.

    Uses the most capable model (Pro) for maximum reasoning depth.
    """
    settings = get_settings()

    return Agent(
        name="deep_think_query_analyzer",
        model=settings.llm.query_analyzer_model,
        description=(
            "Deep Think reasoning agent that uses chain-of-thought analysis to deeply understand "
            "natural language queries, extract entities, resolve ambiguities, and either produce a "
            "structured analysis for SQL generation or request clarification for ambiguous queries."
        ),
        instruction=PromptLibrary.DEEP_THINK_QUERY_ANALYZER,
        tools=[
            search_relevant_tables,
            search_schema_by_keyword,
            get_table_details,
            request_clarification,
        ],
    )


def create_schema_linker_agent() -> Agent:
    """
    Create the Schema Linker Agent.

    Maps extracted natural language entities to exact schema elements
    (table.column) with confidence scoring. Handles:
    - Exact name matches
    - Alias/synonym matches from business glossary
    - Semantic matches via schema search
    - Multi-column entity resolution (e.g., "full name" → first_name + last_name)

    Produces precise EntityMapping objects consumed by the SQL Generator.
    """
    settings = get_settings()

    return Agent(
        name="schema_linker_agent",
        model=settings.llm.query_analyzer_model,
        description=(
            "Schema linker that precisely maps natural language entities from the query analysis "
            "to exact database table.column references with confidence scores."
        ),
        instruction=PromptLibrary.SCHEMA_LINKER_AGENT,
        tools=[
            search_relevant_tables,
            search_schema_by_keyword,
            get_table_details,
            find_related_tables,
            get_sample_values,
        ],
    )
