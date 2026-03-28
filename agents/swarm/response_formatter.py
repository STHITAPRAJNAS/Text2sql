"""
PRISM Response Formatter Agent
Phase M (Monitoring) of the PRISM pipeline — query execution and result formatting.

Executes the final SQL, formats results appropriately, generates a natural
language summary, and assembles the complete QueryResponse.
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.database_tools import execute_sql_query


def create_response_formatter_agent() -> Agent:
    """
    Create the Response Formatter Agent.

    Responsibilities:
    1. Execute the validated, optimized SQL query
    2. Handle execution errors gracefully
    3. Format results based on query type (table, single value, time series)
    4. Generate a natural language summary leading with the key finding
    5. Include metadata: execution time, row count, SQL, confidence
    6. Flag data quality issues (nulls, potential outliers)

    The response format adapts to the query intent:
    - Aggregation → highlight the metric with context
    - Comparison → emphasize differences
    - Time series → show trend direction
    - Listing → clean tabular format
    """
    settings = get_settings()

    return Agent(
        name="response_formatter_agent",
        model=settings.llm.formatter_model,
        description=(
            "Response formatter that executes the final SQL query and presents results "
            "as a clear natural language answer with supporting data and metadata."
        ),
        instruction=PromptLibrary.RESPONSE_FORMATTER_AGENT,
        tools=[
            execute_sql_query,
        ],
    )
