"""
API Request/Response Models
Pydantic models for the FastAPI REST interface.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for a text-to-SQL query."""
    query: str = Field(
        description="Natural language question to convert to SQL",
        min_length=3,
        max_length=2000,
        examples=["What are the top 10 customers by revenue this month?"],
    )
    database_name: str = Field(
        default="default",
        description="Target database name (for multi-database setups)",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn conversations",
    )
    max_rows: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum rows to return in the result",
    )
    execute_query: bool = Field(
        default=True,
        description="Whether to execute the SQL or just generate it",
    )
    explain_results: bool = Field(
        default=True,
        description="Whether to include a natural language explanation",
    )


class QueryResponse(BaseModel):
    """Response body for a text-to-SQL query."""
    success: bool
    session_id: str
    query: str

    # SQL output
    generated_sql: str | None = None
    optimized_sql: str | None = None

    # Results (when execute_query=True)
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    # Natural language answer
    answer: str | None = None

    # Pipeline metadata
    confidence: float = 0.0
    pipeline_stages: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0

    # Error handling
    error: str | None = None
    suggestions: list[str] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    """Response for schema discovery endpoint."""
    database_name: str
    dialect: str
    tables: dict[str, Any]
    relationships: list[dict[str, str]]
    total_tables: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
    database_connected: bool
    agents_ready: bool


class ExampleRequest(BaseModel):
    """Request to add a few-shot example."""
    question: str = Field(min_length=3)
    sql: str = Field(min_length=5)
    database_name: str = "default"
    tags: list[str] = Field(default_factory=list)
    feedback_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ExampleResponse(BaseModel):
    """Response for adding a few-shot example."""
    success: bool
    example_id: str | None = None
    total_examples: int = 0
    error: str | None = None
