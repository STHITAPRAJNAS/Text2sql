"""
API Request/Response Models
Pydantic models for all Text2SQL PRISM REST endpoints.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Query                                                                #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=2000,
        examples=["What are the top 10 customers by revenue this month?"],
    )
    database_name: str = Field(default="default")
    session_id: str | None = None
    max_rows: int = Field(default=100, ge=1, le=1000)
    execute_query: bool = Field(default=True)
    explain_results: bool = Field(default=True)


class QueryResponse(BaseModel):
    success: bool
    session_id: str
    query: str
    generated_sql: str | None = None
    optimized_sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    answer: str | None = None
    confidence: float = 0.0
    pipeline_stages: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0
    error: str | None = None
    suggestions: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ #
# Schema                                                               #
# ------------------------------------------------------------------ #

class SchemaResponse(BaseModel):
    database_name: str
    dialect: str
    tables: dict[str, Any]
    relationships: list[dict[str, str]]
    total_tables: int


# ------------------------------------------------------------------ #
# Schema Index                                                         #
# ------------------------------------------------------------------ #

class IndexRequest(BaseModel):
    """Request to bulk-index a Unity Catalog schema."""
    catalog: str = Field(description="Unity Catalog catalog name, e.g. 'main'")
    schema: str = Field(description="Schema name, e.g. 'sales'")
    max_tables: int = Field(default=500, ge=1, le=5000)


class IndexResponse(BaseModel):
    """Response from bulk schema indexing."""
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)
    total_in_schema: int = 0
    processed: int = 0
    truncated: bool = False


class SchemaIndexListResponse(BaseModel):
    """Response listing tables in the schema index."""
    tables: list[dict[str, str]]
    total_indexed: int
    catalog_filter: str | None = None


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    database_connected: bool
    agents_ready: bool
    schema_index_size: int = 0


# ------------------------------------------------------------------ #
# Examples                                                             #
# ------------------------------------------------------------------ #

class ExampleRequest(BaseModel):
    question: str = Field(min_length=3)
    sql: str = Field(min_length=5)
    database_name: str = "default"
    tags: list[str] = Field(default_factory=list)
    feedback_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ExampleResponse(BaseModel):
    success: bool
    example_id: str | None = None
    total_examples: int = 0
    error: str | None = None
