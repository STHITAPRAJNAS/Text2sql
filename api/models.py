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
    needs_clarification: bool = False
    clarification_question: str | None = None
    clarification_options: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    cache_source: str | None = None
    cost_warning: str | None = None
    pii_report: dict[str, Any] | None = None
    anomalies: list[str] = Field(default_factory=list)


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

# ------------------------------------------------------------------ #
# Feedback                                                             #
# ------------------------------------------------------------------ #

class FeedbackRequest(BaseModel):
    query: str = Field(min_length=3)
    sql: str = Field(min_length=5)
    rating: float = Field(ge=1.0, le=5.0, description="Rating from 1 (bad) to 5 (excellent)")
    database_name: str = "default"
    session_id: str | None = None
    comment: str | None = None
    corrected_sql: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    actions_taken: list[str] = Field(default_factory=list)
    status: str = "recorded"


# ------------------------------------------------------------------ #
# Cache                                                                #
# ------------------------------------------------------------------ #

class CacheStatsResponse(BaseModel):
    l1_backend: str = "redis"
    l2_backend: str = "chromadb"
    l2_size: int = 0
    l1_ttl_seconds: int = 300
    l2_ttl_seconds: int = 3600
    l2_similarity_threshold: float = 0.92


# ------------------------------------------------------------------ #
# Glossary                                                             #
# ------------------------------------------------------------------ #

class GlossaryTerm(BaseModel):
    term: str
    description: str = ""
    table_name: str = ""
    column_name: str = ""
    filter_sql: str = ""
    example_sql: str = ""
    created_at: float | None = None
    updated_at: float | None = None


class GlossaryUpsertRequest(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    description: str = ""
    table_name: str = ""
    column_name: str = ""
    filter_sql: str = ""
    example_sql: str = ""


# ------------------------------------------------------------------ #
# Query History                                                        #
# ------------------------------------------------------------------ #

class HistoryItem(BaseModel):
    query_id: str
    nl_query: str
    generated_sql: str | None = None
    success: bool = False
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0
    row_count: int = 0
    cache_hit: bool = False
    cache_source: str | None = None
    pii_detected: bool = False
    cost_warning: str | None = None
    needs_clarification: bool = False
    created_at: float = 0.0


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


# ------------------------------------------------------------------ #
# Dead Letter Queue                                                    #
# ------------------------------------------------------------------ #

class DeadLetterItem(BaseModel):
    id: str
    query_id: str
    nl_query: str
    generated_sql: str | None = None
    error: str = ""
    failure_reason: str = ""
    database_name: str = "default"
    confidence: float = 0.0
    reviewed: bool = False
    corrected_sql: str | None = None
    created_at: float = 0.0


class DeadLetterResponse(BaseModel):
    items: list[DeadLetterItem]
    total: int


class ResolveDeadLetterRequest(BaseModel):
    corrected_sql: str = Field(min_length=5)

