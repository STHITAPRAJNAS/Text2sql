"""
Query Context Models
Pydantic models representing the PRISM pipeline state at each phase.
These are passed between agents as structured data.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"        # Single table, no aggregation
    MODERATE = "MODERATE"    # 2-3 tables, basic aggregation
    COMPLEX = "COMPLEX"      # 4+ tables, window functions, CTEs
    ADVANCED = "ADVANCED"    # Recursive, pivots, analytical functions


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"        # ≥ 0.85
    MEDIUM = "MEDIUM"    # 0.60 - 0.84
    LOW = "LOW"          # < 0.60


class EntityMapping(BaseModel):
    """Maps a natural language entity to a database schema element."""
    entity: str = Field(description="Natural language entity")
    table: str = Field(description="Target database table")
    column: str | None = Field(default=None, description="Target column (if applicable)")
    transformation: str | None = Field(default=None, description="Required SQL transformation")
    confidence: float = Field(ge=0.0, le=1.0, description="Mapping confidence")
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


class AggregationSpec(BaseModel):
    """Specification for an aggregation in the query."""
    function: str = Field(description="Aggregate function (COUNT, SUM, AVG, MAX, MIN)")
    column: str = Field(description="Column to aggregate")
    alias: str | None = None
    filter_condition: str | None = None  # For FILTER (WHERE ...) syntax


class FilterCondition(BaseModel):
    """A WHERE or HAVING clause condition."""
    column: str
    operator: str  # =, >, <, >=, <=, LIKE, IN, BETWEEN, IS NULL
    value: Any
    logical_op: str = "AND"  # AND, OR


class TimeRange(BaseModel):
    """Time range specification for temporal queries."""
    start_date: str | None = None
    end_date: str | None = None
    relative_period: str | None = None  # "last 30 days", "this month", "YTD"
    date_column: str | None = None
    date_granularity: str | None = None  # day, week, month, quarter, year


class QueryAnalysis(BaseModel):
    """
    Structured output of the Deep Think Query Analyzer.
    Represents complete understanding of the user's query intent.
    """
    # Original query
    original_query: str = Field(description="The user's original natural language query")

    # Intent analysis
    primary_intent: str = Field(description="Primary query intent")
    secondary_intents: list[str] = Field(default_factory=list)

    # Entity extraction
    entities: list[str] = Field(description="Extracted business entities")
    entity_mappings: list[EntityMapping] = Field(default_factory=list)

    # Query structure
    required_tables: list[str] = Field(description="Tables needed for the query")
    joins_needed: list[dict[str, str]] = Field(default_factory=list)
    aggregations: list[AggregationSpec] = Field(default_factory=list)
    filters: list[FilterCondition] = Field(default_factory=list)
    time_range: TimeRange | None = None
    group_by_columns: list[str] = Field(default_factory=list)
    order_by: list[dict[str, str]] = Field(default_factory=list)
    limit: int | None = None

    # Complexity and confidence
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    # Chain of thought
    reasoning_steps: list[str] = Field(default_factory=list, description="CoT steps")
    execution_plan: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class ValidationError(BaseModel):
    """A single SQL validation error."""
    layer: str  # syntax, schema, security, performance
    severity: str  # ERROR, WARNING
    message: str
    location: str | None = None  # SQL snippet where error occurred
    suggestion: str | None = None


class ValidationResult(BaseModel):
    """Result of SQL validation by the PRISM validator agent."""
    is_valid: bool
    original_sql: str
    corrected_sql: str | None = None
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)
    fix_applied: bool = False
    fix_description: str | None = None
    validation_time_ms: float = 0.0


class OptimizationResult(BaseModel):
    """Result of SQL optimization."""
    original_sql: str
    optimized_sql: str
    optimizations_applied: list[str] = Field(default_factory=list)
    estimated_improvement: str | None = None
    index_suggestions: list[str] = Field(default_factory=list)
    explain_plan: str | None = None


class QueryResponse(BaseModel):
    """
    Final PRISM pipeline response returned to the user.
    Contains the SQL, results, explanation, and full pipeline metadata.
    """
    # Results
    answer: str = Field(description="Natural language answer/summary")
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    # SQL
    generated_sql: str = Field(description="The final SQL query used")
    optimized_sql: str | None = None

    # Pipeline metadata
    query_analysis: QueryAnalysis | None = None
    validation_result: ValidationResult | None = None
    optimization_result: OptimizationResult | None = None

    # Performance
    execution_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0

    # Confidence
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH

    # Errors
    error: str | None = None
    suggestions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)


class QueryContext(BaseModel):
    """
    Mutable context object passed through the PRISM pipeline.
    Each agent reads from and writes to this context.
    """
    # Input
    user_query: str
    session_id: str = ""
    database_name: str = "default"
    sql_dialect: str = "postgresql"

    # History (for multi-turn conversations)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # Pipeline state
    schema_context: str | None = None           # Phase P output
    metadata_context: str | None = None          # Phase P output (metadata)
    query_analysis: QueryAnalysis | None = None  # Phase R output
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)
    generated_sql: str | None = None             # Phase I output
    validation_result: ValidationResult | None = None  # Phase S output
    optimized_sql: str | None = None             # Phase S output

    # Iteration control (Deep Think)
    current_iteration: int = 0
    max_iterations: int = 3
    last_error: str | None = None

    # Final response
    response: QueryResponse | None = None

    @property
    def final_sql(self) -> str | None:
        """Return the best available SQL (optimized > generated)."""
        return self.optimized_sql or self.generated_sql

    def add_to_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    model_config = ConfigDict(arbitrary_types_allowed=True)
