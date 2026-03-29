"""
Query Cost Estimation Tools
============================
Estimates the cost (data scanned, compute) of a SQL query before execution
using EXPLAIN / DESCRIBE QUERY PLAN.

For Databricks:
  - Uses EXPLAIN (with COST option where supported) to get bytes scanned
  - Checks for missing partition filters on large partitioned tables
  - Suggests OPTIMIZE hints and Z-ORDER if applicable

For other databases:
  - PostgreSQL: EXPLAIN (FORMAT JSON, ANALYZE FALSE) — total_cost
  - SQLite: EXPLAIN QUERY PLAN — detects full scans

If estimated bytes > cost_warn_gb threshold (default 10 GB), the optimizer
adds a warning and suggests partition filters to the generated SQL.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# 1 GB in bytes
_GB = 1024 ** 3


def estimate_query_cost(
    sql: str,
    table_id: str | None = None,
) -> dict[str, Any]:
    """
    Estimate the cost of a SQL query using EXPLAIN.

    Args:
        sql: The SQL query to analyze
        table_id: Optional primary table for partition filter check

    Returns:
        {
            "estimated_bytes": int,       # -1 if unknown
            "estimated_gb": float,
            "cost_warning": bool,
            "warning_message": str | None,
            "suggestions": list[str],
            "explain_plan": str,
            "has_partition_filter": bool,
        }
    """
    from config.settings import get_settings
    settings = get_settings()
    warn_gb = settings.feedback.cost_warn_gb

    result: dict[str, Any] = {
        "estimated_bytes": -1,
        "estimated_gb": -1.0,
        "cost_warning": False,
        "warning_message": None,
        "suggestions": [],
        "explain_plan": "",
        "has_partition_filter": False,
    }

    # Try Databricks EXPLAIN
    try:
        from core.databricks import get_databricks_connector
        connector = get_databricks_connector()
        if connector is not None:
            _estimate_databricks(sql, result, connector, warn_gb, table_id)
            return result
    except Exception as exc:
        logger.debug("Databricks cost estimation failed", error=str(exc))

    # Try generic EXPLAIN via SQLAlchemy
    try:
        from agents.tools.database_tools import get_query_explain
        plan = get_query_explain(sql=sql)
        result["explain_plan"] = str(plan)[:2000]
        _parse_generic_explain(result, str(plan), warn_gb)
    except Exception as exc:
        logger.debug("Generic EXPLAIN failed", error=str(exc))

    return result


def _estimate_databricks(
    sql: str,
    result: dict[str, Any],
    connector: Any,
    warn_gb: float,
    table_id: str | None,
) -> None:
    """Run Databricks EXPLAIN and parse cost information."""
    try:
        explain_result = connector.execute_query(f"EXPLAIN {sql}", max_rows=100)
        plan_text = "\n".join(
            str(list(row.values())[0]) for row in explain_result.rows if row
        )
        result["explain_plan"] = plan_text[:3000]

        # Parse bytes scanned from Spark plan
        bytes_match = re.search(r"Statistics.*?(\d[\d,]*)\s*bytes", plan_text, re.IGNORECASE)
        if bytes_match:
            bytes_str = bytes_match.group(1).replace(",", "")
            estimated_bytes = int(bytes_str)
            result["estimated_bytes"] = estimated_bytes
            result["estimated_gb"] = round(estimated_bytes / _GB, 2)

        # Check for partition filters
        result["has_partition_filter"] = bool(
            re.search(r"PartitionFilters|partition_filter", plan_text, re.IGNORECASE)
        )

        # Cost warning
        if result["estimated_bytes"] > 0:
            gb = result["estimated_gb"]
            if gb > warn_gb:
                result["cost_warning"] = True
                result["warning_message"] = (
                    f"Query will scan approximately {gb:.1f} GB of data "
                    f"(threshold: {warn_gb} GB)."
                )
                result["suggestions"].extend(
                    _generate_suggestions(sql, plan_text, result["has_partition_filter"])
                )

        # Check for large table scans without partition filters
        if table_id and not result["has_partition_filter"]:
            result["suggestions"].append(
                f"Consider adding a partition filter on {table_id} to reduce data scanned."
            )

    except Exception as exc:
        logger.debug("Databricks EXPLAIN parsing failed", error=str(exc))


def _parse_generic_explain(
    result: dict[str, Any],
    plan_text: str,
    warn_gb: float,
) -> None:
    """Parse PostgreSQL EXPLAIN output for cost estimation."""
    # PostgreSQL JSON format: "Total Cost": 12345.67
    cost_match = re.search(r'"Total Cost":\s*([\d.]+)', plan_text)
    if cost_match:
        total_cost = float(cost_match.group(1))
        # PostgreSQL cost units are in 8KB pages; rough conversion
        estimated_bytes = int(total_cost * 8192)
        result["estimated_bytes"] = estimated_bytes
        result["estimated_gb"] = round(estimated_bytes / _GB, 2)

    # Detect full sequential scans
    if re.search(r"Seq Scan|SCAN TABLE", plan_text, re.IGNORECASE):
        result["suggestions"].append(
            "Full table scan detected. Consider adding a WHERE clause or index."
        )

    if result["estimated_gb"] > warn_gb:
        result["cost_warning"] = True
        result["warning_message"] = (
            f"Query estimated to scan {result['estimated_gb']:.1f} GB "
            f"(threshold: {warn_gb} GB)."
        )


def _generate_suggestions(
    sql: str,
    plan_text: str,
    has_partition_filter: bool,
) -> list[str]:
    """Generate optimization suggestions based on the explain plan."""
    suggestions = []

    if not has_partition_filter:
        suggestions.append(
            "Add a partition filter (e.g., WHERE year = 2024) to significantly reduce data scanned."
        )

    if re.search(r"BroadcastNestedLoop|CartesianProduct", plan_text, re.IGNORECASE):
        suggestions.append(
            "Cartesian product detected. Verify all JOIN conditions are present."
        )

    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        suggestions.append(
            "Add a LIMIT clause if only a sample of results is needed."
        )

    if re.search(r"SELECT\s+\*", sql, re.IGNORECASE):
        suggestions.append(
            "Replace SELECT * with explicit column names to reduce data transfer."
        )

    # Suggest TABLESAMPLE for large exploratory queries
    if re.search(r"COUNT\(\*\)|AVG\(|SUM\(", sql, re.IGNORECASE):
        suggestions.append(
            "For approximate results, consider TABLESAMPLE or approx_count_distinct()."
        )

    return suggestions


def check_partition_coverage(
    sql: str,
    table_id: str,
) -> dict[str, Any]:
    """
    Check whether a query has adequate partition filter coverage for a table.

    Returns information about the partition columns and whether they're
    referenced in the WHERE clause.
    """
    result: dict[str, Any] = {
        "table_id": table_id,
        "partition_columns": [],
        "filter_detected": False,
        "recommendation": None,
    }

    try:
        from core.schema_index import get_schema_index
        import json
        index = get_schema_index()
        metadata = index.get_by_id(table_id)
        if metadata:
            partitioning = metadata.get("partitioning", "[]")
            if isinstance(partitioning, str):
                try:
                    partitioning = json.loads(partitioning)
                except Exception:
                    partitioning = []
            result["partition_columns"] = partitioning
            if partitioning:
                # Check if any partition column appears in the WHERE clause
                where_match = re.search(
                    r'\bWHERE\b(.+?)(?:\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)',
                    sql,
                    re.IGNORECASE | re.DOTALL,
                )
                where_clause = where_match.group(1) if where_match else ""
                result["filter_detected"] = any(
                    col.lower() in where_clause.lower()
                    for col in partitioning
                )
                if not result["filter_detected"]:
                    result["recommendation"] = (
                        f"Table {table_id} is partitioned by {partitioning}. "
                        f"Adding a filter on these columns will dramatically reduce scan cost."
                    )
    except Exception as exc:
        logger.debug("Partition coverage check failed", error=str(exc))

    return result
