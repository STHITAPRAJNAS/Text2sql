"""
Tests for query cost estimation tools.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.tools.cost_tools import (
    _generate_suggestions,
    check_partition_coverage,
    estimate_query_cost,
)


class TestGenerateSuggestions:

    def test_suggests_partition_filter(self):
        suggestions = _generate_suggestions(
            "SELECT * FROM orders",
            "Scan: orders",
            has_partition_filter=False,
        )
        assert any("partition" in s.lower() for s in suggestions)

    def test_suggests_limit_when_missing(self):
        suggestions = _generate_suggestions(
            "SELECT id FROM orders",
            "Scan: orders",
            has_partition_filter=True,
        )
        assert any("LIMIT" in s for s in suggestions)

    def test_no_limit_suggestion_when_limit_present(self):
        suggestions = _generate_suggestions(
            "SELECT id FROM orders LIMIT 100",
            "Scan: orders",
            has_partition_filter=True,
        )
        assert not any("LIMIT" in s for s in suggestions)

    def test_suggests_select_star_replacement(self):
        suggestions = _generate_suggestions(
            "SELECT * FROM orders LIMIT 10",
            "Scan",
            has_partition_filter=True,
        )
        assert any("SELECT *" in s for s in suggestions)

    def test_cartesian_product_warning(self):
        suggestions = _generate_suggestions(
            "SELECT * FROM a, b",
            "BroadcastNestedLoop",
            has_partition_filter=True,
        )
        assert any("Cartesian" in s for s in suggestions)


class TestEstimateQueryCost:

    def test_returns_structure_without_databricks(self):
        with patch("core.databricks.get_databricks_connector", return_value=None):
            with patch("agents.tools.database_tools.get_query_explain", return_value="Seq Scan on orders"):
                result = estimate_query_cost("SELECT * FROM orders")
        assert "estimated_bytes" in result
        assert "cost_warning" in result
        assert "suggestions" in result
        assert "explain_plan" in result

    def test_no_error_without_any_backend(self):
        with patch("core.databricks.get_databricks_connector", side_effect=ImportError):
            with patch("agents.tools.database_tools.get_query_explain", side_effect=Exception("no db")):
                result = estimate_query_cost("SELECT 1")
        assert result["estimated_bytes"] == -1
        assert result["cost_warning"] is False

    def test_seq_scan_no_error(self):
        """estimate_query_cost should not raise even when all backends fail."""
        with patch("core.databricks.get_databricks_connector", side_effect=Exception("no db")):
            result = estimate_query_cost("SELECT id FROM orders")
        assert result["estimated_bytes"] == -1


class TestCheckPartitionCoverage:

    def test_returns_structure(self):
        result = check_partition_coverage("SELECT * FROM orders", "main.sales.orders")
        assert "table_id" in result
        assert "partition_columns" in result
        assert "filter_detected" in result

    def test_detects_partition_filter_in_where(self):
        with patch("core.schema_index.get_schema_index") as mock_idx:
            mock_idx.return_value.get_by_id.return_value = {
                "partitioning": '["order_date", "region"]'
            }
            result = check_partition_coverage(
                "SELECT id FROM orders WHERE order_date = '2024-01-01'",
                "main.sales.orders",
            )
        assert result["filter_detected"] is True
        assert result["recommendation"] is None

    def test_no_filter_generates_recommendation(self):
        with patch("core.schema_index.get_schema_index") as mock_idx:
            mock_idx.return_value.get_by_id.return_value = {
                "partitioning": '["order_date"]'
            }
            result = check_partition_coverage(
                "SELECT id FROM orders",
                "main.sales.orders",
            )
        assert result["filter_detected"] is False
        assert result["recommendation"] is not None
        assert "order_date" in result["recommendation"]

    def test_no_partitioning_no_recommendation(self):
        with patch("core.schema_index.get_schema_index") as mock_idx:
            mock_idx.return_value.get_by_id.return_value = {
                "partitioning": "[]"
            }
            result = check_partition_coverage(
                "SELECT id FROM orders",
                "main.sales.orders",
            )
        assert result["recommendation"] is None
