"""
Tests for SQL Validation Tools
Tests the four validation layers: syntax, schema, security, performance.
"""
from __future__ import annotations

import pytest

from agents.tools.validation_tools import (
    check_performance_safety,
    check_schema_compliance,
    check_sql_security,
    validate_sql_syntax,
)


class TestValidateSQLSyntax:

    def test_valid_simple_select(self):
        sql = "SELECT customer_id, name FROM customers WHERE status = 'active'"
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_valid_cte(self):
        sql = """
        WITH revenue AS (
            SELECT customer_id, SUM(amount) AS total
            FROM orders
            GROUP BY customer_id
        )
        SELECT c.name, r.total
        FROM customers c
        JOIN revenue r ON c.id = r.customer_id
        ORDER BY r.total DESC
        LIMIT 10
        """
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is True

    def test_empty_sql(self):
        result = validate_sql_syntax("")
        assert result["is_valid"] is False
        assert any("Empty" in e["message"] for e in result["errors"])

    def test_unbalanced_parens(self):
        sql = "SELECT (name FROM customers"
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is False
        assert any("parentheses" in e["message"].lower() for e in result["errors"])

    def test_ddl_blocked(self):
        sql = "DROP TABLE customers"
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is False

    def test_dml_blocked(self):
        sql = "DELETE FROM customers WHERE id = 1"
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is False

    def test_multiple_statements_blocked(self):
        sql = "SELECT 1; DROP TABLE customers"
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is False
        assert any("Multiple" in e["message"] for e in result["errors"])

    def test_select_star_warning(self):
        sql = "SELECT * FROM customers"
        result = validate_sql_syntax(sql)
        assert result["is_valid"] is True  # Not an error, just a warning
        assert any("SELECT *" in w["message"] for w in result["warnings"])

    def test_aggregate_without_group_by_warning(self):
        sql = "SELECT customer_id, SUM(amount) FROM orders"
        result = validate_sql_syntax(sql)
        assert any("GROUP BY" in w["message"] for w in result["warnings"])


class TestCheckSchemCompliance:

    def test_valid_table_reference(self):
        result = check_schema_compliance(
            sql="SELECT name FROM customers",
            schema_context="",
            tables_in_schema=["customers", "orders", "products"],
        )
        assert result["is_compliant"] is True
        assert result["missing_tables"] == []

    def test_missing_table(self):
        result = check_schema_compliance(
            sql="SELECT name FROM nonexistent_table",
            schema_context="",
            tables_in_schema=["customers", "orders"],
        )
        assert result["is_compliant"] is False
        assert "nonexistent_table" in result["missing_tables"]

    def test_cte_not_flagged_as_missing(self):
        sql = """
        WITH cte AS (SELECT id FROM customers)
        SELECT id FROM cte
        """
        result = check_schema_compliance(
            sql=sql,
            schema_context="",
            tables_in_schema=["customers"],
        )
        # CTE 'cte' should not be flagged as missing from the schema
        assert "cte" not in result["missing_tables"]

    def test_implicit_join_warning(self):
        sql = "SELECT a.name, b.amount FROM customers a, orders b"
        result = check_schema_compliance(
            sql=sql,
            schema_context="",
            tables_in_schema=["customers", "orders"],
        )
        assert any("implicit" in w["message"].lower() for w in result["warnings"])


class TestCheckSQLSecurity:

    def test_safe_select(self):
        sql = "SELECT name, email FROM customers WHERE status = 'active' LIMIT 100"
        result = check_sql_security(sql)
        assert result["is_safe"] is True
        assert result["risk_level"] == "LOW"

    def test_drop_table_blocked(self):
        result = check_sql_security("DROP TABLE customers")
        assert result["is_safe"] is False
        assert result["risk_level"] == "HIGH"
        assert any(v["type"] == "DDL_STATEMENT" for v in result["violations"])

    def test_insert_blocked(self):
        result = check_sql_security("INSERT INTO customers VALUES (1, 'test')")
        assert result["is_safe"] is False
        assert any(v["type"] == "DML_STATEMENT" for v in result["violations"])

    def test_update_blocked(self):
        result = check_sql_security("UPDATE customers SET status = 'inactive'")
        assert result["is_safe"] is False

    def test_delete_blocked(self):
        result = check_sql_security("DELETE FROM orders WHERE id = 1")
        assert result["is_safe"] is False

    def test_dangerous_function_blocked(self):
        result = check_sql_security("SELECT pg_read_file('/etc/passwd')")
        assert result["is_safe"] is False
        assert any(v["type"] == "DANGEROUS_FUNCTION" for v in result["violations"])

    def test_system_table_blocked(self):
        result = check_sql_security("SELECT * FROM pg_shadow")
        assert result["is_safe"] is False
        assert any(v["type"] == "SYSTEM_TABLE_ACCESS" for v in result["violations"])

    def test_create_blocked(self):
        result = check_sql_security("CREATE TABLE evil (id INT)")
        assert result["is_safe"] is False


class TestCheckPerformanceSafety:

    def test_safe_query_with_limit(self):
        sql = "SELECT name FROM customers WHERE status = 'active' LIMIT 100"
        result = check_performance_safety(sql)
        assert result["risk_level"] in ("LOW", "MEDIUM")
        assert result["has_limit"] is True
        assert result["has_where"] is True

    def test_missing_limit_warning(self):
        sql = "SELECT name, email FROM customers"
        result = check_performance_safety(sql)
        assert any(w["type"] == "MISSING_LIMIT" for w in result["warnings"])

    def test_aggregation_no_limit_required(self):
        sql = "SELECT COUNT(*) AS total FROM customers"
        result = check_performance_safety(sql)
        # Aggregation with COUNT doesn't need LIMIT
        assert not any(w["type"] == "MISSING_LIMIT" for w in result["warnings"])

    def test_function_on_column_warning(self):
        sql = "SELECT name FROM customers WHERE UPPER(email) = 'TEST@EXAMPLE.COM'"
        result = check_performance_safety(sql)
        assert any(w["type"] == "FUNCTION_ON_INDEXED_COLUMN" for w in result["warnings"])

    def test_large_table_full_scan(self):
        sql = "SELECT name FROM large_table"
        result = check_performance_safety(
            sql, table_row_counts={"large_table": 5_000_000}
        )
        assert result["risk_level"] == "HIGH"
