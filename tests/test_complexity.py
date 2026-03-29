"""
Tests for query complexity budget checker.
"""
from __future__ import annotations

import pytest

from agents.tools.validation_tools import check_query_complexity


class TestComplexityScoring:

    def test_simple_select(self):
        sql = "SELECT id, name FROM users WHERE status = 'active'"
        result = check_query_complexity(sql)
        assert result["complexity_score"] <= 3
        assert result["level"] == "SIMPLE"
        assert result["blocked"] is False

    def test_single_join_moderate(self):
        sql = """
        SELECT u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.created_at > '2024-01-01'
        """
        result = check_query_complexity(sql)
        # 1 JOIN = score 1
        assert result["breakdown"]["join_count"] == 1
        assert result["complexity_score"] == 1
        assert result["level"] == "SIMPLE"

    def test_multiple_joins(self):
        sql = """
        SELECT u.name, o.id, oi.product_id, p.name
        FROM users u
        JOIN orders o ON u.id = o.user_id
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        JOIN categories c ON p.category_id = c.id
        """
        result = check_query_complexity(sql)
        assert result["breakdown"]["join_count"] == 4  # 4 JOIN keywords
        # 4 JOINs = score 4, MODERATE
        assert result["level"] == "MODERATE"

    def test_cross_join_high_score(self):
        sql = "SELECT a.x, b.y FROM table_a a CROSS JOIN table_b b"
        result = check_query_complexity(sql)
        assert result["breakdown"]["cross_join_count"] == 1
        # CROSS JOIN = +4, also counted as regular JOIN (+1) = 5
        assert result["complexity_score"] >= 4

    def test_subqueries_add_score(self):
        sql = """
        SELECT *
        FROM (SELECT user_id, COUNT(*) AS cnt FROM orders GROUP BY user_id) sub
        WHERE cnt > 5
        """
        result = check_query_complexity(sql)
        assert result["breakdown"]["subquery_count"] >= 1
        assert result["complexity_score"] >= 2

    def test_window_function(self):
        sql = """
        SELECT name, revenue,
               ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) AS rank
        FROM sales
        """
        result = check_query_complexity(sql)
        assert result["breakdown"]["window_count"] == 1

    def test_cte_counted(self):
        sql = """
        WITH monthly AS (
            SELECT MONTH(created_at) AS mo, SUM(amount) AS total
            FROM orders
            GROUP BY mo
        )
        SELECT * FROM monthly ORDER BY total DESC
        """
        result = check_query_complexity(sql)
        assert result["breakdown"]["cte_count"] >= 1

    def test_union_counted(self):
        sql = """
        SELECT id, name FROM customers_us
        UNION
        SELECT id, name FROM customers_eu
        """
        result = check_query_complexity(sql)
        assert result["breakdown"]["set_op_count"] == 1

    def test_advanced_query_not_blocked_by_default(self):
        """By default block_advanced_queries=False, so ADVANCED queries aren't blocked."""
        # Build a query with score > 15
        sql = """
        WITH cte1 AS (SELECT * FROM t1),
             cte2 AS (SELECT * FROM t2)
        SELECT a.x, b.y, c.z, d.w,
               ROW_NUMBER() OVER (PARTITION BY a.x ORDER BY b.y) AS r1,
               RANK() OVER (ORDER BY c.z) AS r2,
               DENSE_RANK() OVER (PARTITION BY d.w ORDER BY a.x) AS r3
        FROM cte1 a
        JOIN t3 b ON a.id = b.id
        JOIN t4 c ON b.id = c.id
        JOIN t5 d ON c.id = d.id
        JOIN t6 e ON d.id = e.id
        JOIN t7 f ON e.id = f.id
        JOIN (SELECT user_id, SUM(total) as s FROM orders GROUP BY user_id) sub1 ON a.id = sub1.user_id
        JOIN (SELECT item_id, MAX(price) as p FROM items GROUP BY item_id) sub2 ON b.id = sub2.item_id
        """
        result = check_query_complexity(sql)
        assert result["level"] == "ADVANCED" or result["complexity_score"] > 8
        # Not blocked by default
        assert result["blocked"] is False

    def test_breakdown_fields_present(self):
        sql = "SELECT 1"
        result = check_query_complexity(sql)
        assert "complexity_score" in result
        assert "level" in result
        assert "breakdown" in result
        assert "blocked" in result
        assert "block_reason" in result
        assert "suggestions" in result
        breakdown = result["breakdown"]
        assert "join_count" in breakdown
        assert "cross_join_count" in breakdown
        assert "subquery_count" in breakdown
        assert "window_count" in breakdown
        assert "cte_count" in breakdown
        assert "set_op_count" in breakdown

    def test_empty_sql(self):
        result = check_query_complexity("")
        assert result["complexity_score"] == 0
        assert result["level"] == "SIMPLE"

    def test_cross_join_suggestion(self):
        sql = "SELECT * FROM a CROSS JOIN b"
        result = check_query_complexity(sql)
        assert any("CROSS JOIN" in s for s in result["suggestions"])

    def test_many_subqueries_suggestion(self):
        # Use (SELECT without whitespace so the counter detects them
        sql = (
            "SELECT * FROM "
            "(SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT 1) s4) s3) s2) s1"
        )
        result = check_query_complexity(sql)
        assert result["breakdown"]["subquery_count"] >= 4
        assert any("subquer" in s.lower() for s in result["suggestions"])
