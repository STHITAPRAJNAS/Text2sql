"""
Tests for feedback loop tools and active learning.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.tools.feedback_tools import (
    _feedback_store,
    get_feedback_stats,
    record_feedback,
)


class TestRecordFeedback:

    def setup_method(self):
        """Clear feedback store before each test."""
        _feedback_store.clear()

    def test_records_entry(self):
        result = record_feedback(
            query="show top customers",
            sql="SELECT * FROM customers LIMIT 10",
            rating=4.0,
        )
        assert result["status"] == "recorded"
        assert "feedback_id" in result
        assert len(_feedback_store) == 1

    def test_feedback_id_unique(self):
        r1 = record_feedback("q1", "SELECT 1", 3.0)
        r2 = record_feedback("q2", "SELECT 2", 3.0)
        assert r1["feedback_id"] != r2["feedback_id"]

    def test_high_rating_triggers_fewshot_action(self):
        with patch("agents.tools.feedback_tools._add_to_fewshot_store", return_value=True) as mock_fs:
            result = record_feedback("top customers", "SELECT id FROM customers", 5.0)
        assert "added_to_fewshot" in result["actions_taken"]
        assert mock_fs.called

    def test_low_rating_no_fewshot(self):
        with patch("agents.tools.feedback_tools._add_to_fewshot_store") as mock_fs:
            result = record_feedback("top customers", "SELECT id FROM customers", 2.0)
        assert "added_to_fewshot" not in result["actions_taken"]
        assert not mock_fs.called

    def test_rating_1_logged_for_review(self):
        result = record_feedback("bad query", "SELECT WRONG", 1.0)
        assert "logged_for_review" in result["actions_taken"]

    def test_correction_added_as_high_value(self):
        with patch("agents.tools.feedback_tools._add_to_fewshot_store", return_value=True) as mock_fs:
            result = record_feedback(
                query="top revenue customers",
                sql="SELECT * FROM customers",
                rating=2.0,
                corrected_sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            )
        assert "correction_added_to_fewshot" in result["actions_taken"]

    def test_no_correction_no_correction_action(self):
        with patch("agents.tools.feedback_tools._add_to_fewshot_store", return_value=True):
            result = record_feedback("top customers", "SELECT 1", 5.0, corrected_sql=None)
        assert "correction_added_to_fewshot" not in result["actions_taken"]

    def test_memory_action_on_high_rating(self):
        with patch("agents.tools.feedback_tools._add_to_fewshot_store", return_value=True):
            with patch("agents.tools.feedback_tools._add_to_memory_store", return_value=True) as mock_mem:
                result = record_feedback("q", "SELECT 1", 4.0)
        assert "added_to_memory" in result["actions_taken"]


class TestGetFeedbackStats:

    def setup_method(self):
        _feedback_store.clear()

    def test_empty_store(self):
        stats = get_feedback_stats()
        assert stats["total"] == 0
        assert stats["avg_rating"] == 0.0

    def test_calculates_avg_rating(self):
        record_feedback("q1", "s1", 4.0)
        record_feedback("q2", "s2", 2.0)
        stats = get_feedback_stats()
        assert stats["total"] == 2
        assert stats["avg_rating"] == 3.0

    def test_positive_negative_counts(self):
        record_feedback("q1", "s1", 5.0)
        record_feedback("q2", "s2", 1.0)
        record_feedback("q3", "s3", 2.0)
        stats = get_feedback_stats()
        assert stats["positive"] == 1  # rating >= 4
        assert stats["negative"] == 2  # rating <= 2

    def test_corrections_counted(self):
        record_feedback("q1", "s1", 2.0, corrected_sql="SELECT correct")
        stats = get_feedback_stats()
        assert stats["corrections_received"] == 1
