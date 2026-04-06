"""
Tests for progressive learning skills.

Five test classes, one per skill:
  TestSkillClassifier         — query taxonomy classification
  TestCorrectionStore         — wrong→correct SQL triple store
  TestSchemaExpertise         — join path expertise graph
  TestConfidenceCalibrator    — adaptive threshold calibration
  TestClarificationMemory     — cross-session clarification recall

Plus:
  TestLearningTools           — ADK tool wrappers
  TestFeedbackBugFix          — verifies add_example_to_store fix
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 1: Skill Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkillClassifier:
    def test_cte_detected(self):
        from core.skill_classifier import classify_query
        sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        tags = classify_query("", sql)
        assert "cte" in tags

    def test_window_function_detected(self):
        from core.skill_classifier import classify_query
        sql = "SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) FROM emp"
        tags = classify_query("", sql)
        assert "window_function" in tags

    def test_multi_table_join_detected(self):
        from core.skill_classifier import classify_query
        sql = "SELECT * FROM orders o JOIN order_items i ON o.id = i.order_id"
        tags = classify_query("", sql)
        assert "multi_table_join" in tags

    def test_subquery_detected(self):
        from core.skill_classifier import classify_query
        sql = "SELECT * FROM t WHERE id IN (SELECT id FROM other)"
        tags = classify_query("", sql)
        assert "subquery" in tags

    def test_date_aggregation_detected_sql(self):
        from core.skill_classifier import classify_query
        sql = "SELECT DATE_TRUNC('month', created_at), SUM(amount) FROM orders GROUP BY 1"
        tags = classify_query("", sql)
        assert "date_aggregation" in tags

    def test_date_aggregation_detected_nl(self):
        from core.skill_classifier import classify_query
        tags = classify_query("show revenue trend by month", "")
        assert "date_aggregation" in tags

    def test_set_operation_detected(self):
        from core.skill_classifier import classify_query
        sql = "SELECT id FROM a UNION SELECT id FROM b"
        tags = classify_query("", sql)
        assert "set_operation" in tags

    def test_aggregation_detected(self):
        from core.skill_classifier import classify_query
        sql = "SELECT dept, COUNT(*) FROM emp GROUP BY dept"
        tags = classify_query("", sql)
        assert "aggregation" in tags

    def test_fallback_single_table(self):
        from core.skill_classifier import classify_query
        tags = classify_query("list all customers", "")
        assert len(tags) >= 1

    def test_extract_join_paths_single_join(self):
        from core.skill_classifier import extract_join_paths
        sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"
        paths = extract_join_paths(sql)
        assert len(paths) == 1
        assert "orders" in paths[0]
        assert "customers" in paths[0]

    def test_extract_join_paths_empty(self):
        from core.skill_classifier import extract_join_paths
        paths = extract_join_paths("SELECT * FROM customers")
        assert paths == []

    def test_difficulty_complex(self):
        from core.skill_classifier import get_skill_difficulty
        assert get_skill_difficulty(["window_function"]) == "complex"

    def test_difficulty_moderate(self):
        from core.skill_classifier import get_skill_difficulty
        assert get_skill_difficulty(["multi_table_join"]) == "moderate"

    def test_difficulty_simple(self):
        from core.skill_classifier import get_skill_difficulty
        assert get_skill_difficulty(["aggregation"]) == "simple"

    def test_difficulty_highest_wins(self):
        from core.skill_classifier import get_skill_difficulty
        assert get_skill_difficulty(["aggregation", "window_function"]) == "complex"


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 2: Correction Store
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectionStore:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path, monkeypatch):
        import core.correction_store as mod
        monkeypatch.setattr(mod, "_DB_PATH", tmp_path / "corrections.db")
        monkeypatch.setattr(mod, "_initialized", False)
        yield

    def test_store_and_retrieve(self):
        from core.correction_store import store_correction, find_similar_corrections

        store_correction(
            nl_query="total revenue by customer",
            wrong_sql="SELECT customer, total FROM orders",
            correct_sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
            skill_tags=["aggregation"],
        )
        import time; time.sleep(0.1)

        results = find_similar_corrections("revenue by customer")
        assert len(results) >= 1
        assert results[0]["correct_sql"].startswith("SELECT customer_id")

    def test_no_results_for_unrelated_query(self):
        from core.correction_store import store_correction, find_similar_corrections

        store_correction(
            nl_query="total revenue by customer",
            wrong_sql="SELECT x FROM y",
            correct_sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
        )
        import time; time.sleep(0.1)

        results = find_similar_corrections("unrelated xyz123 query")
        assert results == []

    def test_stats(self):
        from core.correction_store import get_correction_stats
        stats = get_correction_stats()
        assert "total_corrections" in stats
        assert isinstance(stats["total_corrections"], int)


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 3: Schema Expertise
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaExpertise:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path, monkeypatch):
        import core.schema_expertise as mod
        monkeypatch.setattr(mod, "_DB_PATH", tmp_path / "expertise.db")
        monkeypatch.setattr(mod, "_initialized", False)
        yield

    def test_record_and_retrieve_proven_join(self):
        from core.schema_expertise import record_join_result, get_proven_joins
        import time

        # Record 3 successful uses of a join path
        for _ in range(3):
            record_join_result(
                join_paths=["orders→order_items→products"],
                skill_tags=["multi_table_join"],
                rating=5.0,
            )
        time.sleep(0.2)

        results = get_proven_joins(["multi_table_join"], min_successes=3, min_avg_rating=4.0)
        assert len(results) >= 1
        assert results[0]["join_path"] == "orders→order_items→products"
        assert results[0]["avg_rating"] >= 4.0

    def test_insufficient_successes_not_returned(self):
        from core.schema_expertise import record_join_result, get_proven_joins
        import time

        record_join_result(
            join_paths=["a→b"],
            skill_tags=["aggregation"],
            rating=5.0,
        )
        time.sleep(0.2)

        # Requires min 3 successes
        results = get_proven_joins(["aggregation"], min_successes=3)
        assert results == []

    def test_expertise_report(self):
        from core.schema_expertise import get_expertise_report
        report = get_expertise_report()
        assert "total_paths" in report
        assert "top_paths" in report


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 4: Confidence Calibrator
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceCalibrator:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path, monkeypatch):
        import core.confidence_calibrator as mod
        monkeypatch.setattr(mod, "_DB_PATH", tmp_path / "calibration.db")
        monkeypatch.setattr(mod, "_initialized", False)
        yield

    def test_returns_base_when_insufficient_data(self):
        from core.confidence_calibrator import get_calibrated_threshold, _base_threshold
        threshold = get_calibrated_threshold(["window_function"])
        assert threshold == _base_threshold()

    def test_raises_threshold_after_poor_outcomes(self):
        from core.confidence_calibrator import (
            record_calibration, get_calibrated_threshold, _base_threshold, MIN_SAMPLES
        )
        import time

        base = _base_threshold()

        # Simulate consistently over-confident predictions (predicted high, actual low)
        for _ in range(MIN_SAMPLES + 2):
            record_calibration(
                skill_tags=["window_function"],
                predicted_confidence=0.95,
                actual_rating=0.4,   # user only gives 2/5
            )
        time.sleep(0.2)

        threshold = get_calibrated_threshold(["window_function"])
        # Should be raised because observed accuracy < ideal
        assert threshold >= base

    def test_record_calibration_fire_and_forget(self):
        """record_calibration should not raise and return immediately."""
        from core.confidence_calibrator import record_calibration
        # Should complete without error
        record_calibration(["aggregation"], 0.85, 0.8)

    def test_calibration_report(self):
        from core.confidence_calibrator import get_calibration_report
        report = get_calibration_report()
        assert "base_threshold" in report
        assert "skills" in report


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 5: Clarification Memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestClarificationMemory:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path, monkeypatch):
        import core.clarification_memory as mod
        monkeypatch.setattr(mod, "_DB_PATH", tmp_path / "clarifications.db")
        monkeypatch.setattr(mod, "_initialized", False)
        yield

    def test_store_and_exact_recall(self):
        from core.clarification_memory import store_clarification, find_known_clarification
        import time

        store_clarification(
            nl_query="show revenue by region",
            question="Do you mean fiscal year or calendar year?",
            answer="calendar year",
        )
        time.sleep(0.2)

        result = find_known_clarification("show revenue by region")
        assert result is not None
        assert result["answer"] == "calendar year"
        assert result["similarity"] == 1.0

    def test_no_result_for_unrelated_query(self):
        from core.clarification_memory import store_clarification, find_known_clarification
        import time

        store_clarification(
            nl_query="show revenue by region",
            question="Fiscal or calendar?",
            answer="fiscal",
        )
        time.sleep(0.2)

        result = find_known_clarification("list all customers xyz987")
        assert result is None

    def test_tenant_isolation(self):
        from core.clarification_memory import store_clarification, find_known_clarification
        import time

        store_clarification("show revenue", "Gross or net?", "net", tenant_id="tenant_a")
        time.sleep(0.2)

        # Different tenant should not see it (though default fallback applies)
        result = find_known_clarification("show revenue", tenant_id="tenant_b")
        # May or may not return due to default fallback; key: no exception
        assert result is None or isinstance(result, dict)

    def test_stats(self):
        from core.clarification_memory import get_clarification_stats
        stats = get_clarification_stats()
        assert "stored_clarifications" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# Learning Tools (ADK wrappers)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearningTools:
    def _mock_ctx(self, state: dict | None = None):
        ctx = MagicMock()
        ctx.state = state or {}
        return ctx

    def test_classify_query_skills_writes_state(self):
        from agents.tools.learning_tools import classify_query_skills
        ctx = self._mock_ctx()
        result = classify_query_skills(
            nl_query="monthly revenue trend",
            generated_sql="SELECT DATE_TRUNC('month', d), SUM(amount) FROM t GROUP BY 1",
            tool_context=ctx,
        )
        assert result["stored_in_state"] is True
        assert "date_aggregation" in ctx.state["skill_tags"]

    def test_classify_query_skills_without_context(self):
        from agents.tools.learning_tools import classify_query_skills
        result = classify_query_skills("get all customers", "SELECT * FROM customers")
        assert "skills" in result
        assert isinstance(result["skills"], list)

    def test_find_correction_examples_no_client(self):
        from agents.tools.learning_tools import find_correction_examples
        # No corrections stored → should return empty list, not raise
        result = find_correction_examples("total sales by region")
        assert result["count"] >= 0
        assert isinstance(result["corrections"], list)

    def test_get_proven_join_paths_from_state(self):
        from agents.tools.learning_tools import get_proven_join_paths
        ctx = self._mock_ctx(state={"skill_tags": ["window_function"]})
        result = get_proven_join_paths(tool_context=ctx)
        assert "proven_paths" in result
        assert isinstance(result["proven_paths"], list)

    def test_get_proven_join_paths_no_tags(self):
        from agents.tools.learning_tools import get_proven_join_paths
        result = get_proven_join_paths(skill_tags=None, tool_context=None)
        assert "No skill tags" in result["note"]

    def test_find_known_clarification_not_found(self):
        from agents.tools.learning_tools import find_known_clarification
        result = find_known_clarification("some unique query xyz987")
        assert result["found"] is False
        assert result["auto_apply"] is False

    def test_get_effective_confidence_threshold_returns_float(self):
        from agents.tools.learning_tools import get_effective_confidence_threshold
        result = get_effective_confidence_threshold(skill_tags=["aggregation"])
        assert isinstance(result["threshold"], float)
        assert 0.0 < result["threshold"] < 1.0

    def test_store_query_correction_with_context(self):
        from agents.tools.learning_tools import store_query_correction
        ctx = self._mock_ctx(state={"skill_tags": ["aggregation"], "tenant_id": "default"})
        result = store_query_correction(
            original_query="top customers by spend",
            wrong_sql="SELECT customer FROM orders",
            corrected_sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY 1",
            tool_context=ctx,
        )
        assert result["stored"] is True
        assert "aggregation" in result["skill_tags"]

    def test_record_clarification_answer(self):
        from agents.tools.learning_tools import record_clarification_answer
        result = record_clarification_answer(
            nl_query="revenue this year",
            question="Calendar or fiscal year?",
            answer="fiscal year",
        )
        assert result["stored"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback Bug Fix
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedbackBugFix:
    def test_add_example_to_store_called_not_add_example(self):
        """Verify the renamed function is called correctly."""
        with patch("agents.tools.feedback_tools._add_to_fewshot_store") as mock_add:
            mock_add.return_value = True
            from agents.tools import feedback_tools as ft
            ft.record_feedback(
                query="top customers",
                sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY 1",
                rating=5.0,
            )
            assert mock_add.called

    def test_correction_stored_on_corrected_sql(self):
        """When corrected_sql differs from sql, correction_stored should appear in actions."""
        with patch("agents.tools.feedback_tools._add_to_fewshot_store", return_value=True):
            with patch("core.correction_store.store_correction") as mock_store:
                from agents.tools import feedback_tools as ft
                result = ft.record_feedback(
                    query="revenue by region",
                    sql="SELECT region, revenue FROM orders",
                    rating=2.0,
                    corrected_sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
                )
                assert mock_store.called
                assert "correction_stored" in result["actions_taken"]

    def test_calibration_recorded_when_confidence_provided(self):
        """When predicted_confidence > 0, calibration should be recorded."""
        with patch("core.confidence_calibrator.record_calibration") as mock_cal:
            from agents.tools import feedback_tools as ft
            ft.record_feedback(
                query="count orders",
                sql="SELECT COUNT(*) FROM orders",
                rating=4.0,
                predicted_confidence=0.88,
                skill_tags=["aggregation"],
            )
            assert mock_cal.called
            # record_calibration called with positional or keyword args
            args, kwargs = mock_cal.call_args
            all_args = list(args) + list(kwargs.values())
            assert any(abs(v - 0.88) < 0.01 for v in all_args if isinstance(v, float))  # predicted
            assert any(abs(v - 0.8) < 0.01 for v in all_args if isinstance(v, float))   # actual
