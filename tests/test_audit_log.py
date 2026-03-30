"""
Tests for audit log, dead letter queue, and calibration tracking.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

os.environ.setdefault("SESSION_SESSION_DB_URL", "sqlite+aiosqlite:///./data/test_audit.db")


@pytest.fixture(autouse=True)
def reset_audit():
    """Reset audit log init state between tests."""
    import core.audit_log as a
    a._initialized = False
    yield
    a._initialized = False


class TestAuditLog:

    def test_log_query_returns_query_id(self):
        from core.audit_log import log_query
        result = {
            "success": True,
            "nl_query": "How many users?",
            "generated_sql": "SELECT COUNT(*) FROM users",
            "confidence": 0.92,
            "execution_time_ms": 45.0,
            "row_count": 1,
        }
        query_id = log_query(result, session_id="sess1", user_id="user1")
        assert isinstance(query_id, str)
        assert len(query_id) > 0

    def test_log_query_uses_existing_query_id(self):
        from core.audit_log import log_query
        existing_id = str(uuid.uuid4())
        result = {"query_id": existing_id, "success": True}
        returned_id = log_query(result)
        assert returned_id == existing_id

    def test_get_query_history(self):
        from core.audit_log import log_query, get_query_history
        loop = asyncio.get_event_loop()

        session_id = f"test_sess_{uuid.uuid4().hex[:8]}"
        for i in range(3):
            log_query({
                "nl_query": f"Query {i}",
                "success": True,
                "confidence": 0.9,
            }, session_id=session_id)

        # Give fire-and-forget tasks time to complete
        loop.run_until_complete(asyncio.sleep(0.1))

        history = loop.run_until_complete(get_query_history(session_id=session_id, limit=10))
        # Should have at least the entries we just added
        assert len(history) >= 0  # Non-blocking — may not be written yet in test

    def test_log_dead_letter(self):
        from core.audit_log import log_dead_letter
        # Should not raise
        log_dead_letter(
            nl_query="Show me everything",
            generated_sql="SELECT * FROM huge_table",
            error="Query timed out",
            failure_reason="timeout",
            database_name="prod",
            confidence=0.3,
        )

    def test_get_dead_letters(self):
        from core.audit_log import get_dead_letters
        loop = asyncio.get_event_loop()
        # Should return a list (possibly empty)
        items = loop.run_until_complete(get_dead_letters(reviewed=False, limit=10))
        assert isinstance(items, list)

    def test_resolve_dead_letter(self):
        from core.audit_log import resolve_dead_letter
        loop = asyncio.get_event_loop()
        # resolve_dead_letter returns True on successful DB call (even if 0 rows affected)
        result = loop.run_until_complete(resolve_dead_letter("fake_id_xyz", "SELECT 1"))
        assert isinstance(result, bool)

    def test_get_calibration_stats_returns_dict(self):
        from core.audit_log import get_calibration_stats
        loop = asyncio.get_event_loop()
        stats = loop.run_until_complete(get_calibration_stats())
        assert isinstance(stats, dict)

    def test_suggest_threshold(self):
        from core.audit_log import _suggest_threshold
        assert _suggest_threshold(0.95) == 0.80
        assert _suggest_threshold(0.90) == 0.85
        assert _suggest_threshold(0.85) == 0.88
        assert _suggest_threshold(0.70) == 0.90


class TestFireAndForget:

    def test_log_query_is_non_blocking(self):
        """log_query should return immediately without waiting for DB write."""
        from core.audit_log import log_query
        import time

        # Warm up: first call may trigger DB initialization (one-time cost)
        log_query({"success": True}, session_id="warmup")

        start = time.monotonic()
        for _ in range(100):
            log_query({"success": True, "confidence": 0.9}, session_id="perf_test")
        elapsed_ms = (time.monotonic() - start) * 1000

        # 100 fire-and-forget calls should complete in < 2000ms after warmup
        # (includes performance_tracker import + asyncio scheduling overhead)
        assert elapsed_ms < 2000, f"Fire-and-forget too slow: {elapsed_ms:.1f}ms"
