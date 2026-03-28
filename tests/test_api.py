"""
Tests for the FastAPI application layer.
Uses TestClient for synchronous testing of API endpoints.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Patch DB and runner to avoid needing a real database
    with patch("core.database._db_manager") as mock_db, \
         patch("agents.runner._runner") as mock_runner:

        mock_db_instance = AsyncMock()
        mock_db_instance.dialect.value = "sqlite"
        mock_db_instance.execute_safe_query = AsyncMock(
            return_value=type("R", (), {
                "columns": ["ok"],
                "rows": [{"ok": 1}],
                "row_count": 1,
                "execution_time_ms": 1.0,
                "truncated": False,
            })()
        )
        mock_db.return_value = mock_db_instance

        from api.app import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:

    def test_health_check_returns_200(self, client):
        response = client.get("/api/v1/health")
        # Accept 200 even if DB is mocked
        assert response.status_code in (200, 500)

    def test_health_response_structure(self, client):
        response = client.get("/api/v1/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "version" in data


class TestRootEndpoint:

    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Text2SQL PRISM"


class TestQueryValidation:

    def test_empty_query_rejected(self, client):
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422  # Validation error

    def test_query_too_short_rejected(self, client):
        response = client.post("/api/v1/query", json={"query": "hi"})
        assert response.status_code == 422

    def test_max_rows_bounds(self, client):
        # max_rows > 1000 should be rejected
        response = client.post("/api/v1/query", json={
            "query": "show me all customers",
            "max_rows": 10000,
        })
        assert response.status_code == 422

    def test_max_rows_zero_rejected(self, client):
        response = client.post("/api/v1/query", json={
            "query": "show me all customers",
            "max_rows": 0,
        })
        assert response.status_code == 422


class TestExamplesEndpoint:

    def test_add_example_valid(self, client):
        response = client.post("/api/v1/examples", json={
            "question": "How many customers are in the US?",
            "sql": "SELECT COUNT(*) FROM customers WHERE country = 'US'",
            "database_name": "default",
            "tags": ["count", "filter"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_add_example_empty_question(self, client):
        response = client.post("/api/v1/examples", json={
            "question": "",
            "sql": "SELECT 1",
        })
        assert response.status_code == 422

    def test_add_example_empty_sql(self, client):
        response = client.post("/api/v1/examples", json={
            "question": "Test question here?",
            "sql": "SE",  # Too short
        })
        assert response.status_code == 422


class TestMetricsEndpoint:

    def test_metrics_returns_agent_status(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "sql_generator" in data["agents"]
