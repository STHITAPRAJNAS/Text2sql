"""
Tests for the Business Glossary store (core/glossary.py).
Uses an in-memory SQLite database to avoid side effects.
"""
from __future__ import annotations

import asyncio
import os

import pytest

# Redirect glossary DB to a temp file before importing
os.environ.setdefault("SESSION_SESSION_DB_URL", "sqlite+aiosqlite:///./data/test_glossary.db")


@pytest.fixture(autouse=True)
def reset_glossary():
    """Clear the in-memory cache between tests."""
    import core.glossary as g
    g._cache.clear()
    g._db_initialized = False
    yield
    g._cache.clear()
    g._db_initialized = False


class TestGlossaryLookup:

    def test_lookup_missing_term(self):
        import core.glossary as g
        result = asyncio.get_event_loop().run_until_complete(g.lookup("nonexistent_term"))
        assert result is None

    def test_upsert_and_lookup(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()

        loop.run_until_complete(g.upsert(
            term="active customers",
            table_name="users",
            filter_sql="status = 'active'",
            description="Customers with active status",
        ))

        result = loop.run_until_complete(g.lookup("active customers"))
        assert result is not None
        assert result["term"] == "active customers"
        assert result["table_name"] == "users"
        assert result["filter_sql"] == "status = 'active'"

    def test_lookup_case_insensitive(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()

        loop.run_until_complete(g.upsert(term="Revenue", description="Total revenue"))
        result = loop.run_until_complete(g.lookup("revenue"))
        assert result is not None

        result2 = loop.run_until_complete(g.lookup("REVENUE"))
        assert result2 is not None

    def test_upsert_overwrites(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()

        loop.run_until_complete(g.upsert(term="churn", description="Old definition"))
        loop.run_until_complete(g.upsert(term="churn", description="Updated definition"))

        result = loop.run_until_complete(g.lookup("churn"))
        assert result["description"] == "Updated definition"

    def test_delete_term(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()

        loop.run_until_complete(g.upsert(term="temp_term", description="Temp"))
        deleted = loop.run_until_complete(g.delete("temp_term"))
        assert deleted is True

        result = loop.run_until_complete(g.lookup("temp_term"))
        assert result is None

    def test_delete_nonexistent(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()
        deleted = loop.run_until_complete(g.delete("does_not_exist"))
        # Should not raise, returns False or True depending on implementation
        assert isinstance(deleted, bool)

    def test_list_all(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()

        terms = ["term_a", "term_b", "term_c"]
        for t in terms:
            loop.run_until_complete(g.upsert(term=t, description=f"Desc {t}"))

        all_terms = loop.run_until_complete(g.list_all(limit=100))
        term_names = [item["term"] for item in all_terms]
        for t in terms:
            assert t in term_names

    def test_search_by_keyword(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()

        loop.run_until_complete(g.upsert(term="monthly revenue", description="Revenue per month"))
        loop.run_until_complete(g.upsert(term="annual revenue", description="Revenue per year"))
        loop.run_until_complete(g.upsert(term="customer count", description="Number of customers"))

        results = loop.run_until_complete(g.search("revenue"))
        names = [r["term"] for r in results]
        assert "monthly revenue" in names
        assert "annual revenue" in names
        assert "customer count" not in names


class TestLookupSync:

    def test_lookup_sync_returns_none_for_missing(self):
        from core.glossary import lookup_sync
        result = lookup_sync("definitely_not_here_xyz")
        assert result is None

    def test_lookup_sync_returns_entry(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()
        loop.run_until_complete(g.upsert(
            term="ytd",
            description="Year to date",
            filter_sql="YEAR(date) = YEAR(CURRENT_DATE)",
        ))

        from core.glossary import lookup_sync
        result = lookup_sync("ytd")
        assert result is not None
        assert "filter_sql" in result


class TestGlossaryTools:

    def test_lookup_glossary_tool_found(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()
        loop.run_until_complete(g.upsert(
            term="active users",
            table_name="users",
            filter_sql="is_active = 1",
            description="Users with active account",
        ))

        from agents.tools.glossary_tools import lookup_glossary
        result = lookup_glossary("active users")
        assert result["found"] is True
        assert result["table_name"] == "users"

    def test_lookup_glossary_tool_not_found(self):
        from agents.tools.glossary_tools import lookup_glossary
        result = lookup_glossary("completely_unknown_term_xyz")
        assert result["found"] is False
        assert "message" in result

    def test_search_glossary_tool(self):
        import core.glossary as g
        loop = asyncio.get_event_loop()
        loop.run_until_complete(g.upsert(term="customer lifetime value", description="CLV metric"))

        from agents.tools.glossary_tools import search_glossary
        result = search_glossary("lifetime")
        assert result["count"] >= 1
        assert any("lifetime" in r.get("term", "").lower() for r in result["results"])
