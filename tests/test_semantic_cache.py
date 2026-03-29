"""
Tests for the two-tier semantic query result cache.
Tests L1 (Redis mock) and L2 (ChromaDB in-memory mock) separately and together.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.semantic_cache import (
    _cache_key,
    _normalize_query,
    cache_result,
    get_cached_result,
    get_l2_cache,
    set_l2_cache,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_result(sql: str = "SELECT 1", success: bool = True) -> dict:
    return {
        "success": success,
        "generated_sql": sql,
        "answer": "42 rows",
        "confidence": 0.9,
        "rows": [],
        "columns": [],
    }


# ------------------------------------------------------------------ #
# _normalize_query                                                     #
# ------------------------------------------------------------------ #

class TestNormalizeQuery:

    def test_lowercases(self):
        assert _normalize_query("TOP Customers") == "top customers"

    def test_collapses_whitespace(self):
        assert _normalize_query("  top  10  customers  ") == "top 10 customers"

    def test_idempotent(self):
        q = "show revenue by region"
        assert _normalize_query(_normalize_query(q)) == _normalize_query(q)


# ------------------------------------------------------------------ #
# _cache_key                                                           #
# ------------------------------------------------------------------ #

class TestCacheKey:

    def test_same_query_same_key(self):
        k1 = _cache_key("top 10 customers", "main")
        k2 = _cache_key("top 10 customers", "main")
        assert k1 == k2

    def test_different_query_different_key(self):
        k1 = _cache_key("top 10 customers", "main")
        k2 = _cache_key("top 5 products", "main")
        assert k1 != k2

    def test_different_database_different_key(self):
        k1 = _cache_key("top 10 customers", "main")
        k2 = _cache_key("top 10 customers", "other_db")
        assert k1 != k2

    def test_case_insensitive(self):
        k1 = _cache_key("TOP 10 CUSTOMERS", "main")
        k2 = _cache_key("top 10 customers", "main")
        assert k1 == k2


# ------------------------------------------------------------------ #
# L2 ChromaDB Semantic Cache                                          #
# ------------------------------------------------------------------ #

class TestL2Cache:

    @pytest.fixture(autouse=True)
    def reset_global_collection(self):
        """Reset ChromaDB collection global before each test."""
        import core.semantic_cache as sc
        sc._chroma_cache_collection = None
        sc._schema_embedder = None
        yield
        sc._chroma_cache_collection = None
        sc._schema_embedder = None

    def _make_mock_collection(self):
        col = MagicMock()
        col.count.return_value = 0
        col.query.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
        return col

    def test_l2_cache_miss_empty_index(self):
        mock_col = self._make_mock_collection()
        with patch("core.semantic_cache._get_chroma_collection", return_value=mock_col):
            result = get_l2_cache("top customers", "main")
        assert result is None

    def test_l2_cache_miss_below_threshold(self):
        mock_col = self._make_mock_collection()
        mock_col.count.return_value = 1
        # Distance 0.2 → similarity 0.8, below default threshold 0.92
        mock_col.query.return_value = {
            "ids": [["key1"]],
            "distances": [[0.2]],
            "metadatas": [[{"database_name": "main", "expires_at": str(time.time() + 3600)}]],
            "documents": [[json.dumps(_make_result())]],
        }
        with patch("core.semantic_cache._get_chroma_collection", return_value=mock_col):
            with patch("core.semantic_cache._embed_query", return_value=[0.1] * 384):
                result = get_l2_cache("different query", "main")
        assert result is None

    def test_l2_cache_hit_above_threshold(self):
        mock_col = self._make_mock_collection()
        mock_col.count.return_value = 1
        # Distance 0.05 → similarity 0.95, above threshold 0.92
        mock_col.query.return_value = {
            "ids": [["key1"]],
            "distances": [[0.05]],
            "metadatas": [[{"database_name": "main", "expires_at": str(time.time() + 3600)}]],
            "documents": [[json.dumps(_make_result("SELECT id FROM customers"))]],
        }
        with patch("core.semantic_cache._get_chroma_collection", return_value=mock_col):
            with patch("core.semantic_cache._embed_query", return_value=[0.1] * 384):
                result = get_l2_cache("top 10 customers", "main")
        assert result is not None
        assert result["generated_sql"] == "SELECT id FROM customers"
        assert result["_cache_source"] == "l2_semantic"
        assert result["_cache_similarity"] >= 0.92

    def test_l2_cache_expired_entry(self):
        mock_col = self._make_mock_collection()
        mock_col.count.return_value = 1
        # expires_at in the past
        mock_col.query.return_value = {
            "ids": [["key1"]],
            "distances": [[0.01]],
            "metadatas": [[{"database_name": "main", "expires_at": str(time.time() - 100)}]],
            "documents": [[json.dumps(_make_result())]],
        }
        with patch("core.semantic_cache._get_chroma_collection", return_value=mock_col):
            with patch("core.semantic_cache._embed_query", return_value=[0.1] * 384):
                result = get_l2_cache("top customers", "main")
        assert result is None

    def test_set_l2_cache_calls_upsert(self):
        mock_col = self._make_mock_collection()
        result = _make_result("SELECT * FROM orders")
        with patch("core.semantic_cache._get_chroma_collection", return_value=mock_col):
            with patch("core.semantic_cache._embed_query", return_value=[0.1] * 384):
                set_l2_cache("show orders", "main", result)
        assert mock_col.upsert.called

    def test_set_l2_cache_no_embedder_skips(self):
        mock_col = self._make_mock_collection()
        with patch("core.semantic_cache._get_chroma_collection", return_value=mock_col):
            with patch("core.semantic_cache._embed_query", return_value=None):
                set_l2_cache("show orders", "main", _make_result())
        assert not mock_col.upsert.called


# ------------------------------------------------------------------ #
# L1 Redis Cache                                                       #
# ------------------------------------------------------------------ #

class TestL1Cache:

    @pytest.fixture(autouse=True)
    def reset_redis(self):
        import core.semantic_cache as sc
        sc._redis_client = None
        yield
        sc._redis_client = None

    @pytest.mark.asyncio
    async def test_l1_cache_miss_when_no_redis(self):
        with patch("core.semantic_cache._get_redis", return_value=None):
            from core.semantic_cache import get_l1_cache
            result = await get_l1_cache("top customers", "main")
        assert result is None

    @pytest.mark.asyncio
    async def test_l1_cache_hit(self):
        mock_redis = AsyncMock()
        cached_data = _make_result("SELECT id FROM t")
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        with patch("core.semantic_cache._get_redis", return_value=mock_redis):
            from core.semantic_cache import get_l1_cache
            result = await get_l1_cache("top customers", "main")
        assert result is not None
        assert result["generated_sql"] == "SELECT id FROM t"

    @pytest.mark.asyncio
    async def test_l1_cache_set(self):
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        with patch("core.semantic_cache._get_redis", return_value=mock_redis):
            from core.semantic_cache import set_l1_cache
            await set_l1_cache("top customers", "main", _make_result())
        assert mock_redis.setex.called


# ------------------------------------------------------------------ #
# get_cached_result / cache_result (combined)                          #
# ------------------------------------------------------------------ #

class TestCombinedCache:

    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        with patch("core.semantic_cache.get_l1_cache", new_callable=AsyncMock, return_value=None):
            with patch("core.semantic_cache.get_l2_cache", return_value=None):
                result, source = await get_cached_result("new query", "main")
        assert result is None
        assert source == "miss"

    @pytest.mark.asyncio
    async def test_l1_hit_returns_l1(self):
        hit = _make_result("SELECT 1")
        with patch("core.semantic_cache.get_l1_cache", new_callable=AsyncMock, return_value=hit):
            result, source = await get_cached_result("top customers", "main")
        assert source == "l1_exact"
        assert result is hit

    @pytest.mark.asyncio
    async def test_l2_hit_when_l1_miss(self):
        hit = _make_result("SELECT 2")
        hit["_cache_source"] = "l2_semantic"
        with patch("core.semantic_cache.get_l1_cache", new_callable=AsyncMock, return_value=None):
            with patch("core.semantic_cache.get_l2_cache", return_value=hit):
                result, source = await get_cached_result("show customers", "main")
        assert source == "l2_semantic"

    @pytest.mark.asyncio
    async def test_cache_result_skips_failed(self):
        with patch("core.semantic_cache.set_l1_cache", new_callable=AsyncMock) as mock_l1:
            await cache_result("q", "db", {"success": False, "generated_sql": None})
        assert not mock_l1.called

    @pytest.mark.asyncio
    async def test_cache_result_stores_success(self):
        with patch("core.semantic_cache.set_l1_cache", new_callable=AsyncMock) as mock_l1:
            with patch("core.semantic_cache.set_l2_cache") as mock_l2:
                await cache_result("q", "db", _make_result("SELECT 1"))
        assert mock_l1.called
        assert mock_l2.called
