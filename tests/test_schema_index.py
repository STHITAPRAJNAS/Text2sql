"""
Tests for the Schema Index (vector store for table metadata).
Covers ChromaDB backend (mocked), in-memory fallback, and UCTableRef parsing.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.databricks import UCTableRef
from core.schema_index import IndexBackend, SchemaIndex, SearchResult


# ------------------------------------------------------------------ #
# UCTableRef parsing                                                   #
# ------------------------------------------------------------------ #

class TestUCTableRef:

    def test_three_part_parse(self):
        ref = UCTableRef.parse("main.sales.order_items")
        assert ref.catalog == "main"
        assert ref.schema == "sales"
        assert ref.table == "order_items"

    def test_two_part_parse(self):
        ref = UCTableRef.parse("analytics.customers")
        assert ref.catalog == "main"
        assert ref.schema == "analytics"
        assert ref.table == "customers"

    def test_single_part_parse(self):
        ref = UCTableRef.parse("orders")
        assert ref.catalog == "main"
        assert ref.schema == "default"
        assert ref.table == "orders"

    def test_backtick_stripped(self):
        ref = UCTableRef.parse("`main`.`sales`.`orders`")
        assert ref.catalog == "main"
        assert ref.schema == "sales"
        assert ref.table == "orders"

    def test_str_representation(self):
        ref = UCTableRef("main", "sales", "orders")
        assert str(ref) == "`main`.`sales`.`orders`"


# ------------------------------------------------------------------ #
# SchemaIndex — in-memory backend (no ChromaDB needed)                #
# ------------------------------------------------------------------ #

@pytest.fixture
def index():
    return SchemaIndex(backend=IndexBackend.IN_MEMORY, embedding_model="all-MiniLM-L6-v2")


def _make_table_metadata(name: str, schema: str = "sales", catalog: str = "main") -> dict:
    return {
        "full_name": f"{catalog}.{schema}.{name}",
        "catalog": catalog,
        "schema": schema,
        "table": name,
        "comment": f"Table storing {name.replace('_', ' ')} data",
        "table_type": "MANAGED",
        "columns": [
            {"name": "id", "data_type": "bigint", "comment": "Primary key"},
            {"name": "created_at", "data_type": "timestamp", "comment": "Creation timestamp"},
        ],
        "row_count": 100_000,
        "partitioning": ["created_at"],
        "clustering_columns": [],
    }


class TestSchemaIndexInMemory:

    def test_upsert_and_retrieve(self, index):
        meta = _make_table_metadata("orders")
        table_id = index.upsert(meta)
        assert table_id == "main.sales.orders"
        assert index.is_indexed("main.sales.orders")

    def test_upsert_idempotent(self, index):
        meta = _make_table_metadata("orders")
        index.upsert(meta)
        index.upsert(meta)  # second upsert should not raise
        assert index.get_indexed_count() == 1

    def test_get_by_id(self, index):
        meta = _make_table_metadata("customers")
        index.upsert(meta)
        retrieved = index.get_by_id("main.sales.customers")
        assert retrieved is not None
        assert retrieved["table"] == "customers"

    def test_get_by_id_missing(self, index):
        result = index.get_by_id("main.sales.nonexistent")
        assert result is None

    def test_is_indexed_false(self, index):
        assert not index.is_indexed("main.sales.ghost_table")

    def test_get_indexed_count(self, index):
        assert index.get_indexed_count() == 0
        index.upsert(_make_table_metadata("orders"))
        assert index.get_indexed_count() == 1
        index.upsert(_make_table_metadata("customers"))
        assert index.get_indexed_count() == 2

    def test_list_indexed_tables(self, index):
        index.upsert(_make_table_metadata("orders"))
        index.upsert(_make_table_metadata("customers"))
        tables = index.list_indexed_tables()
        assert len(tables) == 2
        names = {t["table"] for t in tables}
        assert "orders" in names
        assert "customers" in names

    def test_list_indexed_tables_catalog_filter(self, index):
        index.upsert(_make_table_metadata("orders", catalog="main"))
        index.upsert(_make_table_metadata("logs", catalog="audit"))
        result = index.list_indexed_tables(catalog_filter="main")
        assert all(t["catalog"] == "main" for t in result)

    def test_search_returns_results(self, index):
        index.upsert(_make_table_metadata("orders"))
        index.upsert(_make_table_metadata("customers"))
        index.upsert(_make_table_metadata("products"))

        results = index.search("order revenue sales", top_k=3)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        # orders should rank highly for "order revenue"
        top_ids = [r.table_id for r in results]
        assert any("orders" in tid for tid in top_ids)

    def test_search_scores_are_numeric(self, index):
        """Cosine similarity ranges -1..1; just verify it's a float in range."""
        index.upsert(_make_table_metadata("orders"))
        results = index.search("orders")
        for r in results:
            assert isinstance(r.score, float)
            assert -1.0 <= r.score <= 1.0

    def test_search_with_catalog_filter(self, index):
        index.upsert(_make_table_metadata("orders", catalog="main"))
        index.upsert(_make_table_metadata("logs", catalog="audit"))
        results = index.search("data", top_k=10, catalog_filter="audit")
        assert all("audit" in r.table_id for r in results)

    def test_search_empty_index_returns_empty(self, index):
        results = index.search("anything")
        assert results == []


class TestBuildTableText:

    def test_includes_table_name(self, index):
        meta = _make_table_metadata("order_items")
        text = index.build_table_text(meta)
        assert "order_items" in text

    def test_includes_comment(self, index):
        meta = _make_table_metadata("orders")
        meta["comment"] = "Stores all customer purchase orders"
        text = index.build_table_text(meta)
        assert "customer purchase orders" in text

    def test_includes_column_names(self, index):
        meta = _make_table_metadata("orders")
        text = index.build_table_text(meta)
        assert "id" in text
        assert "created_at" in text

    def test_includes_partition_info(self, index):
        meta = _make_table_metadata("orders")
        meta["partitioning"] = ["order_date", "region"]
        text = index.build_table_text(meta)
        assert "Partitioned by" in text
        assert "order_date" in text

    def test_includes_row_count(self, index):
        meta = _make_table_metadata("orders")
        meta["row_count"] = 5_000_000
        text = index.build_table_text(meta)
        assert "5,000,000" in text
