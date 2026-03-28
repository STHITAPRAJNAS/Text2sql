"""
Tests for the Schema Manager
Tests schema discovery, table info building, and relationship mapping.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.schema_manager import (
    ColumnInfo,
    DatabaseSchema,
    SchemaManager,
    TableInfo,
)


class TestColumnInfo:

    def test_to_dict(self):
        col = ColumnInfo(
            name="customer_id",
            data_type="integer",
            is_nullable=False,
            is_primary_key=True,
            sample_values=[1, 2, 3],
        )
        d = col.to_dict()
        assert d["name"] == "customer_id"
        assert d["is_primary_key"] is True
        assert d["sample_values"] == [1, 2, 3]


class TestTableInfo:

    def _make_table(self) -> TableInfo:
        return TableInfo(
            name="orders",
            columns=[
                ColumnInfo("order_id", "integer", False, True),
                ColumnInfo(
                    "customer_id",
                    "integer",
                    False,
                    False,
                    foreign_key={"table": "customers", "column": "customer_id"},
                ),
                ColumnInfo("amount", "numeric", True, False),
            ],
            row_count=5000,
            description="Customer orders table",
        )

    def test_primary_keys(self):
        table = self._make_table()
        assert table.primary_keys == ["order_id"]

    def test_foreign_keys(self):
        table = self._make_table()
        fks = table.foreign_keys
        assert len(fks) == 1
        assert fks[0]["column"] == "customer_id"
        assert fks[0]["table"] == "customers"

    def test_to_ddl_string(self):
        table = self._make_table()
        ddl = table.to_ddl_string()
        assert "TABLE orders" in ddl
        assert "order_id" in ddl
        assert "customer_id" in ddl
        assert "PRIMARY KEY" in ddl

    def test_get_column(self):
        table = self._make_table()
        col = table.get_column("amount")
        assert col is not None
        assert col.data_type == "numeric"

    def test_get_column_case_insensitive(self):
        table = self._make_table()
        col = table.get_column("ORDER_ID")
        assert col is not None


class TestDatabaseSchema:

    def _make_schema(self) -> DatabaseSchema:
        customers = TableInfo(
            name="customers",
            columns=[
                ColumnInfo("customer_id", "integer", False, True),
                ColumnInfo("name", "text", False, False),
            ],
            row_count=1000,
        )
        orders = TableInfo(
            name="orders",
            columns=[
                ColumnInfo("order_id", "integer", False, True),
                ColumnInfo(
                    "customer_id",
                    "integer",
                    False,
                    False,
                    foreign_key={"table": "customers", "column": "customer_id"},
                ),
            ],
            row_count=5000,
        )

        schema = DatabaseSchema(dialect="postgresql", database_name="test_db")
        schema.tables = {"customers": customers, "orders": orders}
        schema.relationships = [
            {
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers",
                "to_column": "customer_id",
                "relationship_type": "MANY_TO_ONE",
            }
        ]
        return schema

    def test_get_table(self):
        schema = self._make_schema()
        table = schema.get_table("customers")
        assert table is not None
        assert table.name == "customers"

    def test_get_table_case_insensitive(self):
        schema = self._make_schema()
        table = schema.get_table("ORDERS")
        assert table is not None

    def test_find_tables_by_keyword(self):
        schema = self._make_schema()
        results = schema.find_tables_by_keyword("order")
        assert any(t.name == "orders" for t in results)

    def test_get_join_path_direct(self):
        schema = self._make_schema()
        path = schema.get_join_path("orders", "customers")
        assert path is not None
        assert len(path) == 1
        assert path[0]["from_table"] == "orders"

    def test_get_join_path_no_path(self):
        schema = self._make_schema()
        path = schema.get_join_path("orders", "products")
        assert path is None

    def test_to_context_string(self):
        schema = self._make_schema()
        context = schema.to_context_string()
        assert "customers" in context
        assert "orders" in context
        assert "Relationships" in context

    def test_to_context_string_filtered(self):
        schema = self._make_schema()
        context = schema.to_context_string(["customers"])
        assert "customers" in context
        # orders should not be in filtered context
        assert "orders" not in context.split("##")[0]

    def test_to_json(self):
        import json
        schema = self._make_schema()
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert data["database_name"] == "test_db"
        assert "customers" in data["tables"]
        assert len(data["relationships"]) == 1
