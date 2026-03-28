"""
Schema Manager
Builds, caches, and serves rich database schema metadata for the PRISM agents.
Supports business glossary, column descriptions, and relationship graphs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from .database import DatabaseManager

logger = structlog.get_logger(__name__)


@dataclass
class ColumnInfo:
    """Metadata for a single database column."""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    column_default: str | None = None
    description: str | None = None
    sample_values: list[Any] = field(default_factory=list)
    foreign_key: dict[str, str] | None = None  # {"table": ..., "column": ...}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "is_nullable": self.is_nullable,
            "is_primary_key": self.is_primary_key,
            "column_default": self.column_default,
            "description": self.description,
            "sample_values": self.sample_values[:5],  # limit in output
            "foreign_key": self.foreign_key,
        }


@dataclass
class TableInfo:
    """Metadata for a single database table."""
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    description: str | None = None
    business_name: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def primary_keys(self) -> list[str]:
        return [c.name for c in self.columns if c.is_primary_key]

    @property
    def foreign_keys(self) -> list[dict[str, Any]]:
        return [
            {"column": c.name, **c.foreign_key}
            for c in self.columns
            if c.foreign_key
        ]

    def get_column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name.lower() == name.lower()), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "business_name": self.business_name,
            "row_count": self.row_count,
            "tags": self.tags,
            "columns": [c.to_dict() for c in self.columns],
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
        }

    def to_ddl_string(self) -> str:
        """Generate a CREATE TABLE-like DDL string for LLM context."""
        lines = [f"TABLE {self.name}"]
        if self.description:
            lines.append(f"  -- {self.description}")
        lines.append("(")
        col_lines = []
        for col in self.columns:
            nullable = "NULL" if col.is_nullable else "NOT NULL"
            pk = " PRIMARY KEY" if col.is_primary_key else ""
            fk = ""
            if col.foreign_key:
                fk = f" REFERENCES {col.foreign_key['table']}({col.foreign_key['column']})"
            desc = f"  -- {col.description}" if col.description else ""
            samples = ""
            if col.sample_values:
                sample_str = ", ".join(str(v) for v in col.sample_values[:3])
                samples = f"  -- e.g.: {sample_str}"
            col_lines.append(
                f"  {col.name} {col.data_type} {nullable}{pk}{fk}{desc}{samples}"
            )
        lines.append(",\n".join(col_lines))
        lines.append(")")
        return "\n".join(lines)


@dataclass
class DatabaseSchema:
    """Full database schema with all tables and relationships."""
    tables: dict[str, TableInfo] = field(default_factory=dict)
    relationships: list[dict[str, str]] = field(default_factory=list)
    dialect: str = "postgresql"
    database_name: str = "unknown"
    business_glossary: dict[str, str] = field(default_factory=dict)

    def get_table(self, name: str) -> TableInfo | None:
        return self.tables.get(name) or self.tables.get(name.lower())

    def find_tables_by_keyword(self, keyword: str) -> list[TableInfo]:
        """Find tables whose name or description contains the keyword."""
        keyword_lower = keyword.lower()
        return [
            t for t in self.tables.values()
            if keyword_lower in t.name.lower()
            or (t.description and keyword_lower in t.description.lower())
            or (t.business_name and keyword_lower in t.business_name.lower())
        ]

    def get_join_path(self, table1: str, table2: str) -> list[dict[str, str]] | None:
        """Find direct or indirect join paths between two tables."""
        # Direct join
        for rel in self.relationships:
            if (rel["from_table"] == table1 and rel["to_table"] == table2) or \
               (rel["from_table"] == table2 and rel["to_table"] == table1):
                return [rel]

        # Two-hop join (via intermediate table)
        for intermediate in self.tables:
            if intermediate in (table1, table2):
                continue
            hop1 = None
            hop2 = None
            for rel in self.relationships:
                if rel["from_table"] == table1 and rel["to_table"] == intermediate:
                    hop1 = rel
                elif rel["from_table"] == intermediate and rel["to_table"] == table2:
                    hop2 = rel
            if hop1 and hop2:
                return [hop1, hop2]

        return None

    def to_context_string(self, relevant_tables: list[str] | None = None) -> str:
        """Generate a schema context string for LLM prompts."""
        tables = (
            [self.tables[t] for t in relevant_tables if t in self.tables]
            if relevant_tables
            else list(self.tables.values())
        )

        parts = [
            f"Database: {self.database_name} ({self.dialect})",
            "",
            "## Schema",
            "",
        ]

        for table in tables:
            parts.append(table.to_ddl_string())
            parts.append("")

        if self.relationships:
            parts.append("## Relationships")
            for rel in self.relationships:
                parts.append(
                    f"  {rel['from_table']}.{rel['from_column']} "
                    f"→ {rel['to_table']}.{rel['to_column']}"
                )
            parts.append("")

        if self.business_glossary:
            parts.append("## Business Glossary")
            for term, definition in self.business_glossary.items():
                parts.append(f"  {term}: {definition}")

        return "\n".join(parts)

    def to_json(self) -> str:
        return json.dumps(
            {
                "database_name": self.database_name,
                "dialect": self.dialect,
                "tables": {name: t.to_dict() for name, t in self.tables.items()},
                "relationships": self.relationships,
                "business_glossary": self.business_glossary,
            },
            indent=2,
            default=str,
        )


class SchemaManager:
    """
    Manages database schema discovery, enrichment, and caching.

    Features:
    - Full schema introspection
    - Business glossary overlay
    - Relationship graph building
    - Partial schema loading (for large databases)
    - Schema change detection
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._schema_cache: DatabaseSchema | None = None
        self._business_glossary: dict[str, str] = {}
        self._table_descriptions: dict[str, str] = {}

    def load_business_glossary(self, glossary: dict[str, str]) -> None:
        """Load a business glossary mapping terms to definitions."""
        self._business_glossary.update(glossary)
        logger.info("Business glossary loaded", terms=len(glossary))

    def load_table_descriptions(self, descriptions: dict[str, str]) -> None:
        """Load human-readable table descriptions."""
        self._table_descriptions.update(descriptions)

    async def get_full_schema(
        self,
        include_samples: bool = True,
        sample_limit: int = 5,
        force_refresh: bool = False,
    ) -> DatabaseSchema:
        """
        Discover and return the full database schema.
        Cached after first load unless force_refresh=True.
        """
        if self._schema_cache and not force_refresh:
            return self._schema_cache

        logger.info("Discovering database schema...")
        schema = DatabaseSchema(
            dialect=self.db.dialect.value,
            business_glossary=self._business_glossary.copy(),
        )

        table_names = await self.db.get_all_tables()
        logger.info("Tables found", count=len(table_names))

        for table_name in table_names:
            table_info = await self._build_table_info(
                table_name, include_samples, sample_limit
            )
            schema.tables[table_name] = table_info

        # Build relationships
        schema.relationships = self._build_relationships(schema.tables)

        self._schema_cache = schema
        logger.info("Schema discovery complete", tables=len(schema.tables))
        return schema

    async def _build_table_info(
        self,
        table_name: str,
        include_samples: bool,
        sample_limit: int,
    ) -> TableInfo:
        """Build a TableInfo object for a single table."""
        raw_columns = await self.db.get_table_schema(table_name)
        fk_map = {}
        try:
            foreign_keys = await self.db.get_foreign_keys(table_name)
            fk_map = {fk["column_name"]: fk for fk in foreign_keys}
        except Exception:
            pass

        try:
            row_count = await self.db.get_table_row_count(table_name)
        except Exception:
            row_count = 0

        columns = []
        for raw_col in raw_columns:
            col_name = raw_col.get("column_name", "")
            fk = fk_map.get(col_name)
            foreign_key = None
            if fk:
                foreign_key = {
                    "table": fk.get("referenced_table", ""),
                    "column": fk.get("referenced_column", ""),
                }

            sample_values = []
            if include_samples and self._should_sample_column(raw_col.get("data_type", "")):
                try:
                    sample_values = await self.db.get_sample_values(
                        table_name, col_name, sample_limit
                    )
                except Exception:
                    pass

            columns.append(
                ColumnInfo(
                    name=col_name,
                    data_type=raw_col.get("data_type", "unknown"),
                    is_nullable=raw_col.get("is_nullable", "YES") == "YES",
                    is_primary_key=raw_col.get("is_primary_key", "NO") == "YES",
                    column_default=raw_col.get("column_default"),
                    sample_values=sample_values,
                    foreign_key=foreign_key,
                )
            )

        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=row_count,
            description=self._table_descriptions.get(table_name),
        )

    def _should_sample_column(self, data_type: str) -> bool:
        """Determine if sampling makes sense for this data type."""
        sample_types = {"character", "varchar", "text", "enum", "boolean", "char"}
        dt_lower = data_type.lower()
        return any(t in dt_lower for t in sample_types)

    def _build_relationships(
        self, tables: dict[str, TableInfo]
    ) -> list[dict[str, str]]:
        """Build relationship graph from foreign key constraints."""
        relationships = []
        for table_name, table in tables.items():
            for fk in table.foreign_keys:
                relationships.append(
                    {
                        "from_table": table_name,
                        "from_column": fk["column"],
                        "to_table": fk.get("table", ""),
                        "to_column": fk.get("column", ""),
                        "relationship_type": "MANY_TO_ONE",
                    }
                )
        return relationships

    async def get_relevant_schema(
        self, entities: list[str], max_tables: int = 10
    ) -> DatabaseSchema:
        """
        Get a focused schema containing only tables relevant to the given entities.
        Uses keyword matching and relationship traversal.
        """
        full_schema = await self.get_full_schema()

        relevant_tables: set[str] = set()

        # Direct entity matching
        for entity in entities:
            matched = full_schema.find_tables_by_keyword(entity)
            for t in matched:
                relevant_tables.add(t.name)

        # Expand via relationships (include join tables)
        expanded = set(relevant_tables)
        for table_name in relevant_tables:
            for rel in full_schema.relationships:
                if rel["from_table"] == table_name:
                    expanded.add(rel["to_table"])
                elif rel["to_table"] == table_name:
                    expanded.add(rel["from_table"])

        # Trim to max_tables
        final_tables = list(expanded)[:max_tables]

        focused = DatabaseSchema(
            dialect=full_schema.dialect,
            database_name=full_schema.database_name,
            business_glossary=full_schema.business_glossary,
        )
        focused.tables = {k: v for k, v in full_schema.tables.items() if k in final_tables}
        focused.relationships = [
            r for r in full_schema.relationships
            if r["from_table"] in final_tables and r["to_table"] in final_tables
        ]

        return focused
