"""
Schema Tools for PRISM Agents
Google ADK-compatible tool functions for schema discovery and introspection.
Used primarily by the Schema Discovery and Metadata Enrichment agents.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from core.database import get_db_manager
from core.schema_manager import SchemaManager

logger = structlog.get_logger(__name__)

# Module-level schema manager (initialized lazily)
_schema_manager: SchemaManager | None = None


async def _get_schema_manager() -> SchemaManager:
    global _schema_manager
    if _schema_manager is None:
        db = await get_db_manager()
        _schema_manager = SchemaManager(db)
    return _schema_manager


async def get_database_schema(
    include_samples: bool = True,
    max_tables: int = 50,
) -> dict[str, Any]:
    """
    Retrieve the complete database schema including all tables, columns,
    data types, primary keys, foreign keys, and relationships.

    This is the primary schema discovery tool. Call this first to understand
    the database structure before generating any SQL.

    Args:
        include_samples: Whether to include sample column values (helps with filter inference)
        max_tables: Maximum number of tables to return (for very large databases)

    Returns:
        dict with keys:
            - database_name (str): Database name
            - dialect (str): SQL dialect (postgresql, mysql, sqlite, etc.)
            - tables (dict): Table schemas keyed by table name
            - relationships (list): Foreign key relationships
            - business_glossary (dict): Business term mappings
            - schema_context (str): Formatted schema for LLM consumption
            - total_tables (int): Total number of tables
    """
    try:
        manager = await _get_schema_manager()
        schema = await manager.get_full_schema(include_samples=include_samples)

        # Limit tables for output
        table_subset = dict(list(schema.tables.items())[:max_tables])

        return {
            "database_name": schema.database_name,
            "dialect": schema.dialect,
            "tables": {name: t.to_dict() for name, t in table_subset.items()},
            "relationships": schema.relationships,
            "business_glossary": schema.business_glossary,
            "schema_context": schema.to_context_string(
                list(table_subset.keys()) if len(schema.tables) > max_tables else None
            ),
            "total_tables": len(schema.tables),
            "truncated": len(schema.tables) > max_tables,
        }
    except Exception as e:
        logger.error("Schema discovery failed", error=str(e))
        return {"error": str(e), "tables": {}, "relationships": []}


async def get_table_details(
    table_name: str,
    include_samples: bool = True,
) -> dict[str, Any]:
    """
    Get detailed schema information for a specific table.

    Use this after get_database_schema to get deeper information about
    a particular table, including full column details and sample values.

    Args:
        table_name: The exact table name to retrieve details for
        include_samples: Whether to include sample column values

    Returns:
        dict with the table's full schema details including:
            - name, description, business_name
            - columns (list): Full column info with types, constraints, samples
            - primary_keys, foreign_keys
            - row_count
            - ddl_string: SQL CREATE TABLE representation
    """
    try:
        db = await get_db_manager()
        manager = await _get_schema_manager()

        schema = await manager.get_full_schema(include_samples=include_samples)
        table = schema.get_table(table_name)

        if not table:
            return {"error": f"Table '{table_name}' not found in schema"}

        result = table.to_dict()
        result["ddl_string"] = table.to_ddl_string()
        return result
    except Exception as e:
        logger.error("Failed to get table details", table=table_name, error=str(e))
        return {"error": str(e)}


async def find_related_tables(
    table_name: str,
    max_hops: int = 2,
) -> dict[str, Any]:
    """
    Find tables related to the specified table through foreign key relationships.

    Use this to discover which tables need to be JOINed together to answer
    a query that spans multiple tables.

    Args:
        table_name: The source table to find relations for
        max_hops: Maximum relationship hops (1=direct, 2=two-hop joins)

    Returns:
        dict with:
            - direct_relations (list): Directly related tables
            - indirect_relations (list): Tables reachable via join paths
            - join_paths (list): Full join path specifications
    """
    try:
        manager = await _get_schema_manager()
        schema = await manager.get_full_schema(include_samples=False)

        direct_relations = []
        indirect_relations = []
        join_paths = []

        for rel in schema.relationships:
            if rel["from_table"] == table_name:
                direct_relations.append({
                    "table": rel["to_table"],
                    "join_on": f"{table_name}.{rel['from_column']} = {rel['to_table']}.{rel['to_column']}",
                    "direction": "outbound",
                })
            elif rel["to_table"] == table_name:
                direct_relations.append({
                    "table": rel["from_table"],
                    "join_on": f"{rel['from_table']}.{rel['from_column']} = {table_name}.{rel['to_column']}",
                    "direction": "inbound",
                })

        # Two-hop relationships
        if max_hops >= 2:
            direct_table_names = {r["table"] for r in direct_relations}
            for hop1_table in direct_table_names:
                for rel in schema.relationships:
                    if rel["from_table"] == hop1_table and rel["to_table"] != table_name:
                        if rel["to_table"] not in direct_table_names:
                            indirect_relations.append({
                                "table": rel["to_table"],
                                "via": hop1_table,
                                "hops": 2,
                            })
                            path = schema.get_join_path(table_name, rel["to_table"])
                            if path:
                                join_paths.append({
                                    "from": table_name,
                                    "to": rel["to_table"],
                                    "path": path,
                                })

        return {
            "source_table": table_name,
            "direct_relations": direct_relations,
            "indirect_relations": indirect_relations[:10],
            "join_paths": join_paths[:5],
        }
    except Exception as e:
        logger.error("Failed to find related tables", table=table_name, error=str(e))
        return {"error": str(e), "direct_relations": [], "indirect_relations": []}


async def get_sample_values(
    table_name: str,
    column_name: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Get sample/distinct values from a specific column.

    Use this to understand the actual values in filter columns,
    especially for enum-like columns, status fields, or category columns.
    This helps generate accurate WHERE clause conditions.

    Args:
        table_name: The table containing the column
        column_name: The column to sample values from
        limit: Number of distinct values to return (max 20)

    Returns:
        dict with:
            - table_name, column_name
            - values (list): Sample distinct values
            - value_count (int): Number of values returned
    """
    try:
        db = await get_db_manager()
        limit = min(limit, 20)
        values = await db.get_sample_values(table_name, column_name, limit)
        return {
            "table_name": table_name,
            "column_name": column_name,
            "values": values,
            "value_count": len(values),
        }
    except Exception as e:
        return {
            "table_name": table_name,
            "column_name": column_name,
            "values": [],
            "value_count": 0,
            "error": str(e),
        }


async def search_schema_by_keyword(
    keyword: str,
    search_columns: bool = True,
) -> dict[str, Any]:
    """
    Search the schema for tables and columns matching a keyword.

    Use this when you need to find which tables/columns correspond to
    a business term or natural language entity.

    Args:
        keyword: The search term to find in table/column names and descriptions
        search_columns: Whether to also search column names (default: True)

    Returns:
        dict with:
            - matching_tables (list): Tables matching the keyword
            - matching_columns (list): Columns matching the keyword
            - suggestions (list): Best guesses for the intended schema element
    """
    try:
        manager = await _get_schema_manager()
        schema = await manager.get_full_schema(include_samples=False)

        keyword_lower = keyword.lower()
        matching_tables = []
        matching_columns = []

        for table_name, table in schema.tables.items():
            if keyword_lower in table_name.lower():
                matching_tables.append({
                    "table": table_name,
                    "match_type": "table_name",
                    "description": table.description,
                })
            elif table.description and keyword_lower in table.description.lower():
                matching_tables.append({
                    "table": table_name,
                    "match_type": "table_description",
                    "description": table.description,
                })

            if search_columns:
                for col in table.columns:
                    if keyword_lower in col.name.lower():
                        matching_columns.append({
                            "table": table_name,
                            "column": col.name,
                            "data_type": col.data_type,
                            "match_type": "column_name",
                        })

        # Check business glossary
        glossary_matches = {
            term: defn
            for term, defn in schema.business_glossary.items()
            if keyword_lower in term.lower()
        }

        # Build suggestions (top 5 most relevant)
        suggestions = []
        for m in matching_tables[:3]:
            suggestions.append(f"Table: {m['table']}")
        for m in matching_columns[:5]:
            suggestions.append(f"Column: {m['table']}.{m['column']} ({m['data_type']})")

        return {
            "keyword": keyword,
            "matching_tables": matching_tables[:10],
            "matching_columns": matching_columns[:20],
            "glossary_matches": glossary_matches,
            "suggestions": suggestions,
        }
    except Exception as e:
        return {"keyword": keyword, "error": str(e), "matching_tables": [], "matching_columns": []}
