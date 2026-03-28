"""
Enterprise Database Manager
Async SQLAlchemy-based database layer with connection pooling, multi-dialect support,
and safe query execution for the Text2SQL PRISM system.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from enum import Enum
from functools import lru_cache
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


class SQLDialect(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"


def detect_dialect(database_url: str) -> SQLDialect:
    """Detect SQL dialect from the database URL."""
    url_lower = database_url.lower()
    if "postgresql" in url_lower or "postgres" in url_lower:
        return SQLDialect.POSTGRESQL
    elif "mysql" in url_lower:
        return SQLDialect.MYSQL
    elif "bigquery" in url_lower:
        return SQLDialect.BIGQUERY
    elif "snowflake" in url_lower:
        return SQLDialect.SNOWFLAKE
    elif "redshift" in url_lower:
        return SQLDialect.REDSHIFT
    return SQLDialect.SQLITE


class QueryResult:
    """Encapsulates a SQL query execution result."""

    def __init__(
        self,
        columns: list[str],
        rows: list[dict[str, Any]],
        row_count: int,
        execution_time_ms: float,
        truncated: bool = False,
    ):
        self.columns = columns
        self.rows = rows
        self.row_count = row_count
        self.execution_time_ms = execution_time_ms
        self.truncated = truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "truncated": self.truncated,
        }


class DatabaseManager:
    """
    Async database manager with:
    - Connection pooling
    - Multi-database support
    - Safe read-only query execution
    - Schema introspection
    - Query timeout enforcement
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._db_settings = settings.database
        self.dialect = detect_dialect(self._db_settings.database_url)

        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker | None = None

        logger.info("DatabaseManager initialized", dialect=self.dialect.value)

    async def initialize(self) -> None:
        """Create engine and session factory."""
        connect_args: dict[str, Any] = {}

        # Add connect_args per dialect
        if self.dialect == SQLDialect.POSTGRESQL:
            connect_args["command_timeout"] = 30  # seconds

        self._engine = create_async_engine(
            self._db_settings.database_url,
            echo=self._db_settings.database_echo,
            pool_size=self._db_settings.database_pool_size,
            max_overflow=self._db_settings.database_max_overflow,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        logger.info("Database engine created", url=self._db_settings.database_url[:50])

    async def close(self) -> None:
        """Dispose the engine and release all connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database engine disposed")

    @asynccontextmanager
    async def session(self):
        """Async context manager providing a database session."""
        if not self._session_factory:
            raise RuntimeError("DatabaseManager not initialized. Call initialize() first.")
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def execute_safe_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        timeout_ms: int = 30000,
    ) -> QueryResult:
        """
        Execute a read-only SQL query safely.

        Enforces:
        - SELECT-only queries (no DML/DDL)
        - Row limit to prevent OOM
        - Execution timeout
        """
        sql_stripped = sql.strip().rstrip(";")

        # Safety: only allow SELECT statements
        if not sql_stripped.upper().startswith(("SELECT", "WITH", "EXPLAIN")):
            raise ValueError(
                f"Only SELECT queries are allowed. Got: {sql_stripped[:50]}..."
            )

        start_time = time.monotonic()

        async with self.session() as sess:
            # Set statement-level timeout for PostgreSQL
            if self.dialect == SQLDialect.POSTGRESQL:
                await sess.execute(
                    text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'")
                )

            result = await sess.execute(
                text(sql_stripped),
                params or {},
            )

            execution_time_ms = (time.monotonic() - start_time) * 1000

            keys = list(result.keys())
            max_rows = self._db_settings.max_rows_return
            all_rows = result.fetchmany(max_rows + 1)

            truncated = len(all_rows) > max_rows
            rows_to_return = all_rows[:max_rows]

            rows = [dict(zip(keys, row)) for row in rows_to_return]

            logger.info(
                "Query executed",
                row_count=len(rows),
                execution_time_ms=round(execution_time_ms, 2),
                truncated=truncated,
            )

            return QueryResult(
                columns=keys,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=round(execution_time_ms, 2),
                truncated=truncated,
            )

    async def get_all_tables(self) -> list[str]:
        """Return a list of all table names in the database."""
        dialect_queries = {
            SQLDialect.POSTGRESQL: """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """,
            SQLDialect.MYSQL: """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """,
            SQLDialect.SQLITE: """
                SELECT name FROM sqlite_master
                WHERE type = 'table'
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """,
        }

        query = dialect_queries.get(
            self.dialect,
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
        )

        result = await self.execute_safe_query(query)
        return [row[list(row.keys())[0]] for row in result.rows]

    async def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get full column schema for a specific table."""
        dialect_queries = {
            SQLDialect.POSTGRESQL: """
                SELECT
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    c.character_maximum_length,
                    c.numeric_precision,
                    c.numeric_scale,
                    CASE WHEN kcu.column_name IS NOT NULL THEN 'YES' ELSE 'NO' END AS is_primary_key,
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.columns c
                LEFT JOIN information_schema.key_column_usage kcu
                    ON c.table_name = kcu.table_name
                    AND c.column_name = kcu.column_name
                    AND kcu.constraint_name IN (
                        SELECT constraint_name FROM information_schema.table_constraints
                        WHERE constraint_type = 'PRIMARY KEY' AND table_name = :table_name
                    )
                LEFT JOIN information_schema.referential_constraints rc
                    ON kcu.constraint_name = rc.constraint_name
                LEFT JOIN information_schema.constraint_column_usage ccu
                    ON rc.unique_constraint_name = ccu.constraint_name
                WHERE c.table_name = :table_name
                AND c.table_schema = 'public'
                ORDER BY c.ordinal_position
            """,
            SQLDialect.SQLITE: """
                SELECT
                    name AS column_name,
                    type AS data_type,
                    CASE "notnull" WHEN 0 THEN 'YES' ELSE 'NO' END AS is_nullable,
                    dflt_value AS column_default,
                    CASE pk WHEN 1 THEN 'YES' ELSE 'NO' END AS is_primary_key,
                    NULL AS foreign_table,
                    NULL AS foreign_column
                FROM pragma_table_info(:table_name)
                ORDER BY cid
            """,
            SQLDialect.MYSQL: """
                SELECT
                    c.COLUMN_NAME AS column_name,
                    c.DATA_TYPE AS data_type,
                    c.IS_NULLABLE AS is_nullable,
                    c.COLUMN_DEFAULT AS column_default,
                    c.CHARACTER_MAXIMUM_LENGTH AS character_maximum_length,
                    c.NUMERIC_PRECISION AS numeric_precision,
                    c.NUMERIC_SCALE AS numeric_scale,
                    CASE WHEN c.COLUMN_KEY = 'PRI' THEN 'YES' ELSE 'NO' END AS is_primary_key,
                    kcu.REFERENCED_TABLE_NAME AS foreign_table,
                    kcu.REFERENCED_COLUMN_NAME AS foreign_column
                FROM information_schema.COLUMNS c
                LEFT JOIN information_schema.KEY_COLUMN_USAGE kcu
                    ON c.TABLE_NAME = kcu.TABLE_NAME
                    AND c.COLUMN_NAME = kcu.COLUMN_NAME
                    AND c.TABLE_SCHEMA = kcu.TABLE_SCHEMA
                    AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
                WHERE c.TABLE_NAME = :table_name
                AND c.TABLE_SCHEMA = DATABASE()
                ORDER BY c.ORDINAL_POSITION
            """,
        }

        query = dialect_queries.get(self.dialect, dialect_queries[SQLDialect.POSTGRESQL])
        result = await self.execute_safe_query(query, {"table_name": table_name})
        return result.rows

    async def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """Get foreign key relationships for a table."""
        if self.dialect == SQLDialect.POSTGRESQL:
            query = """
                SELECT
                    kcu.column_name,
                    ccu.table_name AS referenced_table,
                    ccu.column_name AS referenced_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.referential_constraints rc
                    ON tc.constraint_name = rc.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON rc.unique_constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = :table_name
            """
        elif self.dialect == SQLDialect.SQLITE:
            query = """
                SELECT
                    "from" AS column_name,
                    "table" AS referenced_table,
                    "to" AS referenced_column
                FROM pragma_foreign_key_list(:table_name)
            """
        else:
            return []

        result = await self.execute_safe_query(query, {"table_name": table_name})
        return result.rows

    async def get_sample_values(
        self, table_name: str, column_name: str, limit: int = 10
    ) -> list[Any]:
        """Get distinct sample values for a column."""
        query = f"""
            SELECT DISTINCT {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            LIMIT {limit}
        """
        result = await self.execute_safe_query(query)
        return [row[column_name] for row in result.rows]

    async def get_table_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table."""
        if self.dialect == SQLDialect.POSTGRESQL:
            query = """
                SELECT reltuples::bigint AS row_count
                FROM pg_class
                WHERE relname = :table_name
            """
            result = await self.execute_safe_query(query, {"table_name": table_name})
            if result.rows:
                return int(result.rows[0].get("row_count", 0))

        # Fallback: exact count
        result = await self.execute_safe_query(
            f"SELECT COUNT(*) AS row_count FROM {table_name}"
        )
        return int(result.rows[0].get("row_count", 0)) if result.rows else 0

    async def explain_query(self, sql: str) -> str:
        """Get the query execution plan."""
        explain_prefix = {
            SQLDialect.POSTGRESQL: "EXPLAIN (FORMAT JSON, ANALYZE false)",
            SQLDialect.MYSQL: "EXPLAIN FORMAT=JSON",
            SQLDialect.SQLITE: "EXPLAIN QUERY PLAN",
        }.get(self.dialect, "EXPLAIN")

        result = await self.execute_safe_query(f"{explain_prefix} {sql}")
        if result.rows:
            return str(result.rows)
        return "EXPLAIN not available"


# Global singleton
_db_manager: DatabaseManager | None = None


async def get_db_manager() -> DatabaseManager:
    """Get or create the global DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        settings = get_settings()
        _db_manager = DatabaseManager(settings)
        await _db_manager.initialize()
    return _db_manager
