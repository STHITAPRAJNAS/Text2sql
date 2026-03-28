"""
Databricks Native Connector
Full Unity Catalog support using databricks-sdk + databricks-sql-connector.
Falls back gracefully to SQLAlchemy for non-Databricks databases.

Why native over SQLAlchemy for Databricks:
  - 3-level namespace  (catalog.schema.table) — SQLAlchemy loses the catalog level
  - HTTP transport to SQL Warehouse — SQLAlchemy DBAPI wrapper is slower
  - Unity Catalog metadata APIs — richer than INFORMATION_SCHEMA
  - Proper token/OAuth support — databricks-sdk handles credential chain
  - Result caching hints, compute routing — only available natively
  - Delta Lake specifics: DESCRIBE DETAIL, DESCRIBE HISTORY, OPTIMIZE hints
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DatabricksConfig:
    """Connection configuration for a Databricks workspace."""
    host: str                          # e.g. "adb-xxxx.azuredatabricks.net"
    http_path: str                     # SQL warehouse: /sql/1.0/warehouses/xxx
    token: str | None = None           # PAT or None for credential chain
    catalog: str = "main"              # Default Unity Catalog catalog
    schema: str = "default"            # Default schema
    client_id: str | None = None       # For OAuth M2M
    client_secret: str | None = None
    connect_timeout: int = 30
    query_timeout: int = 600           # 10 min default for complex queries


@dataclass
class UCTableRef:
    """Three-level Unity Catalog table reference."""
    catalog: str
    schema: str
    table: str

    def __str__(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{self.table}`"

    @classmethod
    def parse(cls, ref: str) -> "UCTableRef":
        """Parse 'catalog.schema.table' or 'schema.table' or 'table'.
        Handles backtick-quoted identifiers like `main`.`sales`.`orders`.
        """
        parts = [p.strip("`") for p in ref.split(".")]
        if len(parts) == 3:
            return cls(*parts)
        if len(parts) == 2:
            return cls("main", *parts)
        return cls("main", "default", parts[0])


class DatabricksQueryResult:
    """Query result from the Databricks SQL connector."""

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


class DatabricksConnector:
    """
    Native Databricks connector using databricks-sql-connector for query
    execution and databricks-sdk (WorkspaceClient) for Unity Catalog metadata.

    Design principles:
    - Connection pooling via the SQL connector's built-in pool
    - Separate paths for query execution vs metadata (different APIs)
    - Lazy initialization — connect on first use
    - Graceful degradation if SDK not available (INFORMATION_SCHEMA fallback)
    """

    def __init__(self, config: DatabricksConfig):
        self.config = config
        self._sql_connection = None
        self._sdk_client = None
        self._sdk_available = False

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    def _get_sql_connection(self):
        """Get or create the Databricks SQL connector connection."""
        if self._sql_connection is None:
            try:
                from databricks import sql as dbsql

                connect_kwargs: dict[str, Any] = {
                    "server_hostname": self.config.host,
                    "http_path": self.config.http_path,
                    "connection_timeout": self.config.connect_timeout,
                }
                if self.config.token:
                    connect_kwargs["access_token"] = self.config.token
                elif self.config.client_id:
                    connect_kwargs["client_id"] = self.config.client_id
                    connect_kwargs["client_secret"] = self.config.client_secret

                self._sql_connection = dbsql.connect(**connect_kwargs)
                logger.info("Databricks SQL connection established", host=self.config.host)
            except ImportError:
                raise RuntimeError(
                    "databricks-sql-connector not installed. "
                    "Run: pip install databricks-sql-connector"
                )
        return self._sql_connection

    def _get_sdk_client(self):
        """Get or create the Databricks SDK WorkspaceClient."""
        if self._sdk_client is None:
            try:
                from databricks.sdk import WorkspaceClient

                if self.config.token:
                    self._sdk_client = WorkspaceClient(
                        host=f"https://{self.config.host}",
                        token=self.config.token,
                    )
                elif self.config.client_id:
                    self._sdk_client = WorkspaceClient(
                        host=f"https://{self.config.host}",
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret,
                    )
                else:
                    # Use credential chain (env vars, ~/.databrickscfg, etc.)
                    self._sdk_client = WorkspaceClient(
                        host=f"https://{self.config.host}"
                    )
                self._sdk_available = True
                logger.info("Databricks SDK client initialized")
            except ImportError:
                logger.warning(
                    "databricks-sdk not installed — using INFORMATION_SCHEMA fallback. "
                    "Run: pip install databricks-sdk for full Unity Catalog support."
                )
        return self._sdk_client

    def close(self) -> None:
        if self._sql_connection:
            try:
                self._sql_connection.close()
            except Exception:
                pass
            self._sql_connection = None

    # ------------------------------------------------------------------ #
    # Query Execution                                                      #
    # ------------------------------------------------------------------ #

    def execute_query(
        self,
        sql: str,
        max_rows: int = 1000,
        timeout_ms: int = 30_000,
    ) -> DatabricksQueryResult:
        """
        Execute a SQL query on Databricks SQL warehouse.
        Only SELECT, WITH, EXPLAIN are allowed (enforced before this call).
        """
        conn = self._get_sql_connection()
        start = time.monotonic()

        with conn.cursor() as cursor:
            # Set query timeout at session level
            cursor.execute(
                f"SET spark.sql.execution.timeout={timeout_ms // 1000}s"
            )
            cursor.execute(sql)

            schema = cursor.description  # list of (name, type, ...) tuples
            if not schema:
                return DatabricksQueryResult([], [], 0, 0.0)

            columns = [col[0] for col in schema]
            raw_rows = cursor.fetchmany(max_rows + 1)

            elapsed_ms = (time.monotonic() - start) * 1000
            truncated = len(raw_rows) > max_rows
            rows_to_return = raw_rows[:max_rows]

            rows = [dict(zip(columns, row)) for row in rows_to_return]

            return DatabricksQueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=round(elapsed_ms, 2),
                truncated=truncated,
            )

    # ------------------------------------------------------------------ #
    # Unity Catalog Metadata (SDK-first, INFORMATION_SCHEMA fallback)     #
    # ------------------------------------------------------------------ #

    def list_catalogs(self) -> list[str]:
        """List all Unity Catalog catalogs the current identity can access."""
        client = self._get_sdk_client()
        if client and self._sdk_available:
            return [c.name for c in client.catalogs.list() if c.name]
        # Fallback
        result = self.execute_query("SHOW CATALOGS")
        return [r.get("catalog", "") for r in result.rows]

    def list_schemas(self, catalog: str) -> list[str]:
        """List all schemas in a catalog."""
        client = self._get_sdk_client()
        if client and self._sdk_available:
            return [
                s.name for s in client.schemas.list(catalog_name=catalog)
                if s.name and s.name not in ("information_schema",)
            ]
        result = self.execute_query(f"SHOW SCHEMAS IN `{catalog}`")
        key = result.columns[0] if result.columns else "databaseName"
        return [r.get(key, "") for r in result.rows]

    def list_tables(self, catalog: str, schema: str) -> list[dict[str, Any]]:
        """List tables in a schema with basic metadata."""
        client = self._get_sdk_client()
        if client and self._sdk_available:
            tables = []
            for t in client.tables.list(catalog_name=catalog, schema_name=schema):
                tables.append({
                    "catalog": catalog,
                    "schema": schema,
                    "table": t.name,
                    "table_type": str(t.table_type),
                    "comment": t.comment or "",
                    "full_name": f"{catalog}.{schema}.{t.name}",
                })
            return tables
        # Fallback via SQL
        result = self.execute_query(f"SHOW TABLES IN `{catalog}`.`{schema}`")
        return [
            {
                "catalog": catalog,
                "schema": schema,
                "table": r.get("tableName", r.get("table", "")),
                "table_type": r.get("tableType", "MANAGED"),
                "comment": "",
                "full_name": f"{catalog}.{schema}.{r.get('tableName', '')}",
            }
            for r in result.rows
        ]

    def get_table_metadata(self, ref: UCTableRef) -> dict[str, Any]:
        """
        Get full column metadata for a specific table.
        Uses SDK TableInfo when available for richest metadata.
        """
        client = self._get_sdk_client()

        if client and self._sdk_available:
            try:
                t_info = client.tables.get(f"{ref.catalog}.{ref.schema}.{ref.table}")
                columns = []
                for col in (t_info.columns or []):
                    columns.append({
                        "name": col.name,
                        "data_type": str(col.type_name) if col.type_name else "string",
                        "type_text": col.type_text or "",
                        "comment": col.comment or "",
                        "is_nullable": col.nullable if col.nullable is not None else True,
                        "is_partition_col": col.partition_index is not None,
                        "partition_index": col.partition_index,
                    })
                return {
                    "full_name": f"{ref.catalog}.{ref.schema}.{ref.table}",
                    "catalog": ref.catalog,
                    "schema": ref.schema,
                    "table": ref.table,
                    "comment": t_info.comment or "",
                    "table_type": str(t_info.table_type),
                    "columns": columns,
                    "owner": t_info.owner or "",
                    "created_at": str(t_info.created_at) if t_info.created_at else "",
                    "updated_at": str(t_info.updated_at) if t_info.updated_at else "",
                    "properties": dict(t_info.properties or {}),
                }
            except Exception as e:
                logger.warning("SDK table fetch failed, using SQL fallback", error=str(e))

        # SQL fallback via INFORMATION_SCHEMA
        return self._get_table_metadata_sql(ref)

    def _get_table_metadata_sql(self, ref: UCTableRef) -> dict[str, Any]:
        """Fallback: fetch column metadata via INFORMATION_SCHEMA."""
        sql = f"""
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            comment
        FROM `{ref.catalog}`.information_schema.columns
        WHERE table_schema = '{ref.schema}'
          AND table_name   = '{ref.table}'
        ORDER BY ordinal_position
        """
        result = self.execute_query(sql)
        columns = [
            {
                "name": r.get("column_name", ""),
                "data_type": r.get("data_type", "string"),
                "type_text": r.get("data_type", ""),
                "comment": r.get("comment", "") or "",
                "is_nullable": r.get("is_nullable", "YES") == "YES",
                "is_partition_col": False,
                "partition_index": None,
            }
            for r in result.rows
        ]
        return {
            "full_name": str(ref),
            "catalog": ref.catalog,
            "schema": ref.schema,
            "table": ref.table,
            "comment": "",
            "table_type": "MANAGED",
            "columns": columns,
            "owner": "",
            "created_at": "",
            "updated_at": "",
            "properties": {},
        }

    def get_table_stats(self, ref: UCTableRef) -> dict[str, Any]:
        """Get Delta table statistics (row count, size, last modified)."""
        try:
            result = self.execute_query(f"DESCRIBE DETAIL {ref}")
            if result.rows:
                row = result.rows[0]
                return {
                    "row_count": row.get("numRows", -1),
                    "size_bytes": row.get("sizeInBytes", -1),
                    "num_files": row.get("numFiles", -1),
                    "last_modified": str(row.get("lastModified", "")),
                    "format": row.get("format", "delta"),
                    "location": row.get("location", ""),
                    "partitioning": row.get("partitionColumns", []),
                    "clustering_columns": row.get("clusteringColumns", []),
                }
        except Exception:
            pass
        # Fallback: COUNT(*)
        try:
            result = self.execute_query(f"SELECT COUNT(*) AS cnt FROM {ref}")
            count = result.rows[0].get("cnt", -1) if result.rows else -1
            return {"row_count": count, "size_bytes": -1, "num_files": -1}
        except Exception:
            return {"row_count": -1, "size_bytes": -1, "num_files": -1}

    def get_sample_values(
        self, ref: UCTableRef, column: str, limit: int = 10
    ) -> list[Any]:
        """Get distinct sample values for a column."""
        try:
            result = self.execute_query(
                f"SELECT DISTINCT `{column}` FROM {ref} "
                f"WHERE `{column}` IS NOT NULL LIMIT {limit}"
            )
            return [r.get(column) for r in result.rows]
        except Exception:
            return []

    def explain_query(self, sql: str) -> str:
        """Get the Spark SQL EXPLAIN plan."""
        try:
            result = self.execute_query(f"EXPLAIN EXTENDED {sql}")
            if result.rows:
                return str(result.rows[0])
        except Exception as e:
            return f"EXPLAIN failed: {e}"
        return ""

    def search_tables_by_comment(
        self,
        keyword: str,
        catalog: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search UC tables by name or comment via information_schema."""
        cat = catalog or self.config.catalog
        sql = f"""
        SELECT
            table_catalog, table_schema, table_name, comment
        FROM `{cat}`.information_schema.tables
        WHERE LOWER(table_name)    LIKE LOWER('%{keyword}%')
           OR LOWER(comment)       LIKE LOWER('%{keyword}%')
        LIMIT {limit}
        """
        try:
            result = self.execute_query(sql)
            return result.rows
        except Exception:
            return []


# ------------------------------------------------------------------ #
# Module-level singleton management                                   #
# ------------------------------------------------------------------ #

_databricks_connector: DatabricksConnector | None = None


def get_databricks_connector() -> DatabricksConnector | None:
    """Return the active Databricks connector, or None if not configured."""
    return _databricks_connector


def init_databricks_connector(config: DatabricksConfig) -> DatabricksConnector:
    """Initialize and return the singleton Databricks connector."""
    global _databricks_connector
    _databricks_connector = DatabricksConnector(config)
    logger.info(
        "Databricks connector initialized",
        host=config.host,
        catalog=config.catalog,
    )
    return _databricks_connector


def databricks_config_from_settings() -> DatabricksConfig | None:
    """Build a DatabricksConfig from application settings, or None if not configured."""
    from config.settings import get_settings
    settings = get_settings()
    db_cfg = settings.databricks
    if not db_cfg.host:
        return None
    return DatabricksConfig(
        host=db_cfg.host,
        http_path=db_cfg.http_path,
        token=db_cfg.token.get_secret_value() if db_cfg.token else None,
        catalog=db_cfg.default_catalog,
        schema=db_cfg.default_schema,
        client_id=db_cfg.client_id,
        client_secret=db_cfg.client_secret.get_secret_value() if db_cfg.client_secret else None,
    )
