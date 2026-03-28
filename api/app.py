"""
Text2SQL PRISM — FastAPI Application
=====================================
Uses Google ADK's get_fast_api_app() as the base, which provides:
  - POST /apps/{app_name}/users/{user_id}/sessions
  - GET  /apps/{app_name}/users/{user_id}/sessions/{session_id}
  - POST /apps/{app_name}/users/{user_id}/sessions/{session_id}/runs
  - DELETE /apps/{app_name}/users/{user_id}/sessions/{session_id}
  - POST /run / POST /run_sse   (convenience shorthand)

We mount custom enterprise endpoints ON TOP of the ADK app:
  - POST /api/v1/query          ← high-level NL→SQL convenience wrapper
  - GET  /api/v1/schema         ← schema discovery
  - GET  /api/v1/schema/index   ← what's in the vector index
  - POST /api/v1/schema/index   ← trigger bulk indexing of a schema
  - POST /api/v1/examples       ← add few-shot examples
  - GET  /api/v1/health         ← health check
  - GET  /api/v1/metrics        ← agent status

ADK app_name = "text2sql_prism"
Loaded from: agents/text2sql_prism/agent.py  (exports root_agent)
"""
from __future__ import annotations

import pathlib
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from api.models import (
    ExampleRequest,
    ExampleResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SchemaIndexListResponse,
    SchemaResponse,
)
from config.settings import get_settings

logger = structlog.get_logger(__name__)

# Path to the agents directory — ADK will discover text2sql_prism/ inside it
AGENTS_DIR = pathlib.Path(__file__).parent.parent / "agents"


def _build_adk_app() -> FastAPI:
    """
    Create the base FastAPI app using Google ADK's get_fast_api_app().

    Falls back to a plain FastAPI instance if google-adk is not installed,
    so the app still starts cleanly in development / test environments.
    """
    settings = get_settings()

    try:
        from google.adk.cli.fast_api import get_fast_api_app

        app = get_fast_api_app(
            agents_dir=str(AGENTS_DIR),
            # None = InMemorySessionService (swap for Redis URI in production)
            session_service_uri=None,
            artifact_service_uri=None,
            memory_service_uri=None,
            allow_origins=settings.api.allowed_origins,
            web=False,   # True enables the ADK Dev UI (useful in local dev)
        )
        logger.info(
            "ADK get_fast_api_app loaded",
            agents_dir=str(AGENTS_DIR),
            app_name="text2sql_prism",
        )
        return app

    except ImportError:
        logger.warning(
            "google-adk not installed — creating plain FastAPI app. "
            "Install google-adk to enable full ADK agent endpoints."
        )
        return FastAPI(
            title="Text2SQL PRISM API",
            description=(
                "Enterprise Text-to-SQL powered by the PRISM multi-agent swarm. "
                "Install google-adk for full ADK endpoints."
            ),
            version="1.0.0",
        )


def create_app() -> FastAPI:
    """
    Build the full application:
      base = ADK-managed FastAPI (agent run, session management)
      + custom enterprise endpoints mounted on /api/v1/...
    """
    settings = get_settings()
    app = _build_adk_app()

    # ------------------------------------------------------------------ #
    # Middleware (added after get_fast_api_app so they apply to all routes)
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = str(
            round((time.monotonic() - start) * 1000, 2)
        )
        return response

    # ------------------------------------------------------------------ #
    # Optional API key security                                           #
    # ------------------------------------------------------------------ #
    api_key_header = APIKeyHeader(
        name=settings.api.api_key_header, auto_error=False
    )

    async def get_api_key(key: str = Security(api_key_header)) -> str | None:
        return key

    # ------------------------------------------------------------------ #
    # Custom Enterprise Endpoints                                          #
    # ------------------------------------------------------------------ #

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Text2SQL PRISM",
            "version": "1.0.0",
            "docs": "/docs",
            "adk_app": "text2sql_prism",
            "adk_run_endpoint": "/apps/text2sql_prism/users/{user_id}/sessions/{session_id}/runs",
        }

    # ---- Health --------------------------------------------------------

    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
    )
    async def health_check():
        """Database connectivity and agent readiness check."""
        db_ok = False
        try:
            # Try Databricks first
            from core.databricks import get_databricks_connector
            conn = get_databricks_connector()
            if conn:
                conn.execute_query("SELECT 1")
                db_ok = True
            else:
                from core.database import get_db_manager
                db = await get_db_manager()
                await db.execute_safe_query("SELECT 1 AS ok")
                db_ok = True
        except Exception:
            pass

        index_count = 0
        try:
            from core.schema_index import get_schema_index
            index_count = get_schema_index().get_indexed_count()
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if db_ok else "degraded",
            database_connected=db_ok,
            agents_ready=True,
            schema_index_size=index_count,
        )

    # ---- NL Query → SQL ------------------------------------------------

    @app.post(
        "/api/v1/query",
        response_model=QueryResponse,
        tags=["Text2SQL"],
        summary="Natural language → SQL (convenience wrapper)",
        description=(
            "High-level endpoint that runs the full PRISM pipeline and returns "
            "structured results. Internally uses the ADK runner via "
            "agents/runner.py. For streaming or session-aware usage, use the "
            "native ADK endpoint: POST /apps/text2sql_prism/users/{uid}/sessions/{sid}/runs"
        ),
    )
    async def query_to_sql(
        request: QueryRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> QueryResponse:
        session_id = request.session_id or str(uuid.uuid4())
        pipeline_start = time.monotonic()

        logger.info(
            "Query received", session_id=session_id,
            preview=request.query[:80],
        )
        try:
            from agents.runner import run_prism_query
            result = await run_prism_query(
                query=request.query,
                session_id=session_id,
                database_name=request.database_name,
                max_rows=request.max_rows,
                execute_query=request.execute_query,
            )
            pipeline_ms = round((time.monotonic() - pipeline_start) * 1000, 2)
            return QueryResponse(
                success=result.get("success", False),
                session_id=session_id,
                query=request.query,
                generated_sql=result.get("generated_sql"),
                optimized_sql=result.get("optimized_sql"),
                columns=result.get("columns", []),
                rows=result.get("rows", []),
                row_count=result.get("row_count", 0),
                truncated=result.get("truncated", False),
                answer=result.get("answer"),
                confidence=result.get("confidence", 0.0),
                pipeline_stages=result.get("pipeline_stages", []),
                execution_time_ms=result.get("execution_time_ms", 0.0),
                pipeline_time_ms=pipeline_ms,
                error=result.get("error"),
                suggestions=result.get("suggestions", []),
            )
        except Exception as e:
            logger.error("Pipeline error", error=str(e), session_id=session_id)
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Schema discovery ----------------------------------------------

    @app.get(
        "/api/v1/schema",
        response_model=SchemaResponse,
        tags=["Schema"],
        summary="Get database / Unity Catalog schema",
    )
    async def get_schema(
        include_samples: bool = True,
        catalog: str | None = None,
        schema: str | None = None,
        api_key: str | None = Depends(get_api_key),
    ) -> SchemaResponse:
        """
        Retrieve schema. For Databricks, pass catalog + schema to scope results.
        For SQLAlchemy databases, uses the configured DATABASE_URL.
        """
        from core.databricks import get_databricks_connector
        conn = get_databricks_connector()

        if conn and catalog and schema:
            tables = conn.list_tables(catalog, schema)
            return SchemaResponse(
                database_name=f"{catalog}.{schema}",
                dialect="databricks_spark_sql",
                tables={
                    t["full_name"]: t for t in tables
                },
                relationships=[],
                total_tables=len(tables),
            )

        from agents.tools.schema_tools import get_database_schema
        data = await get_database_schema(include_samples=include_samples)
        return SchemaResponse(
            database_name=data.get("database_name", "unknown"),
            dialect=data.get("dialect", "unknown"),
            tables=data.get("tables", {}),
            relationships=data.get("relationships", []),
            total_tables=data.get("total_tables", 0),
        )

    # ---- Schema Index management --------------------------------------

    @app.get(
        "/api/v1/schema/index",
        response_model=SchemaIndexListResponse,
        tags=["Schema Index"],
        summary="List indexed tables in the vector store",
    )
    async def list_schema_index(
        catalog: str | None = None,
        limit: int = 50,
        api_key: str | None = Depends(get_api_key),
    ) -> SchemaIndexListResponse:
        """List tables currently in the schema vector index."""
        from agents.tools.indexing_tools import list_indexed_tables
        result = list_indexed_tables(catalog=catalog, limit=limit)
        return SchemaIndexListResponse(
            tables=result["tables"],
            total_indexed=result["total_indexed"],
            catalog_filter=catalog,
        )

    @app.post(
        "/api/v1/schema/index",
        response_model=IndexResponse,
        tags=["Schema Index"],
        summary="Bulk-index a Unity Catalog schema",
        description=(
            "Index all tables in a catalog.schema into the vector store. "
            "Run this once after connecting a new Databricks workspace, "
            "then agents auto-index newly encountered tables incrementally."
        ),
    )
    async def index_schema(
        request: IndexRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> IndexResponse:
        """Trigger bulk indexing of a Unity Catalog schema."""
        from agents.tools.indexing_tools import bulk_index_schema
        result = bulk_index_schema(
            catalog=request.catalog,
            schema=request.schema,
            max_tables=request.max_tables,
        )
        return IndexResponse(**result)

    @app.post(
        "/api/v1/schema/index/{table_id:path}",
        tags=["Schema Index"],
        summary="Index or refresh a specific table",
    )
    async def index_single_table(
        table_id: str,
        force_refresh: bool = False,
        api_key: str | None = Depends(get_api_key),
    ) -> dict[str, Any]:
        """Index a specific table by its fully-qualified ID (catalog.schema.table)."""
        from agents.tools.indexing_tools import index_table_if_new, refresh_table_index
        if force_refresh:
            return refresh_table_index(table_id)
        return index_table_if_new(table_id)

    # ---- Few-shot examples --------------------------------------------

    @app.post(
        "/api/v1/examples",
        response_model=ExampleResponse,
        tags=["Examples"],
        summary="Add a verified NL→SQL example to the few-shot store",
    )
    async def add_example(
        request: ExampleRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> ExampleResponse:
        from agents.tools.few_shot_tools import add_example_to_store
        result = add_example_to_store(
            question=request.question,
            sql=request.sql,
            database_name=request.database_name,
            tags=request.tags,
            feedback_score=request.feedback_score,
        )
        return ExampleResponse(
            success=result.get("success", False),
            example_id=result.get("example_id"),
            total_examples=result.get("total_examples", 0),
            error=result.get("error"),
        )

    # ---- Metrics -------------------------------------------------------

    @app.get("/api/v1/metrics", tags=["System"], summary="Agent and index metrics")
    async def metrics(api_key: str | None = Depends(get_api_key)) -> dict[str, Any]:
        index_count = 0
        try:
            from core.schema_index import get_schema_index
            index_count = get_schema_index().get_indexed_count()
        except Exception:
            pass

        return {
            "service": "text2sql-prism",
            "version": "1.0.0",
            "schema_index_size": index_count,
            "agents": {
                "schema_discovery": "ready",
                "metadata_enrichment": "ready",
                "deep_think_analyzer": "ready",
                "schema_linker": "ready",
                "sql_generator": "ready",
                "sql_validator": "ready",
                "query_optimizer": "ready",
                "response_formatter": "ready",
            },
            "adk_app": "text2sql_prism",
            "adk_endpoints": {
                "run": "/apps/text2sql_prism/users/{user_id}/sessions/{session_id}/runs",
                "create_session": "/apps/text2sql_prism/users/{user_id}/sessions",
                "get_session": "/apps/text2sql_prism/users/{user_id}/sessions/{session_id}",
            },
        }

    return app


app = create_app()
