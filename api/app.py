"""
Text2SQL PRISM — FastAPI Application
Enterprise REST API for the PRISM Text2SQL multi-agent system.

Endpoints:
  POST /api/v1/query         - Convert NL query to SQL and execute
  GET  /api/v1/schema        - Get database schema information
  POST /api/v1/examples      - Add few-shot examples
  GET  /api/v1/health        - Health check
  GET  /api/v1/metrics       - System metrics
"""
from __future__ import annotations

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
    QueryRequest,
    QueryResponse,
    SchemaResponse,
)
from config.settings import get_settings
from core.database import get_db_manager
from agents.tools.schema_tools import get_database_schema
from agents.tools.few_shot_tools import add_example_to_store

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown application resources."""
    settings = get_settings()
    logger.info("Starting Text2SQL PRISM API...")

    # Initialize database
    try:
        db = await get_db_manager()
        logger.info("Database connected", dialect=db.dialect.value)
    except Exception as e:
        logger.warning("Database initialization failed", error=str(e))

    # Initialize ADK Runner (lazy — done on first query)
    logger.info("Text2SQL PRISM API ready")
    yield

    # Cleanup
    logger.info("Shutting down Text2SQL PRISM API...")
    try:
        db = await get_db_manager()
        await db.close()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Text2SQL PRISM API",
        description=(
            "Enterprise Text-to-SQL API powered by the PRISM multi-agent swarm. "
            "Converts natural language questions into accurate, optimized SQL queries "
            "using Google ADK agents with Deep Think reasoning."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # --- Request timing middleware ---
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.monotonic()
        response = await call_next(request)
        process_time = (time.monotonic() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
        return response

    # --- API Key security ---
    api_key_header = APIKeyHeader(
        name=settings.api.api_key_header,
        auto_error=False,
    )

    async def get_api_key(api_key: str = Security(api_key_header)) -> str | None:
        """Validate API key (if configured)."""
        # In production, validate against a secure store
        # For now, accept any key or no key
        return api_key

    # ------------------------------------------------------------------ #
    # Routes                                                               #
    # ------------------------------------------------------------------ #

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "Text2SQL PRISM", "version": "1.0.0", "docs": "/docs"}

    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
    )
    async def health_check():
        """Check the health status of the API and its dependencies."""
        db_connected = False
        try:
            db = await get_db_manager()
            await db.execute_safe_query("SELECT 1 AS ok")
            db_connected = True
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if db_connected else "degraded",
            database_connected=db_connected,
            agents_ready=True,
        )

    @app.post(
        "/api/v1/query",
        response_model=QueryResponse,
        tags=["Text2SQL"],
        summary="Convert natural language to SQL and execute",
        description=(
            "The main endpoint. Accepts a natural language question and runs it through "
            "the full PRISM pipeline: Schema Discovery → Deep Think Reasoning → "
            "SQL Generation → Validation → Optimization → Execution."
        ),
    )
    async def query_to_sql(
        request: QueryRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> QueryResponse:
        """Convert a natural language query to SQL using the PRISM swarm."""
        session_id = request.session_id or str(uuid.uuid4())
        pipeline_start = time.monotonic()

        logger.info(
            "Query received",
            session_id=session_id,
            query_preview=request.query[:100],
        )

        try:
            # Import here to avoid circular imports during startup
            from agents.runner import run_prism_query

            result = await run_prism_query(
                query=request.query,
                session_id=session_id,
                database_name=request.database_name,
                max_rows=request.max_rows,
                execute_query=request.execute_query,
            )

            pipeline_time_ms = (time.monotonic() - pipeline_start) * 1000

            logger.info(
                "Query completed",
                session_id=session_id,
                pipeline_time_ms=round(pipeline_time_ms, 2),
                success=result.get("success", False),
            )

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
                pipeline_time_ms=round(pipeline_time_ms, 2),
                error=result.get("error"),
                suggestions=result.get("suggestions", []),
            )

        except Exception as e:
            logger.error("Query pipeline failed", error=str(e), session_id=session_id)
            pipeline_time_ms = (time.monotonic() - pipeline_start) * 1000
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "session_id": session_id,
                    "pipeline_time_ms": round(pipeline_time_ms, 2),
                },
            )

    @app.get(
        "/api/v1/schema",
        response_model=SchemaResponse,
        tags=["Schema"],
        summary="Get database schema",
        description="Retrieve the full database schema including tables, columns, and relationships.",
    )
    async def get_schema(
        include_samples: bool = True,
        api_key: str | None = Depends(get_api_key),
    ) -> SchemaResponse:
        """Get the database schema."""
        try:
            schema_data = await get_database_schema(include_samples=include_samples)
            return SchemaResponse(
                database_name=schema_data.get("database_name", "unknown"),
                dialect=schema_data.get("dialect", "unknown"),
                tables=schema_data.get("tables", {}),
                relationships=schema_data.get("relationships", []),
                total_tables=schema_data.get("total_tables", 0),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/api/v1/examples",
        response_model=ExampleResponse,
        tags=["Examples"],
        summary="Add a few-shot example",
        description="Add a verified question-SQL pair to the few-shot example store for improved generation.",
    )
    async def add_example(
        request: ExampleRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> ExampleResponse:
        """Add a new few-shot example to the store."""
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

    @app.get(
        "/api/v1/metrics",
        tags=["System"],
        summary="System metrics",
    )
    async def get_metrics(api_key: str | None = Depends(get_api_key)) -> dict[str, Any]:
        """Return basic system metrics."""
        return {
            "service": "text2sql-prism",
            "version": "1.0.0",
            "uptime": "available",
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
        }

    return app


# Create the application instance
app = create_app()
