"""
Text2SQL PRISM — FastAPI Application
=====================================
Uses Google ADK's get_fast_api_app() as the base, which provides:
  - POST /apps/{app_name}/users/{user_id}/sessions
  - GET  /apps/{app_name}/users/{user_id}/sessions/{session_id}
  - POST /apps/{app_name}/users/{user_id}/sessions/{session_id}/runs
  - DELETE /apps/{app_name}/users/{user_id}/sessions/{session_id}
  - POST /run / POST /run_sse   (convenience shorthand)

Custom enterprise endpoints:
  - POST /api/v1/query          ← high-level NL→SQL convenience wrapper
  - GET  /api/v1/schema         ← schema discovery
  - GET  /api/v1/schema/index   ← what's in the vector index
  - POST /api/v1/schema/index   ← trigger bulk indexing of a schema
  - POST /api/v1/examples       ← add few-shot examples
  - POST /api/v1/feedback       ← user ratings + active learning
  - DELETE /api/v1/cache        ← flush semantic cache
  - GET  /api/v1/health         ← health check
  - GET  /api/v1/metrics        ← telemetry + cache + feedback stats

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
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader

from api.models import (
    CacheStatsResponse,
    DeadLetterItem,
    DeadLetterResponse,
    ExampleRequest,
    ExampleResponse,
    ExportFormat,
    FeedbackRequest,
    FeedbackResponse,
    GlossaryTerm,
    GlossaryUpsertRequest,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    IndexRequest,
    IndexResponse,
    PerformanceBaseline,
    QueryRequest,
    QueryResponse,
    RateLimitStats,
    ResolveDeadLetterRequest,
    SavedQuery,
    SavedQueryRequest,
    SavedQueryRunRequest,
    SchemaIndexListResponse,
    SchemaResponse,
)
from config.settings import get_settings

logger = structlog.get_logger(__name__)

# Path to the agents directory — ADK will discover text2sql_prism/ inside it
AGENTS_DIR = pathlib.Path(__file__).parent.parent / "agents"


def _init_otel(app: FastAPI) -> None:
    """
    4. FastAPI auto-instrumentation via FastAPIInstrumentor.
    Adds http.method, http.route, http.status_code, http.duration spans
    for every request automatically.
    """
    settings = get_settings()
    try:
        from core.telemetry import init_telemetry
        init_telemetry(
            service_name="text2sql-prism",
            otlp_endpoint=settings.observability.otel_exporter_otlp_endpoint,
            enabled=settings.observability.enable_tracing,
        )
        if settings.observability.enable_tracing:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(
                app,
                excluded_urls="health,metrics,docs,openapi.json",
                server_request_hook=_otel_server_request_hook,
            )
            logger.info("FastAPI OTel auto-instrumentation enabled")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-fastapi not installed — skipping auto-instrumentation")
    except Exception as exc:
        logger.warning("OTel FastAPI instrumentation failed", error=str(exc))


def _otel_server_request_hook(span, scope: dict) -> None:
    """Add correlation ID and user ID to every auto-instrumented HTTP span."""
    if span and span.is_recording():
        headers = dict(scope.get("headers", []))
        correlation_id = headers.get(b"x-request-id", b"").decode()
        user_id = headers.get(b"x-user-id", b"").decode()
        if correlation_id:
            span.set_attribute("http.request_id", correlation_id)
        if user_id:
            span.set_attribute("user.id", user_id)


def _build_adk_app() -> FastAPI:
    """
    Create the base FastAPI app using Google ADK's get_fast_api_app().

    Passes session_service_uri (DatabaseSessionService) and memory_service_uri
    (VertexAiMemoryBankService or None for in-memory) so the ADK Dev UI and
    /run endpoint both use persistent storage.

    Falls back to a plain FastAPI instance if google-adk is not installed.
    """
    settings = get_settings()

    try:
        from google.adk.cli.fast_api import get_fast_api_app
        from core.session_store import get_session_service_uri
        from core.memory_store import get_memory_service_uri

        session_uri = get_session_service_uri()
        memory_uri = get_memory_service_uri()

        app = get_fast_api_app(
            agents_dir=str(AGENTS_DIR),
            session_service_uri=session_uri,
            artifact_service_uri=None,
            memory_service_uri=memory_uri,
            allow_origins=settings.api.allowed_origins,
            web=False,
        )
        logger.info(
            "ADK get_fast_api_app loaded",
            agents_dir=str(AGENTS_DIR),
            app_name="text2sql_prism",
            session_backend="database" if session_uri else "in_memory",
            memory_backend=settings.memory.backend,
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
    # 4. OTel FastAPI auto-instrumentation (must be before middleware)     #
    # ------------------------------------------------------------------ #
    _init_otel(app)

    # ------------------------------------------------------------------ #
    # Middleware                                                           #
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def add_timing_and_correlation(request: Request, call_next):
        # 2. W3C TraceContext propagation: extract from incoming headers
        # so downstream spans are children of the caller's trace.
        try:
            from core.telemetry import extract_trace_context, get_current_trace_ids
            from opentelemetry import context as otel_context
            ctx = extract_trace_context(dict(request.headers))
            token = otel_context.attach(ctx) if ctx else None
        except Exception:
            token = None

        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            if token is not None:
                try:
                    otel_context.detach(token)
                except Exception:
                    pass

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        response.headers["X-Request-ID"] = correlation_id

        # Inject trace IDs into response headers for client-side correlation
        try:
            from core.telemetry import get_current_trace_ids
            trace_ids = get_current_trace_ids()
            if trace_ids.get("trace_id"):
                response.headers["X-Trace-ID"] = trace_ids["trace_id"]
                response.headers["X-Span-ID"] = trace_ids.get("span_id", "")
        except Exception:
            pass
        return response

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Only apply to API endpoints
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)
        try:
            from core.rate_limiter import check_rate_limit
            api_key_val = request.headers.get(settings.api.api_key_header)
            user_id_val = request.headers.get("X-User-ID")
            allowed, retry_after = check_rate_limit(user_id=user_id_val, api_key=api_key_val)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after_seconds": retry_after},
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception:
            pass
        return await call_next(request)

    # ------------------------------------------------------------------ #
    # Optional API key security                                           #
    # ------------------------------------------------------------------ #
    api_key_header = APIKeyHeader(name=settings.api.api_key_header, auto_error=False)

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
        summary="Natural language → SQL",
        description=(
            "Runs the full PRISM pipeline with semantic cache (L1+L2). "
            "Returns clarification request when query is ambiguous. "
            "For streaming, use the ADK endpoint: "
            "POST /apps/text2sql_prism/users/{uid}/sessions/{sid}/runs"
        ),
    )
    async def query_to_sql(
        request: QueryRequest,
        http_request: Request,
        api_key: str | None = Depends(get_api_key),
    ) -> QueryResponse:
        session_id = request.session_id or str(uuid.uuid4())
        correlation_id = http_request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tenant_id = (
            http_request.headers.get(settings.tenant.tenant_id_header)
            or request.tenant_id
            or settings.tenant.default_tenant
        )
        pipeline_start = time.monotonic()

        logger.info(
            "Query received", session_id=session_id,
            preview=request.query[:80], cid=correlation_id,
        )
        try:
            from agents.runner import run_prism_query
            result = await run_prism_query(
                query=request.query,
                session_id=session_id,
                database_name=request.database_name,
                max_rows=request.max_rows,
                execute_query=request.execute_query,
                correlation_id=correlation_id,
                tenant_id=tenant_id,
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
                needs_clarification=result.get("needs_clarification", False),
                clarification_question=result.get("clarification_question"),
                clarification_options=result.get("clarification_options", []),
                cache_hit=result.get("cache_hit", False),
                cache_source=result.get("cache_source"),
                cost_warning=result.get("cost_warning"),
                pii_report=result.get("pii_report"),
                anomalies=result.get("anomalies", []),
                correlation_id=correlation_id,
                token_input=result.get("token_input"),
                token_output=result.get("token_output"),
            )
        except Exception as e:
            logger.error("Pipeline error", error=str(e), session_id=session_id)
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Feedback ------------------------------------------------------

    @app.post(
        "/api/v1/feedback",
        response_model=FeedbackResponse,
        tags=["Feedback"],
        summary="Submit feedback on a generated SQL query",
        description=(
            "Record user rating (1-5) on generated SQL. "
            "Ratings ≥ 4 automatically add the query to the few-shot store "
            "and ADK memory service for self-improvement. "
            "Corrections (corrected_sql) are immediately added as high-value examples."
        ),
    )
    async def submit_feedback(
        request: FeedbackRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> FeedbackResponse:
        from agents.tools.feedback_tools import record_feedback
        result = record_feedback(
            query=request.query,
            sql=request.sql,
            rating=request.rating,
            database_name=request.database_name,
            session_id=request.session_id,
            comment=request.comment,
            corrected_sql=request.corrected_sql,
        )
        return FeedbackResponse(**result)

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
        """For Databricks, pass catalog + schema to scope results."""
        from core.databricks import get_databricks_connector
        conn = get_databricks_connector()

        if conn and catalog and schema:
            tables = conn.list_tables(catalog, schema)
            return SchemaResponse(
                database_name=f"{catalog}.{schema}",
                dialect="databricks_spark_sql",
                tables={t["full_name"]: t for t in tables},
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
    )
    async def index_schema(
        request: IndexRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> IndexResponse:
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
        from agents.tools.indexing_tools import index_table_if_new, refresh_table_index
        if force_refresh:
            return refresh_table_index(table_id)
        return index_table_if_new(table_id)

    # ---- Cache management ---------------------------------------------

    @app.delete(
        "/api/v1/cache",
        tags=["Cache"],
        summary="Flush semantic query result cache",
    )
    async def flush_cache(
        database_name: str | None = None,
        api_key: str | None = Depends(get_api_key),
    ) -> dict[str, Any]:
        """Invalidate L1 (Redis) and L2 (ChromaDB) cache entries."""
        from core.semantic_cache import clear_cache
        deleted = await clear_cache(database_name=database_name)
        return {"status": "flushed", "l1_keys_deleted": deleted, "database": database_name}

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

    # ---- SSE Streaming query ------------------------------------------

    @app.get(
        "/api/v1/query/stream",
        tags=["Text2SQL"],
        summary="Stream NL→SQL pipeline events via SSE",
        description=(
            "Server-Sent Events stream of pipeline progress. "
            "Events: phase, token, sql, result, error, done."
        ),
    )
    async def stream_query(
        q: str = Query(min_length=3, max_length=2000, description="Natural language question"),
        database_name: str = Query(default="default"),
        session_id: str | None = Query(default=None),
        max_rows: int = Query(default=100, ge=1, le=1000),
        api_key: str | None = Depends(get_api_key),
    ):
        from agents.runner import stream_prism_query

        async def event_generator():
            async for event in stream_prism_query(
                query=q,
                session_id=session_id,
                database_name=database_name,
                max_rows=max_rows,
                execute_query=True,
            ):
                yield event

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---- Glossary CRUD ------------------------------------------------

    @app.get(
        "/api/v1/glossary",
        response_model=list[GlossaryTerm],
        tags=["Glossary"],
        summary="List all business glossary terms",
    )
    async def list_glossary(
        search: str | None = Query(default=None, description="Keyword search"),
        limit: int = Query(default=200, ge=1, le=1000),
        api_key: str | None = Depends(get_api_key),
    ) -> list[GlossaryTerm]:
        from core.glossary import list_all, search as glossary_search
        if search:
            items = await glossary_search(search, limit=limit)
        else:
            items = await list_all(limit=limit)
        return [GlossaryTerm(**item) for item in items]

    @app.get(
        "/api/v1/glossary/{term}",
        response_model=GlossaryTerm,
        tags=["Glossary"],
        summary="Look up a single business term",
    )
    async def get_glossary_term(
        term: str,
        api_key: str | None = Depends(get_api_key),
    ) -> GlossaryTerm:
        from core.glossary import lookup
        entry = await lookup(term)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Term '{term}' not found")
        return GlossaryTerm(**entry)

    @app.post(
        "/api/v1/glossary",
        response_model=GlossaryTerm,
        tags=["Glossary"],
        summary="Add or update a business glossary term",
        status_code=201,
    )
    async def upsert_glossary_term(
        request: GlossaryUpsertRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> GlossaryTerm:
        from core.glossary import upsert, lookup
        await upsert(
            term=request.term,
            table_name=request.table_name,
            column_name=request.column_name,
            filter_sql=request.filter_sql,
            description=request.description,
            example_sql=request.example_sql,
        )
        entry = await lookup(request.term)
        return GlossaryTerm(**(entry or {"term": request.term}))

    @app.delete(
        "/api/v1/glossary/{term}",
        tags=["Glossary"],
        summary="Delete a business glossary term",
    )
    async def delete_glossary_term(
        term: str,
        api_key: str | None = Depends(get_api_key),
    ) -> dict[str, Any]:
        from core.glossary import delete
        deleted = await delete(term)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Term '{term}' not found")
        return {"status": "deleted", "term": term}

    # ---- Query History ------------------------------------------------

    @app.get(
        "/api/v1/history",
        response_model=HistoryResponse,
        tags=["History"],
        summary="Query history for a user or session",
    )
    async def get_history(
        user_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        api_key: str | None = Depends(get_api_key),
    ) -> HistoryResponse:
        from core.audit_log import get_query_history
        rows = await get_query_history(
            user_id=user_id, session_id=session_id, limit=limit, offset=offset
        )
        items = [
            HistoryItem(
                query_id=r.get("query_id", ""),
                nl_query=r.get("nl_query", ""),
                generated_sql=r.get("generated_sql"),
                success=bool(r.get("success")),
                confidence=float(r.get("confidence") or 0.0),
                execution_time_ms=float(r.get("execution_time_ms") or 0.0),
                pipeline_time_ms=float(r.get("pipeline_time_ms") or 0.0),
                row_count=int(r.get("row_count") or 0),
                cache_hit=bool(r.get("cache_hit")),
                cache_source=r.get("cache_source"),
                pii_detected=bool(r.get("pii_detected")),
                cost_warning=r.get("cost_warning"),
                needs_clarification=bool(r.get("needs_clarification")),
                created_at=float(r.get("created_at") or 0.0),
            )
            for r in rows
        ]
        return HistoryResponse(items=items, total=len(items), limit=limit, offset=offset)

    # ---- Dead Letter Queue --------------------------------------------

    @app.get(
        "/api/v1/review-queue",
        response_model=DeadLetterResponse,
        tags=["Review Queue"],
        summary="Failed queries pending human review",
    )
    async def get_review_queue(
        reviewed: bool | None = Query(default=False, description="Filter by reviewed status"),
        limit: int = Query(default=50, ge=1, le=500),
        api_key: str | None = Depends(get_api_key),
    ) -> DeadLetterResponse:
        from core.audit_log import get_dead_letters
        rows = await get_dead_letters(reviewed=reviewed, limit=limit)
        items = [
            DeadLetterItem(
                id=r.get("id", ""),
                query_id=r.get("query_id", ""),
                nl_query=r.get("nl_query", ""),
                generated_sql=r.get("generated_sql"),
                error=r.get("error", ""),
                failure_reason=r.get("failure_reason", ""),
                database_name=r.get("database_name", "default"),
                confidence=float(r.get("confidence") or 0.0),
                reviewed=bool(r.get("reviewed")),
                corrected_sql=r.get("corrected_sql"),
                created_at=float(r.get("created_at") or 0.0),
            )
            for r in rows
        ]
        return DeadLetterResponse(items=items, total=len(items))

    @app.post(
        "/api/v1/review-queue/{item_id}/resolve",
        tags=["Review Queue"],
        summary="Mark a dead letter as resolved with corrected SQL",
    )
    async def resolve_dead_letter(
        item_id: str,
        request: ResolveDeadLetterRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> dict[str, Any]:
        from core.audit_log import resolve_dead_letter as _resolve
        ok = await _resolve(item_id, request.corrected_sql)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Dead letter '{item_id}' not found")
        # Also add to few-shot store for auto-improvement
        try:
            from agents.tools.few_shot_tools import add_example_to_store
            add_example_to_store(
                question="",
                sql=request.corrected_sql,
                database_name="default",
                tags=["dead_letter_correction"],
                feedback_score=1.0,
            )
        except Exception:
            pass
        return {"status": "resolved", "id": item_id}

    # ---- Prometheus metrics -------------------------------------------

    @app.get(
        "/metrics",
        include_in_schema=False,
        tags=["System"],
        summary="Prometheus metrics endpoint",
    )
    async def prometheus_metrics():
        """Expose Prometheus metrics in text/plain format."""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from fastapi.responses import Response as FastAPIResponse
            return FastAPIResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            # Fallback: return basic stats as plain text
            from core.telemetry import get_metrics_snapshot
            snapshot = get_metrics_snapshot()
            lines = [f"# HELP text2sql_queries_total Total queries processed"]
            lines.append(f"# TYPE text2sql_queries_total counter")
            for k, v in snapshot.items():
                if isinstance(v, (int, float)):
                    safe_k = k.replace("-", "_").replace(" ", "_")
                    lines.append(f"text2sql_{safe_k} {v}")
            return JSONResponse({"raw": "\n".join(lines)})

    # ---- Rate limit stats ---------------------------------------------

    @app.get(
        "/api/v1/rate-limit",
        response_model=RateLimitStats,
        tags=["System"],
        summary="Rate limit usage for current user",
    )
    async def rate_limit_stats(
        request: Request,
        api_key: str | None = Depends(get_api_key),
    ) -> RateLimitStats:
        from core.rate_limiter import get_rate_limit_stats
        user_id = request.headers.get("X-User-ID")
        stats = get_rate_limit_stats(user_id=user_id)
        return RateLimitStats(**stats)

    # ---- Saved Queries ------------------------------------------------

    @app.get(
        "/api/v1/saved-queries",
        response_model=list[SavedQuery],
        tags=["Saved Queries"],
        summary="List saved/named query templates",
    )
    async def list_saved_queries_endpoint(
        request: Request,
        tag: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        api_key: str | None = Depends(get_api_key),
    ) -> list[SavedQuery]:
        from core.saved_queries import list_saved_queries, extract_params
        user_id = request.headers.get("X-User-ID", "default")
        tenant_id = request.headers.get(settings.tenant.tenant_id_header, "default")
        items = await list_saved_queries(user_id=user_id, tenant_id=tenant_id, limit=limit, tag=tag)
        return [
            SavedQuery(
                **{k: v for k, v in item.items() if k in SavedQuery.model_fields},
                params=extract_params(item.get("nl_query", "")),
            )
            for item in items
        ]

    @app.post(
        "/api/v1/saved-queries",
        response_model=SavedQuery,
        tags=["Saved Queries"],
        summary="Save a named query template",
        status_code=201,
    )
    async def create_saved_query(
        request: Request,
        body: SavedQueryRequest,
        api_key: str | None = Depends(get_api_key),
    ) -> SavedQuery:
        from core.saved_queries import save_query, get_saved_query, extract_params
        user_id = request.headers.get("X-User-ID", "default")
        tenant_id = request.headers.get(settings.tenant.tenant_id_header, "default")
        await save_query(
            name=body.name,
            nl_query=body.nl_query,
            description=body.description,
            user_id=user_id,
            tenant_id=tenant_id,
            tags=body.tags,
        )
        item = await get_saved_query(body.name, user_id=user_id, tenant_id=tenant_id)
        if not item:
            raise HTTPException(status_code=500, detail="Failed to retrieve saved query after creation")
        return SavedQuery(
            **{k: v for k, v in item.items() if k in SavedQuery.model_fields},
            params=extract_params(item.get("nl_query", "")),
        )

    @app.post(
        "/api/v1/saved-queries/{name}/run",
        response_model=QueryResponse,
        tags=["Saved Queries"],
        summary="Execute a saved query with parameter substitution",
    )
    async def run_saved_query(
        name: str,
        body: SavedQueryRunRequest,
        request: Request,
        api_key: str | None = Depends(get_api_key),
    ) -> QueryResponse:
        from core.saved_queries import get_saved_query, populate_params
        from agents.runner import run_prism_query
        user_id = request.headers.get("X-User-ID", "default")
        tenant_id = request.headers.get(settings.tenant.tenant_id_header, "default")

        saved = await get_saved_query(name, user_id=user_id, tenant_id=tenant_id)
        if not saved:
            raise HTTPException(status_code=404, detail=f"Saved query '{name}' not found")

        nl_query = populate_params(saved["nl_query"], body.params)
        session_id = body.session_id or str(uuid.uuid4())
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        result = await run_prism_query(
            query=nl_query,
            session_id=session_id,
            correlation_id=correlation_id,
            tenant_id=tenant_id,
            max_rows=body.max_rows,
        )
        return QueryResponse(
            success=result.get("success", False),
            session_id=session_id,
            query=nl_query,
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
            pipeline_time_ms=result.get("pipeline_time_ms", 0.0),
            error=result.get("error"),
            suggestions=result.get("suggestions", []),
            needs_clarification=result.get("needs_clarification", False),
            cache_hit=result.get("cache_hit", False),
            pii_report=result.get("pii_report"),
            anomalies=result.get("anomalies", []),
            correlation_id=correlation_id,
            token_input=result.get("token_input"),
            token_output=result.get("token_output"),
        )

    @app.delete(
        "/api/v1/saved-queries/{name}",
        tags=["Saved Queries"],
        summary="Delete a saved query",
    )
    async def delete_saved_query_endpoint(
        name: str,
        request: Request,
        api_key: str | None = Depends(get_api_key),
    ) -> dict[str, Any]:
        from core.saved_queries import delete_saved_query
        user_id = request.headers.get("X-User-ID", "default")
        tenant_id = request.headers.get(settings.tenant.tenant_id_header, "default")
        deleted = await delete_saved_query(name, user_id=user_id, tenant_id=tenant_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Saved query '{name}' not found")
        return {"status": "deleted", "name": name}

    # ---- Result Export ------------------------------------------------

    @app.get(
        "/api/v1/history/{query_id}/export",
        tags=["History"],
        summary="Export query results as CSV, JSON, or Excel",
    )
    async def export_query_results(
        query_id: str,
        format: ExportFormat = Query(default=ExportFormat.json),
        api_key: str | None = Depends(get_api_key),
    ):
        from core.audit_log import get_query_history

        # Fetch from audit log (only metadata stored, not rows)
        rows_data = await get_query_history(limit=1, offset=0)
        # For a full export we'd need to re-execute; return the audit record
        # In production, rows should be cached with the query_id
        history = [r for r in rows_data if r.get("query_id") == query_id]
        if not history:
            raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found in history")

        record = history[0]

        if format == ExportFormat.json:
            return JSONResponse(content=record)

        if format == ExportFormat.csv:
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerow(record)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=query_{query_id}.csv"},
            )

        if format == ExportFormat.excel:
            try:
                import openpyxl
                import io
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.append(list(record.keys()))
                ws.append([str(v) for v in record.values()])
                buf = io.BytesIO()
                wb.save(buf)
                buf.seek(0)
                return StreamingResponse(
                    buf,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename=query_{query_id}.xlsx"},
                )
            except ImportError:
                raise HTTPException(status_code=422, detail="openpyxl not installed. Use csv or json format.")

    # ---- Performance regression report --------------------------------

    @app.get(
        "/api/v1/performance/slow-queries",
        response_model=list[PerformanceBaseline],
        tags=["System"],
        summary="Slowest query patterns by p95 latency",
    )
    async def slow_query_report(
        limit: int = Query(default=20, ge=1, le=100),
        api_key: str | None = Depends(get_api_key),
    ) -> list[PerformanceBaseline]:
        from core.performance_tracker import get_slow_query_report
        rows = await get_slow_query_report(limit=limit)
        return [PerformanceBaseline(**r) for r in rows]

    # ---- Metrics -------------------------------------------------------

    @app.get("/api/v1/metrics", tags=["System"], summary="Telemetry + cache + feedback metrics")
    async def metrics(api_key: str | None = Depends(get_api_key)) -> dict[str, Any]:
        from core.telemetry import get_metrics_snapshot
        from core.semantic_cache import get_cache_stats
        from agents.tools.feedback_tools import get_feedback_stats

        index_count = 0
        try:
            from core.schema_index import get_schema_index
            index_count = get_schema_index().get_indexed_count()
        except Exception:
            pass

        telemetry = get_metrics_snapshot()

        cache_stats = {}
        try:
            cache_stats = get_cache_stats()
        except Exception:
            pass

        feedback_stats = {}
        try:
            feedback_stats = get_feedback_stats()
        except Exception:
            pass

        calibration_stats = {}
        try:
            from core.audit_log import get_calibration_stats
            calibration_stats = await get_calibration_stats()
        except Exception:
            pass

        return {
            "service": "text2sql-prism",
            "version": "1.0.0",
            "schema_index_size": index_count,
            "telemetry": telemetry,
            "cache": cache_stats,
            "feedback": feedback_stats,
            "calibration": calibration_stats,
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
