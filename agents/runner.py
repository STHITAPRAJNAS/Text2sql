"""
PRISM ADK Runner
================
Bridges the FastAPI layer and the Google ADK multi-agent system.

Latency-safe design:
  - Semantic cache checked FIRST (L1 Redis + L2 ChromaDB) — skip pipeline on hit
  - Audit log writes are fire-and-forget (asyncio.create_task)
  - PII scan happens inside response_formatter (< 2ms)
  - Glossary lookups are in-memory O(1) in Phase R
  - Auto-retry fires only when SQL execution fails (rare path, ~10% worst case)

Features:
  - ADK DatabaseSessionService for persistent multi-turn sessions
  - Semantic cache (Redis L1 + ChromaDB L2) before pipeline
  - Auto-retry on SQL execution error (max 2 retries, error fed back to generator)
  - Audit log (fire-and-forget to SQLite/PostgreSQL)
  - Dead letter queue for persistent failures
  - Clarification detection
  - Memory persistence via after_agent_callback
  - OpenTelemetry per-phase spans
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, AsyncGenerator

import structlog

from config.settings import get_settings
from core.telemetry import trace_phase, record_query

try:
    from google.genai import types as genai_types
    from agents.orchestrator import get_orchestrator
except ImportError:
    genai_types = None  # type: ignore
    get_orchestrator = None  # type: ignore

logger = structlog.get_logger(__name__)

APP_NAME = "text2sql_prism"
_runner = None


def _get_runner():
    global _runner
    if _runner is not None:
        return _runner

    orchestrator = get_orchestrator()

    try:
        from google.adk.runners import Runner
        from core.session_store import get_session_service
        session_service = get_session_service()
        if session_service is not None:
            _runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)
            logger.info("ADK Runner initialized with DatabaseSessionService", app=APP_NAME)
            return _runner
    except (ImportError, Exception) as e:
        logger.debug("Runner with session service failed", error=str(e))

    from google.adk.runners import InMemoryRunner
    _runner = InMemoryRunner(agent=orchestrator, app_name=APP_NAME)
    logger.info("ADK InMemoryRunner initialized", app=APP_NAME)
    return _runner


# ------------------------------------------------------------------ #
# Parsing helpers                                                      #
# ------------------------------------------------------------------ #

def _extract_sql(text: str) -> str | None:
    m = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        c = m.group(1).strip()
        if c.upper().startswith(("SELECT", "WITH")):
            return c
    m = re.search(r"((?:WITH|SELECT)\s+.+?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _check_clarification(text: str) -> dict[str, Any] | None:
    j = _extract_json(text)
    if j and j.get("needs_clarification"):
        return j
    if re.search(r"needs_clarification.*?true|clarification.*?needed", text, re.IGNORECASE | re.DOTALL):
        q = re.search(r'"question"\s*:\s*"([^"]+)"', text)
        return {
            "needs_clarification": True,
            "question": q.group(1) if q else text[:200],
            "options": [],
            "ambiguities": [],
        }
    return None


def _parse_response(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False, "generated_sql": None, "optimized_sql": None,
        "answer": None, "columns": [], "rows": [], "row_count": 0,
        "truncated": False, "confidence": 0.0, "pipeline_stages": [],
        "execution_time_ms": 0.0, "error": None, "suggestions": [],
        "needs_clarification": False, "clarification_question": None,
        "clarification_options": [], "cost_warning": None,
        "pii_report": None, "anomalies": [], "execution_error": None,
    }

    clarification = _check_clarification(text)
    if clarification:
        result["needs_clarification"] = True
        result["clarification_question"] = clarification.get("question")
        result["clarification_options"] = clarification.get("options", [])
        return result

    sql = _extract_sql(text)
    if sql:
        result["generated_sql"] = sql

    j = _extract_json(text)
    if j:
        result.update({
            "generated_sql": j.get("sql", sql),
            "optimized_sql": j.get("optimized_sql"),
            "columns": j.get("columns", []),
            "rows": j.get("rows", []),
            "row_count": j.get("row_count", 0),
            "truncated": j.get("truncated", False),
            "confidence": j.get("confidence", 0.0),
            "answer": j.get("answer") or j.get("summary"),
            "execution_time_ms": j.get("execution_time_ms", 0.0),
            "cost_warning": j.get("cost_warning"),
            "pii_report": j.get("pii_report"),
            "anomalies": j.get("anomalies", []),
            "execution_error": j.get("execution_error"),
        })

    if not result["answer"]:
        m = re.search(r"(?:Answer|Summary|Result):\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m:
            result["answer"] = m.group(1).strip()
        else:
            for para in text.split("\n\n"):
                para = para.strip()
                if para and not para.startswith(("SELECT", "WITH", "{", "```")):
                    result["answer"] = para[:500]
                    break

    result["success"] = bool(result["generated_sql"])
    return result


# ------------------------------------------------------------------ #
# Core pipeline invocation                                             #
# ------------------------------------------------------------------ #

async def _run_pipeline_once(
    message: str,
    session_id: str,
    user_id: str,
) -> str:
    """Run one pass of the PRISM pipeline. Returns raw response text."""
    runner = _get_runner()
    text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
    return text


async def run_prism_query(
    query: str,
    session_id: str | None = None,
    database_name: str = "default",
    max_rows: int = 100,
    execute_query: bool = True,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Run a natural language query through the full PRISM pipeline.

    Flow:
      1. Semantic cache check (L1 Redis + L2 ChromaDB) — instant return on hit
      2. Run PRISM pipeline once
      3. On SQL execution error: auto-retry up to max_execution_retries times
         with error context fed back to the generator
      4. Fire-and-forget audit log write
      5. Dead letter queue write on persistent failure

    Args:
        query: Natural language question
        session_id: Conversation session ID
        database_name: Target database
        max_rows: Maximum result rows
        execute_query: Whether to execute the generated SQL
        user_id: Caller user ID for history tracking

    Returns:
        Structured result dict
    """
    settings = get_settings()
    session_id = session_id or str(uuid.uuid4())
    user_id = user_id or session_id
    pipeline_start = time.monotonic()

    logger.info("PRISM query", sid=session_id, preview=query[:80], db=database_name)

    # ------------------------------------------------------------------ #
    # 1. Semantic cache check                                              #
    # ------------------------------------------------------------------ #
    if execute_query:
        try:
            from core.semantic_cache import get_cached_result
            cached, cache_source = await get_cached_result(query, database_name)
            if cached is not None:
                elapsed = (time.monotonic() - pipeline_start) * 1000
                record_query(f"cache_hit_{cache_source.split('_')[0]}")
                cached.update({
                    "session_id": session_id, "pipeline_time_ms": round(elapsed, 2),
                    "cache_hit": True, "cache_source": cache_source,
                })
                _fire_audit(cached, session_id, user_id, query, database_name)
                logger.info("Cache hit", source=cache_source, ms=round(elapsed, 2))
                return cached
        except Exception as exc:
            logger.debug("Cache check failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # 2. Build user message                                                #
    # ------------------------------------------------------------------ #
    base_message = _build_message(query, database_name, max_rows, execute_query)

    # ------------------------------------------------------------------ #
    # 3. Run pipeline with auto-retry on execution error                   #
    # ------------------------------------------------------------------ #
    max_retries = getattr(settings.deep_think, "max_execution_retries", 2)
    result: dict[str, Any] = {}
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            message = base_message
            if attempt > 0 and last_error:
                message += (
                    f"\n\n⚠️ Previous attempt (#{attempt}) failed at SQL execution:\n"
                    f"Error: {last_error}\n"
                    f"Previously generated SQL:\n```sql\n{result.get('generated_sql', '')}\n```\n"
                    "Please generate corrected SQL that avoids this error. "
                    "Do NOT re-run schema discovery (use cached context). "
                    "Focus only on fixing Phase I (SQL generation) and Phase S (validation)."
                )
                logger.info("Auto-retry", attempt=attempt, error=last_error[:80])

            with trace_phase("total", {"query": query[:80], "database": database_name, "attempt": attempt}):
                response_text = await _run_pipeline_once(message, session_id, user_id)

            result = _parse_response(response_text)
            result.update({
                "pipeline_time_ms": round((time.monotonic() - pipeline_start) * 1000, 2),
                "session_id": session_id,
                "cache_hit": False,
                "pipeline_stages": [
                    "schema_discovery", "metadata_enrichment", "deep_think_analysis",
                    "schema_linking", "sql_generation", "sql_validation",
                    "query_optimization", "response_formatting",
                ],
            })

            # Check if Phase M reported an execution error
            exec_err = result.get("execution_error")
            if exec_err and attempt < max_retries:
                last_error = str(exec_err)
                continue  # Retry with error context

            break  # Success or clarification or non-retryable failure

        except Exception as e:
            last_error = str(e)
            if attempt == max_retries:
                result = _error_result(session_id, query, str(e), pipeline_start)

    # ------------------------------------------------------------------ #
    # 4. Record telemetry and cache successful results                     #
    # ------------------------------------------------------------------ #
    if result.get("needs_clarification"):
        record_query("clarification_needed")
    elif result.get("success"):
        record_query("success")
        if execute_query:
            try:
                from core.semantic_cache import cache_result
                await cache_result(query, database_name, result)
            except Exception:
                pass
    else:
        record_query("failed")
        # Write to dead letter queue for human review
        if not result.get("needs_clarification"):
            from core.audit_log import log_dead_letter
            log_dead_letter(
                nl_query=query,
                generated_sql=result.get("generated_sql"),
                error=result.get("error") or "unknown",
                failure_reason="pipeline_failure",
                database_name=database_name,
                confidence=result.get("confidence", 0.0),
            )

    # Fire-and-forget audit write
    _fire_audit(result, session_id, user_id, query, database_name)

    logger.info(
        "PRISM complete", sid=session_id,
        success=result.get("success"), ms=result.get("pipeline_time_ms"),
        clarification=result.get("needs_clarification", False),
        pii=bool(result.get("pii_report", {}) and result["pii_report"].get("pii_detected")),
    )
    return result


def _fire_audit(result: dict, session_id: str, user_id: str, query: str, database_name: str) -> None:
    """Fire-and-forget audit log write."""
    try:
        from core.audit_log import log_query
        log_query({
            **result,
            "nl_query": query,
            "database_name": database_name,
            "pii_detected": bool(
                result.get("pii_report") and result["pii_report"].get("pii_detected")
            ),
        }, session_id=session_id, user_id=user_id)
    except Exception:
        pass


def _error_result(session_id: str, query: str, error: str, start: float) -> dict[str, Any]:
    return {
        "success": False, "session_id": session_id, "generated_sql": None,
        "optimized_sql": None, "answer": None, "columns": [], "rows": [],
        "row_count": 0, "truncated": False, "confidence": 0.0,
        "pipeline_stages": [], "execution_time_ms": 0.0,
        "pipeline_time_ms": round((time.monotonic() - start) * 1000, 2),
        "error": error, "needs_clarification": False,
        "clarification_question": None, "clarification_options": [],
        "cache_hit": False, "pii_report": None, "anomalies": [],
        "suggestions": [
            "Check your database connection",
            "Verify GOOGLE_API_KEY is set correctly",
            "Ensure the query is a valid question about the database",
        ],
    }


# ------------------------------------------------------------------ #
# SSE streaming generator                                              #
# ------------------------------------------------------------------ #

async def stream_prism_query(
    query: str,
    session_id: str | None = None,
    database_name: str = "default",
    max_rows: int = 100,
    execute_query: bool = True,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream PRISM pipeline events as Server-Sent Events (SSE).

    Yields JSON-encoded event strings suitable for text/event-stream responses.
    Each event: "data: {json}\\n\\n"

    Event types:
      {"type": "phase", "phase": "schema_discovery", "status": "running"}
      {"type": "token", "text": "..."}
      {"type": "sql", "sql": "SELECT ..."}
      {"type": "result", "data": {...}}   -- final complete result
      {"type": "error", "error": "..."}
      {"type": "done"}
    """
    settings = get_settings()
    session_id = session_id or str(uuid.uuid4())
    user_id = user_id or session_id

    # Check cache first — if hit, return immediately as a single result event
    if execute_query:
        try:
            from core.semantic_cache import get_cached_result
            cached, source = await get_cached_result(query, database_name)
            if cached is not None:
                cached.update({"session_id": session_id, "cache_hit": True, "cache_source": source})
                yield f"data: {json.dumps({'type': 'result', 'data': cached})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
        except Exception:
            pass

    message = _build_message(query, database_name, max_rows, execute_query)

    # Stream pipeline phases
    phase_events = [
        ("schema_discovery", "Phase P: discovering schema"),
        ("deep_think", "Phase R: deep think reasoning"),
        ("sql_generation", "Phase I: generating SQL"),
        ("validation", "Phase S: validating + optimizing"),
        ("formatting", "Phase M: formatting response"),
    ]
    phase_idx = 0

    try:
        runner = _get_runner()
        full_text = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=message)],
            ),
        ):
            # Emit phase progress events based on agent name
            agent_name = getattr(event, "author", "") or ""
            if phase_idx < len(phase_events):
                phase_key, phase_label = phase_events[phase_idx]
                if phase_key in agent_name.lower() or (phase_idx == 0 and agent_name):
                    yield f"data: {json.dumps({'type': 'phase', 'phase': phase_key, 'label': phase_label, 'status': 'running'})}\n\n"
                    phase_idx += 1

            # Stream text tokens
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        full_text += part.text
                        # Stream tokens for streaming clients
                        yield f"data: {json.dumps({'type': 'token', 'text': part.text})}\n\n"

            # Emit final result
            if event.is_final_response():
                result = _parse_response(full_text)
                result.update({
                    "session_id": session_id,
                    "cache_hit": False,
                    "pipeline_stages": [
                        "schema_discovery", "metadata_enrichment", "deep_think_analysis",
                        "schema_linking", "sql_generation", "sql_validation",
                        "query_optimization", "response_formatting",
                    ],
                })

                # Emit SQL as soon as it's available
                if result.get("generated_sql"):
                    yield f"data: {json.dumps({'type': 'sql', 'sql': result['generated_sql']})}\n\n"

                yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"

                # Background tasks
                _fire_audit(result, session_id, user_id, query, database_name)
                if result.get("success") and execute_query:
                    try:
                        from core.semantic_cache import cache_result
                        await cache_result(query, database_name, result)
                    except Exception:
                        pass

    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def _build_message(query: str, database_name: str, max_rows: int, execute_query: bool) -> str:
    settings = get_settings()
    db_url = settings.database.database_url
    dialect = db_url.split("+")[0].split(":")[0] if ":" in db_url else "sql"
    if settings.databricks.is_configured:
        dialect = "spark_sql (Databricks)"

    return f"""## Text2SQL PRISM Request

**User Question:** {query}

**Configuration:**
- Database: {database_name}
- SQL Dialect: {dialect}
- Max Rows: {max_rows}
- Execute Query: {execute_query}
- Deep Think Iterations: {settings.deep_think.deep_think_max_iterations}
- Confidence Threshold: {settings.deep_think.deep_think_confidence_threshold}

**Instructions:**
1. Phase P: Discover schema (vector search first), check for schema changes
2. Phase R: Resolve business terms via glossary FIRST, then Deep Think analysis.
   Request clarification if confidence < {settings.deep_think.deep_think_confidence_threshold}
3. Phase I: Generate SQL (check semantic memory for similar past queries first)
4. Phase S: Validate complexity budget, syntax, schema, security, performance.
   Estimate cost, check partition coverage.
5. Phase M: Execute SQL, call interpret_results for PII masking + data story,
   return structured response.

Return: SQL, execution results, natural language answer with key finding,
confidence score, PII report, cost warning (if applicable), anomalies detected.
"""
