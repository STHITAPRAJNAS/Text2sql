"""
PRISM ADK Runner
================
Bridges the FastAPI layer and the Google ADK multi-agent system.

Handles:
- ADK session management (DatabaseSessionService for persistence)
- Semantic cache check (Redis L1 + ChromaDB L2) before pipeline
- Agent invocation and response parsing
- Clarification detection (pipeline exits early via request_clarification)
- Memory persistence after successful queries (add_session_to_memory)
- OpenTelemetry tracing per PRISM phase
- Error recovery and fallback
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import structlog
from config.settings import get_settings

try:
    from google.genai import types as genai_types
    from agents.orchestrator import get_orchestrator
except ImportError:
    genai_types = None  # type: ignore
    get_orchestrator = None  # type: ignore
from core.telemetry import trace_phase, record_query, get_metrics_snapshot

logger = structlog.get_logger(__name__)

APP_NAME = "text2sql_prism"

# Runner singleton
_runner = None


def _get_runner():
    """Get or create the ADK Runner singleton (InMemoryRunner or DatabaseSessionService-backed)."""
    global _runner
    if _runner is not None:
        return _runner

    orchestrator = get_orchestrator()

    # Try to use Runner with DatabaseSessionService for persistent sessions
    try:
        from google.adk.runners import Runner
        from core.session_store import get_session_service
        session_service = get_session_service()
        if session_service is not None:
            _runner = Runner(
                agent=orchestrator,
                app_name=APP_NAME,
                session_service=session_service,
            )
            logger.info("ADK Runner initialized with DatabaseSessionService", app=APP_NAME)
            return _runner
    except (ImportError, Exception) as e:
        logger.debug("Runner with session service failed, trying InMemoryRunner", error=str(e))

    # Fallback to InMemoryRunner
    try:
        from google.adk.runners import InMemoryRunner
        _runner = InMemoryRunner(agent=orchestrator, app_name=APP_NAME)
        logger.info("ADK InMemoryRunner initialized", app=APP_NAME)
    except ImportError as e:
        logger.error("google-adk not installed", error=str(e))
        raise

    return _runner


def _extract_sql_from_response(text: str) -> str | None:
    """Extract SQL from LLM response text (handles markdown code blocks)."""
    sql_block = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if sql_block:
        return sql_block.group(1).strip()

    code_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        candidate = code_block.group(1).strip()
        if candidate.upper().startswith(("SELECT", "WITH")):
            return candidate

    select_match = re.search(r"((?:WITH|SELECT)\s+.+?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()

    return None


def _extract_json_from_response(text: str) -> dict[str, Any] | None:
    """Extract JSON object from LLM response text."""
    json_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _check_clarification_in_response(text: str) -> dict[str, Any] | None:
    """Detect if the response contains a clarification request."""
    # Structural JSON clarification
    json_data = _extract_json_from_response(text)
    if json_data and json_data.get("needs_clarification"):
        return json_data

    # Pattern match for clarification signal
    if re.search(r"needs_clarification.*?true|clarification.*?needed", text, re.IGNORECASE | re.DOTALL):
        question_match = re.search(
            r'"question"\s*:\s*"([^"]+)"', text, re.IGNORECASE
        )
        question = question_match.group(1) if question_match else text[:200]
        return {
            "needs_clarification": True,
            "question": question,
            "options": [],
            "ambiguities": [],
        }

    return None


def _parse_agent_response(response_text: str) -> dict[str, Any]:
    """Parse the agent's final response text into a structured result."""
    result: dict[str, Any] = {
        "success": False,
        "generated_sql": None,
        "optimized_sql": None,
        "answer": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "confidence": 0.0,
        "pipeline_stages": [],
        "execution_time_ms": 0.0,
        "error": None,
        "suggestions": [],
        "needs_clarification": False,
        "clarification_question": None,
        "clarification_options": [],
        "cost_warning": None,
    }

    # Check for clarification first
    clarification = _check_clarification_in_response(response_text)
    if clarification:
        result["needs_clarification"] = True
        result["clarification_question"] = clarification.get("question")
        result["clarification_options"] = clarification.get("options", [])
        result["success"] = False
        return result

    # Extract SQL
    sql = _extract_sql_from_response(response_text)
    if sql:
        result["generated_sql"] = sql

    # Extract structured JSON result
    json_data = _extract_json_from_response(response_text)
    if json_data:
        result.update({
            "generated_sql": json_data.get("sql", sql),
            "optimized_sql": json_data.get("optimized_sql"),
            "columns": json_data.get("columns", []),
            "rows": json_data.get("rows", []),
            "row_count": json_data.get("row_count", 0),
            "truncated": json_data.get("truncated", False),
            "confidence": json_data.get("confidence", 0.0),
            "answer": json_data.get("answer") or json_data.get("summary"),
            "execution_time_ms": json_data.get("execution_time_ms", 0.0),
            "cost_warning": json_data.get("cost_warning"),
        })

    # Extract natural language answer
    if not result["answer"]:
        answer_match = re.search(
            r"(?:Answer|Summary|Result):\s*(.+?)(?:\n\n|\Z)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        else:
            paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]
            for para in paragraphs:
                if not para.startswith(("SELECT", "WITH", "{", "```")):
                    result["answer"] = para[:500]
                    break

    result["success"] = bool(result["generated_sql"])
    return result


async def run_prism_query(
    query: str,
    session_id: str | None = None,
    database_name: str = "default",
    max_rows: int = 100,
    execute_query: bool = True,
) -> dict[str, Any]:
    """
    Run a natural language query through the full PRISM pipeline.

    Flow:
    1. Check semantic cache (L1 Redis + L2 ChromaDB) — return immediately on hit
    2. Create/retrieve ADK session
    3. Invoke PRISM orchestrator
    4. Detect clarification requests
    5. On success: cache result, record metrics, record_query telemetry

    Args:
        query: Natural language question
        session_id: Session ID for conversation continuity
        database_name: Target database name
        max_rows: Maximum result rows
        execute_query: Whether to execute the generated SQL

    Returns:
        Structured result dict with SQL, results, answer, cache info, and metadata
    """
    settings = get_settings()
    session_id = session_id or str(uuid.uuid4())
    pipeline_start = time.monotonic()

    logger.info(
        "Running PRISM query",
        session_id=session_id,
        query_preview=query[:100],
        database=database_name,
    )

    # ------------------------------------------------------------------ #
    # 1. Semantic cache check (skip if execute_query=False or clarifying) #
    # ------------------------------------------------------------------ #
    if execute_query and settings.semantic_cache.enable_l1_cache or settings.semantic_cache.enable_l2_cache:
        try:
            from core.semantic_cache import get_cached_result
            cached, cache_source = await get_cached_result(query, database_name)
            if cached is not None:
                elapsed = (time.monotonic() - pipeline_start) * 1000
                record_query(f"cache_hit_{cache_source.split('_')[0]}")
                cached["session_id"] = session_id
                cached["pipeline_time_ms"] = round(elapsed, 2)
                cached["cache_hit"] = True
                cached["cache_source"] = cache_source
                logger.info(
                    "Cache hit",
                    source=cache_source,
                    session_id=session_id,
                    ms=round(elapsed, 2),
                )
                return cached
        except Exception as exc:
            logger.warning("Cache check failed, proceeding to pipeline", error=str(exc))

    # ------------------------------------------------------------------ #
    # 2. Run PRISM pipeline                                               #
    # ------------------------------------------------------------------ #
    try:
        runner = _get_runner()

        user_message = _build_user_message(
            query=query,
            database_name=database_name,
            max_rows=max_rows,
            execute_query=execute_query,
        )

        response_text = ""
        with trace_phase("total", {"query": query[:80], "database": database_name}):
            async for event in runner.run_async(
                user_id=session_id,
                session_id=session_id,
                new_message=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_message)],
                ),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text

        pipeline_time_ms = (time.monotonic() - pipeline_start) * 1000

        # Parse response
        result = _parse_agent_response(response_text)
        result["pipeline_time_ms"] = round(pipeline_time_ms, 2)
        result["session_id"] = session_id
        result["cache_hit"] = False
        result["pipeline_stages"] = [
            "schema_discovery",
            "metadata_enrichment",
            "deep_think_analysis",
            "schema_linking",
            "sql_generation",
            "sql_validation",
            "query_optimization",
            "response_formatting",
        ]

        # Record telemetry
        if result.get("needs_clarification"):
            record_query("clarification_needed")
        elif result["success"]:
            record_query("success")
            # Store in cache for future hits
            if execute_query:
                try:
                    from core.semantic_cache import cache_result
                    await cache_result(query, database_name, result)
                except Exception as exc:
                    logger.debug("Cache write failed", error=str(exc))
        else:
            record_query("failed")

        logger.info(
            "PRISM query complete",
            session_id=session_id,
            success=result["success"],
            clarification=result.get("needs_clarification", False),
            pipeline_time_ms=round(pipeline_time_ms, 2),
        )

        return result

    except Exception as e:
        pipeline_time_ms = (time.monotonic() - pipeline_start) * 1000
        record_query("failed")
        logger.error(
            "PRISM pipeline error",
            session_id=session_id,
            error=str(e),
            pipeline_time_ms=round(pipeline_time_ms, 2),
        )
        return {
            "success": False,
            "session_id": session_id,
            "generated_sql": None,
            "optimized_sql": None,
            "answer": None,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "confidence": 0.0,
            "pipeline_stages": [],
            "execution_time_ms": 0.0,
            "pipeline_time_ms": round(pipeline_time_ms, 2),
            "error": str(e),
            "needs_clarification": False,
            "clarification_question": None,
            "clarification_options": [],
            "cache_hit": False,
            "suggestions": [
                "Check your database connection",
                "Verify GOOGLE_API_KEY is set correctly",
                "Ensure the query is a valid question about the database",
            ],
        }


def _build_user_message(
    query: str,
    database_name: str,
    max_rows: int,
    execute_query: bool,
) -> str:
    """Build the full user message for the orchestrator."""
    settings = get_settings()
    db_url = settings.database.database_url
    dialect = db_url.split("+")[0].split(":")[0] if ":" in db_url else "sql"

    # Detect Databricks for Spark SQL dialect hint
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

Please run the full PRISM pipeline (P→R→I→S→M) to answer this question:
1. **Phase P**: Discover the schema (vector search first), check for schema changes
2. **Phase R**: Apply Deep Think reasoning — request clarification if confidence < {settings.deep_think.deep_think_confidence_threshold}
3. **Phase I**: Generate accurate, dialect-correct SQL (use semantic memory for similar past queries)
4. **Phase S**: Validate (syntax/schema/security/performance), estimate cost, optimize
5. **Phase M**: Execute the SQL and provide a clear answer

Return:
- The final SQL query
- {"Query execution results" if execute_query else "Only the SQL (do not execute)"}
- A natural language answer summarizing the key finding
- Confidence score (0.0-1.0)
- Any cost warnings if the query scans > {settings.feedback.cost_warn_gb} GB
"""
