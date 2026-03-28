"""
PRISM ADK Runner
Bridges the FastAPI layer and the Google ADK multi-agent system.

Handles:
- ADK session management
- Agent invocation and response parsing
- Pipeline state tracking
- Error recovery and fallback
- Observability (tracing, logging)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import structlog
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from config.settings import get_settings
from agents.orchestrator import get_orchestrator

logger = structlog.get_logger(__name__)

# Session service (in-memory for single-node; use Redis-backed for multi-node)
_session_service = InMemorySessionService()

# Runner singleton
_runner: Runner | None = None

APP_NAME = "text2sql_prism"


def _get_runner() -> Runner:
    """Get or create the ADK Runner singleton."""
    global _runner
    if _runner is None:
        orchestrator = get_orchestrator()
        _runner = Runner(
            agent=orchestrator,
            app_name=APP_NAME,
            session_service=_session_service,
        )
        logger.info("ADK Runner initialized", app=APP_NAME)
    return _runner


def _extract_sql_from_response(text: str) -> str | None:
    """Extract SQL from LLM response text (handles markdown code blocks)."""
    # Try ```sql ... ``` blocks first
    sql_block = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if sql_block:
        return sql_block.group(1).strip()

    # Try ``` ... ``` blocks
    code_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        candidate = code_block.group(1).strip()
        if candidate.upper().startswith(("SELECT", "WITH")):
            return candidate

    # Try bare SELECT/WITH statements
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

    # Try inline JSON
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_agent_response(response_text: str) -> dict[str, Any]:
    """
    Parse the agent's final response text into a structured result.
    Extracts SQL, execution results, confidence, and answer.
    """
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
    }

    # Extract SQL
    sql = _extract_sql_from_response(response_text)
    if sql:
        result["generated_sql"] = sql

    # Extract structured JSON result if present
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
        })

    # Extract natural language answer from text
    if not result["answer"]:
        # Look for answer/summary patterns
        answer_match = re.search(
            r"(?:Answer|Summary|Result):\s*(.+?)(?:\n\n|\Z)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        else:
            # Use first paragraph as answer
            paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]
            if paragraphs:
                # Skip paragraphs that look like SQL or JSON
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

    This is the main entry point called by the FastAPI layer.
    It:
    1. Creates or retrieves an ADK session
    2. Builds the user message with context
    3. Invokes the PRISM orchestrator via the ADK Runner
    4. Parses and returns the structured response

    Args:
        query: Natural language question
        session_id: Session ID for conversation continuity
        database_name: Target database name
        max_rows: Maximum result rows
        execute_query: Whether to execute the generated SQL

    Returns:
        Structured result dict with SQL, results, answer, and metadata
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

    try:
        runner = _get_runner()

        # Get or create ADK session
        session = await _session_service.get_session(
            app_name=APP_NAME,
            user_id=session_id,
            session_id=session_id,
        )
        if session is None:
            session = await _session_service.create_session(
                app_name=APP_NAME,
                user_id=session_id,
                session_id=session_id,
                state={
                    "database_name": database_name,
                    "max_rows": max_rows,
                    "execute_query": execute_query,
                },
            )

        # Build the user message
        user_message = _build_user_message(
            query=query,
            database_name=database_name,
            max_rows=max_rows,
            execute_query=execute_query,
        )

        # Invoke the PRISM orchestrator
        response_text = ""
        async for event in runner.run_async(
            user_id=session_id,
            session_id=session_id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            ),
        ):
            # Collect the final response
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text

        pipeline_time_ms = (time.monotonic() - pipeline_start) * 1000

        # Parse the response
        result = _parse_agent_response(response_text)
        result["pipeline_time_ms"] = round(pipeline_time_ms, 2)
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

        logger.info(
            "PRISM query complete",
            session_id=session_id,
            success=result["success"],
            pipeline_time_ms=round(pipeline_time_ms, 2),
        )

        return result

    except Exception as e:
        pipeline_time_ms = (time.monotonic() - pipeline_start) * 1000
        logger.error(
            "PRISM pipeline error",
            session_id=session_id,
            error=str(e),
            pipeline_time_ms=round(pipeline_time_ms, 2),
        )
        return {
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
            "pipeline_time_ms": round(pipeline_time_ms, 2),
            "error": str(e),
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
    return f"""## Text2SQL PRISM Request

**User Question:** {query}

**Configuration:**
- Database: {database_name}
- SQL Dialect: {settings.database.database_url.split("+")[0].split(":")[0]}
- Max Rows: {max_rows}
- Execute Query: {execute_query}
- Deep Think Iterations: {settings.deep_think.deep_think_max_iterations}

Please run the full PRISM pipeline (P→R→I→S→M) to answer this question:
1. **Phase P**: Discover the schema and enrich with business metadata
2. **Phase R**: Apply Deep Think reasoning to analyze the query
3. **Phase I**: Generate accurate, dialect-correct SQL
4. **Phase S**: Validate (syntax/schema/security/performance) and optimize
5. **Phase M**: Execute the SQL and provide a clear answer

Return:
- The final SQL query
- {"Query execution results" if execute_query else "Only the SQL (do not execute)"}
- A natural language answer summarizing the key finding
- Confidence score (0.0-1.0)
"""
