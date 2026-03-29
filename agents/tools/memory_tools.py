"""
ADK Memory Tools
================
Tools for agent self-improvement via the ADK MemoryService.

The SQL Generator calls load_memory before generating SQL to retrieve
semantically similar past query→SQL pairs as dynamic few-shot context.

After a successful query, the runner calls add_session_to_memory so the
entire reasoning trace (schema discovery → deep think → SQL → validation)
is stored and searchable for future similar queries.

Note: The primary memory retrieval mechanism is ADK's built-in
LoadMemoryTool (added to the SQL generator's tools list). The tools
here provide additional programmatic access for storing and
inspecting memory entries.
"""
from __future__ import annotations

import json
import time
from typing import Any, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger = structlog.get_logger(__name__)


def store_successful_query(
    query: str,
    sql: str,
    database_name: str,
    confidence: float,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Manually store a successful query→SQL pair in the memory store.

    Called by the response formatter after successful SQL execution
    when confidence ≥ threshold. The ADK runner also calls
    add_session_to_memory automatically after each successful session.

    Args:
        query: Natural language question
        sql: The validated SQL query
        database_name: Target database/catalog
        confidence: Confidence score (0.0–1.0)
        tool_context: ADK ToolContext (optional, for session state access)

    Returns:
        {"status": "stored", "query": query[:50]} or {"status": "skipped", ...}
    """
    from config.settings import get_settings
    settings = get_settings()

    # Only store high-confidence results
    threshold = settings.deep_think.deep_think_confidence_threshold
    if confidence < threshold:
        return {
            "status": "skipped",
            "reason": f"confidence {confidence:.2f} below threshold {threshold:.2f}",
        }

    # Store in session state for runner to persist to memory service
    if tool_context is not None:
        try:
            existing = tool_context.state.get("successful_queries", [])
            if not isinstance(existing, list):
                existing = []
            existing.append({
                "query": query,
                "sql": sql,
                "database_name": database_name,
                "confidence": confidence,
                "stored_at": int(time.time()),
            })
            # Keep last 20 per session
            tool_context.state["successful_queries"] = existing[-20:]
        except Exception as exc:
            logger.warning("Failed to write to session state", error=str(exc))

    logger.info(
        "Successful query stored in session state",
        query=query[:60],
        confidence=confidence,
    )
    return {
        "status": "stored",
        "query": query[:60],
        "confidence": confidence,
    }


def get_memory_context(
    query: str,
    database_name: str,
    top_k: int = 3,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Retrieve similar past queries from session state as few-shot context.

    This supplements the ADK LoadMemoryTool (which searches persistent
    memory) with in-session examples stored during the current conversation.

    Args:
        query: Current natural language query
        database_name: Target database
        top_k: Maximum number of examples to return
        tool_context: ADK ToolContext for session state access

    Returns:
        {"examples": [...], "count": N} with past query→SQL pairs
    """
    examples = []

    # Pull from session state (current conversation history)
    if tool_context is not None:
        try:
            past = tool_context.state.get("successful_queries", [])
            db_past = [p for p in past if p.get("database_name") == database_name]
            # Simple relevance: prefer examples with overlapping words
            query_words = set(query.lower().split())
            scored = []
            for p in db_past:
                past_words = set(p.get("query", "").lower().split())
                overlap = len(query_words & past_words) / max(len(query_words), 1)
                scored.append((overlap, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            examples = [p for _, p in scored[:top_k]]
        except Exception as exc:
            logger.warning("Failed to read session state", error=str(exc))

    return {
        "examples": examples,
        "count": len(examples),
        "source": "session_state",
        "note": (
            "These are in-session examples. For cross-session memory, "
            "the LoadMemoryTool provides semantic search across all past sessions."
        ),
    }
