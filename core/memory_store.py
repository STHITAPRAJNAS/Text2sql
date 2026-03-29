"""
ADK Memory Store
================
Configures the Google ADK MemoryService — a semantic vector store that
persists successful query→SQL mappings across sessions so agents can
self-improve over time.

How agent self-improvement works:
1. After a successful query (confidence ≥ threshold OR user rates ≥ 4/5),
   the runner calls `add_session_to_memory(session)` which extracts all
   events from that session and stores them in the memory service.
2. The SQL Generator Agent has `LoadMemoryTool` in its tools list.
   Before generating SQL, it calls `load_memory(query)` to retrieve
   semantically similar past query→SQL pairs as few-shot context.
3. Over time the model learns from real production queries in the same
   catalog/schema, outperforming generic few-shot examples.

Supported backends:
  in_memory    (dev)   — keyword-based, resets on restart
  vertex_ai    (prod)  — Vertex AI Memory Bank (persistent, semantic search)

Usage:
    from core.memory_store import get_memory_service, add_session_to_memory
    memory_service = get_memory_service()
    await add_session_to_memory(session)
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_memory_service = None


def get_memory_service():
    """
    Return the singleton ADK memory service.

    Chooses backend based on settings.memory.backend:
      - "in_memory"  → InMemoryMemoryService (default, resets on restart)
      - "vertex_ai"  → VertexAiMemoryBankService (needs agent_engine_id)
    """
    global _memory_service
    if _memory_service is not None:
        return _memory_service

    from config.settings import get_settings
    settings = get_settings()

    backend = settings.memory.backend

    try:
        if backend == "vertex_ai":
            _memory_service = _build_vertex_memory(settings)
        else:
            _memory_service = _build_in_memory()
    except ImportError:
        logger.warning("google-adk not installed — memory service unavailable")
        _memory_service = None

    return _memory_service


def _build_in_memory():
    from google.adk.memory import InMemoryMemoryService
    svc = InMemoryMemoryService()
    logger.info("ADK InMemoryMemoryService initialized")
    return svc


def _build_vertex_memory(settings):
    from google.adk.memory import VertexAiMemoryBankService
    engine_id = settings.memory.agent_engine_id
    project = settings.google_cloud_project
    location = settings.google_cloud_location
    if not engine_id:
        logger.warning(
            "MEMORY_AGENT_ENGINE_ID not set — falling back to InMemoryMemoryService"
        )
        return _build_in_memory()
    svc = VertexAiMemoryBankService(
        project=project,
        location=location,
        agent_engine_id=engine_id,
    )
    logger.info(
        "ADK VertexAiMemoryBankService initialized",
        project=project,
        location=location,
        engine_id=engine_id,
    )
    return svc


async def add_session_to_memory(session: Any) -> bool:
    """
    Add all events from a completed session to the memory service.

    Called by the runner after a successful query execution.
    The memory service extracts meaningful query→answer pairs
    that future agents can retrieve via load_memory.

    Returns True if successfully added, False otherwise.
    """
    svc = get_memory_service()
    if svc is None:
        return False
    try:
        await svc.add_session_to_memory(session)
        logger.debug("Session added to memory store", session_id=getattr(session, "id", "?"))
        return True
    except Exception as exc:
        logger.warning("Failed to add session to memory", error=str(exc))
        return False


def get_memory_service_uri() -> str | None:
    """
    Return the memory_service_uri for get_fast_api_app().

    Only Vertex AI Memory Bank has a URI format:
      "agentengine://<agent_engine_id>"

    Returns None for in_memory (ADK handles it internally).
    """
    from config.settings import get_settings
    settings = get_settings()

    if settings.memory.backend == "vertex_ai" and settings.memory.agent_engine_id:
        return f"agentengine://{settings.memory.agent_engine_id}"
    return None
