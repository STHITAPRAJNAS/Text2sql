"""
PRISM Orchestrator — The Master Coordinator
Enterprise Text2SQL Agent using Google ADK Multi-Agent Swarm

Architecture:
  PRISM = Pre-processing → Reasoning → Intent Mapping → SQL Synthesis → Monitoring

  ┌─────────────────────────────────────────────────────────────────────┐
  │                     PRISM ORCHESTRATOR                              │
  │                                                                     │
  │  Phase P: Pre-processing (ParallelAgent)                            │
  │    ├── Schema Discovery Agent  ──→ DB schema + change detection     │
  │    └── Metadata Enrichment Agent ─→ Business context + glossary    │
  │                                                                     │
  │  Phase R: Reasoning (SequentialAgent - Deep Think)                  │
  │    ├── Deep Think Query Analyzer ─→ CoT decomposition               │
  │    │                                 OR request_clarification       │
  │    └── Schema Linker Agent ───────→ Entity-to-schema mapping        │
  │                                                                     │
  │  Phase I: Intent → SQL (Agent + LoadMemoryTool)                     │
  │    └── SQL Generator Agent ───────→ Dialect-aware SQL               │
  │                                                                     │
  │  Phase S: Synthesis / Validation Loop (LoopAgent, max 3 iter)      │
  │    ├── SQL Validator Agent ───────→ Multi-layer validation          │
  │    └── Query Optimizer Agent ─────→ Cost-aware optimization         │
  │                                                                     │
  │  Phase M: Monitoring / Response (Agent)                             │
  │    └── Response Formatter Agent ──→ Execute + NL answer             │
  └─────────────────────────────────────────────────────────────────────┘

After-agent callback:
  after_agent_callback calls add_session_to_memory to persist the full
  reasoning trace for future self-improvement via LoadMemoryTool.
"""
from __future__ import annotations

import asyncio
from typing import Any

from google.adk.agents import Agent, LlmAgent, ParallelAgent, SequentialAgent, LoopAgent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.swarm.schema_agent import (
    create_schema_discovery_agent,
    create_metadata_enrichment_agent,
)
from agents.swarm.query_analyzer import (
    create_deep_think_query_analyzer,
    create_schema_linker_agent,
)
from agents.swarm.sql_generator import create_sql_generator_agent
from agents.swarm.sql_validator import create_sql_validator_agent
from agents.swarm.query_optimizer import create_query_optimizer_agent
from agents.swarm.response_formatter import create_response_formatter_agent


async def _add_session_to_memory_callback(callback_context: Any) -> None:
    """
    After-agent callback: persist the completed session to the ADK MemoryService,
    then record progressive learning signals from ADK session state.

    ADK session state keys written by learning tools during the pipeline:
      skill_tags            — set by classify_query_skills (Phase I)
      join_paths_used       — set by classify_query_skills after SQL generation
      predicted_confidence  — set by Deep Think query analyzer (Phase R)

    Learning signals recorded here (all fire-and-forget, zero latency):
      schema_expertise   — join path success/failure per skill type
      confidence_calibrator — predicted vs actual confidence per skill
    """
    # ── Primary: persist session to ADK MemoryService ──────────────────
    try:
        await callback_context.add_session_to_memory()
    except AttributeError:
        try:
            from core.memory_store import add_session_to_memory
            session = getattr(callback_context, "session", None)
            if session is not None:
                await add_session_to_memory(session)
        except Exception:
            pass
    except Exception:
        pass

    # ── Secondary: record progressive learning signals ─────────────────
    try:
        state = getattr(callback_context, "state", None) or {}
        skill_tags: list[str] = state.get("skill_tags", [])
        join_paths: list[str] = state.get("join_paths_used", [])
        predicted_conf: float = float(state.get("predicted_confidence", 0.0))

        if skill_tags:
            # Record join path expertise (with placeholder rating — real rating
            # comes from feedback, here we use confidence as a proxy)
            if join_paths:
                from core.schema_expertise import record_join_result
                # Fire-and-forget — use predicted_confidence as proxy rating (scale 0→5)
                asyncio.create_task(
                    _record_join_result_async(join_paths, skill_tags, predicted_conf * 5.0)
                )

            # Record confidence calibration (actual rating will be updated via
            # feedback endpoint; here we seed with the predicted value)
            if predicted_conf > 0:
                from core.confidence_calibrator import record_calibration
                # actual_rating unknown yet — seeded as None-equivalent (0)
                # The real calibration update happens in record_feedback()
                # This just records that a prediction was made at this confidence level
                asyncio.create_task(
                    _record_calibration_async(skill_tags, predicted_conf)
                )
    except Exception:
        pass  # Learning signals must never affect primary response


async def _record_join_result_async(
    join_paths: list[str],
    skill_tags: list[str],
    rating_proxy: float,
) -> None:
    """Async wrapper for fire-and-forget join expertise recording."""
    try:
        from core.schema_expertise import record_join_result
        record_join_result(join_paths, skill_tags, rating_proxy)
    except Exception:
        pass


async def _record_calibration_async(
    skill_tags: list[str],
    predicted_conf: float,
) -> None:
    """Async wrapper for fire-and-forget calibration recording."""
    try:
        from core.confidence_calibrator import record_calibration
        # 0.0 actual means "pending feedback" — feedback endpoint updates real value
        record_calibration(skill_tags, predicted_conf, actual_rating=0.0)
    except Exception:
        pass


def create_prism_orchestrator() -> Agent:
    """
    Build and return the full PRISM Text2SQL orchestrator.

    This assembles the complete multi-agent swarm using Google ADK's
    composite agent types:
    - ParallelAgent: Runs schema discovery and metadata enrichment simultaneously
    - SequentialAgent: Chains the reasoning phases in order
    - LoopAgent: Iteratively refines SQL through validation + optimization
    - Agent: The root orchestrator that coordinates all phases

    Returns:
        The root PRISM orchestrator Agent ready for use with a Runner.
    """
    settings = get_settings()

    # ------------------------------------------------------------------ #
    # Phase P: Pre-processing (runs in parallel)                          #
    # ------------------------------------------------------------------ #
    phase_p_preprocessing = ParallelAgent(
        name="phase_p_preprocessing",
        description=(
            "Phase P: Simultaneously discovers database schema and enriches it with "
            "business metadata. Schema agent auto-detects Delta table changes. "
            "Both agents run in parallel to minimize latency."
        ),
        sub_agents=[
            create_schema_discovery_agent(),
            create_metadata_enrichment_agent(),
        ],
    )

    # ------------------------------------------------------------------ #
    # Phase R: Reasoning — Deep Think (sequential chain-of-thought)       #
    # ------------------------------------------------------------------ #
    phase_r_deep_think = SequentialAgent(
        name="phase_r_deep_think_reasoning",
        description=(
            "Phase R: Deep Think sequential reasoning pipeline. "
            "First decomposes the query with chain-of-thought analysis "
            "(calls request_clarification if confidence < threshold), "
            "then precisely links entities to schema elements."
        ),
        sub_agents=[
            create_deep_think_query_analyzer(),
            create_schema_linker_agent(),
        ],
    )

    # ------------------------------------------------------------------ #
    # Phase I: Intent → SQL Generation                                     #
    # ------------------------------------------------------------------ #
    phase_i_sql_generation = create_sql_generator_agent()

    # ------------------------------------------------------------------ #
    # Phase S: Synthesis — Validation + Optimization Loop                 #
    # The LoopAgent iterates up to max_iterations times.                   #
    # The loop exits when the validator calls exit_validation_loop or      #
    # the query analyzer calls request_clarification (escalate=True).     #
    # ------------------------------------------------------------------ #
    phase_s_validation_loop = LoopAgent(
        name="phase_s_validation_optimization_loop",
        description=(
            "Phase S: Iterative SQL validation and cost-aware optimization loop. "
            "Validates across 4 layers (syntax, schema, security, performance), "
            "estimates scan cost and partition coverage, then optimizes. "
            "Repeats up to max_iterations times."
        ),
        sub_agents=[
            create_sql_validator_agent(),
            create_query_optimizer_agent(),
        ],
        max_iterations=settings.deep_think.deep_think_max_iterations,
    )

    # ------------------------------------------------------------------ #
    # Phase M: Monitoring — Execute & Format Response                      #
    # ------------------------------------------------------------------ #
    phase_m_response = create_response_formatter_agent()

    # ------------------------------------------------------------------ #
    # Full PRISM Pipeline (Sequential)                                     #
    # ------------------------------------------------------------------ #
    prism_pipeline = SequentialAgent(
        name="prism_pipeline",
        description="Full PRISM pipeline: P→R→I→S→M sequential execution",
        sub_agents=[
            phase_p_preprocessing,
            phase_r_deep_think,
            phase_i_sql_generation,
            phase_s_validation_loop,
            phase_m_response,
        ],
    )

    # ------------------------------------------------------------------ #
    # Root Orchestrator Agent                                              #
    # after_agent_callback persists session to ADK MemoryService for      #
    # cross-session self-improvement via LoadMemoryTool.                  #
    # ------------------------------------------------------------------ #
    orchestrator = Agent(
        name="prism_text2sql_orchestrator",
        model=settings.llm.orchestrator_model,
        description=(
            "PRISM Text2SQL Orchestrator — Enterprise-grade natural language to SQL agent. "
            "Converts natural language questions into accurate, optimized SQL queries using "
            "a 5-phase multi-agent swarm with Deep Think reasoning and self-improvement."
        ),
        instruction=PromptLibrary.ORCHESTRATOR_AGENT,
        sub_agents=[prism_pipeline],
        after_agent_callback=_add_session_to_memory_callback,
    )

    return orchestrator


# Singleton orchestrator instance
_orchestrator: Agent | None = None


def get_orchestrator() -> Agent:
    """Return the singleton PRISM orchestrator, creating it on first call."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_prism_orchestrator()
    return _orchestrator
