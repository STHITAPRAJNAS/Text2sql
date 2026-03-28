"""
PRISM Orchestrator — The Master Coordinator
Enterprise Text2SQL Agent using Google ADK Multi-Agent Swarm

Architecture:
  PRISM = Pre-processing → Reasoning → Intent Mapping → SQL Synthesis → Monitoring

  ┌─────────────────────────────────────────────────────────────────────┐
  │                     PRISM ORCHESTRATOR                              │
  │                                                                     │
  │  Phase P: Pre-processing (ParallelAgent)                            │
  │    ├── Schema Discovery Agent  ──→ DB schema + relationships        │
  │    └── Metadata Enrichment Agent ─→ Business context + glossary    │
  │                                                                     │
  │  Phase R: Reasoning (SequentialAgent - Deep Think)                  │
  │    ├── Deep Think Query Analyzer ─→ CoT query decomposition         │
  │    └── Schema Linker Agent ───────→ Entity-to-schema mapping        │
  │                                                                     │
  │  Phase I: Intent → SQL (Agent)                                      │
  │    └── SQL Generator Agent ───────→ Dialect-aware SQL               │
  │                                                                     │
  │  Phase S: Synthesis / Validation Loop (LoopAgent, max 3 iter)      │
  │    ├── SQL Validator Agent ───────→ Multi-layer validation          │
  │    └── Query Optimizer Agent ─────→ Performance optimization        │
  │                                                                     │
  │  Phase M: Monitoring / Response (Agent)                             │
  │    └── Response Formatter Agent ──→ Execute + NL answer             │
  └─────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

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
            "business metadata. Both agents run in parallel to minimize latency."
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
            "First decomposes the query with chain-of-thought analysis, "
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
    # The loop exits when the validator returns is_valid=True              #
    # or max iterations are reached.                                       #
    # ------------------------------------------------------------------ #
    phase_s_validation_loop = LoopAgent(
        name="phase_s_validation_optimization_loop",
        description=(
            "Phase S: Iterative SQL validation and optimization loop. "
            "Validates the generated SQL across 4 layers (syntax, schema, security, performance), "
            "then optimizes for maximum performance. Repeats up to max_iterations times."
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
    # The root agent is what gets called by the Runner.                   #
    # It coordinates the pipeline and handles top-level decisions.        #
    # ------------------------------------------------------------------ #
    orchestrator = Agent(
        name="prism_text2sql_orchestrator",
        model=settings.llm.orchestrator_model,
        description=(
            "PRISM Text2SQL Orchestrator — Enterprise-grade natural language to SQL agent. "
            "Converts natural language questions into accurate, optimized SQL queries using "
            "a 5-phase multi-agent swarm with Deep Think reasoning."
        ),
        instruction=PromptLibrary.ORCHESTRATOR_AGENT,
        sub_agents=[prism_pipeline],
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
