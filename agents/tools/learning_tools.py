"""
Progressive Learning Tools
============================
ADK tool wrappers for the 5 progressive learning skills.

All tools follow ADK conventions:
  - Plain Python functions (sync) — ADK auto-wraps as FunctionTool
  - Accept optional `tool_context: Any` for session state access
  - Return `dict[str, Any]` — never raise

ADK Session State Usage
-----------------------
These tools use `tool_context.state` — ADK's per-session key-value store —
as the communication bus between pipeline phases:

  Phase P / R: classify_query_skills stores skill tags
    → tool_context.state["skill_tags"] = ["multi_table_join", "aggregation"]
    → tool_context.state["join_paths_used"] = ["orders→order_items→products"]

  Phase I: SQL Generator reads tags to:
    → find_correction_examples — avoid past mistakes
    → get_proven_join_paths    — prefer historically-reliable join paths
    → find_known_clarification — skip redundant clarification requests

  After-agent callback: reads state to feed calibration + expertise stores
    → Captured by orchestrator._add_session_to_memory_callback

Tool Registration
-----------------
  SQL Generator (Phase I):
    find_correction_examples, get_proven_join_paths, find_known_clarification,
    classify_query_skills, get_effective_confidence_threshold

  Deep Think Query Analyzer (Phase R):
    find_known_clarification — auto-apply stored answer before asking user

  Response Formatter (Phase M):
    store_query_correction — called when user provides corrected SQL inline
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ── ADK session state keys (shared across phases) ─────────────────────────────
STATE_SKILL_TAGS = "skill_tags"
STATE_JOIN_PATHS = "join_paths_used"
STATE_PREDICTED_CONFIDENCE = "predicted_confidence"


# ─────────────────────────────────────────────────────────────────────────────
# Skill 1: Query Skill Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_query_skills(
    nl_query: str,
    generated_sql: str = "",
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Classify the current query into skill taxonomy tags and store them in
    the ADK session state for downstream tools to use.

    Call this early in the pipeline (Phase R or Phase I) so:
      • SQL Generator can fetch skill-specific corrections and join paths
      • After-agent callback can record expertise + calibration data

    Args:
        nl_query:      Natural language question
        generated_sql: Generated SQL (empty string at classification time,
                       can also be called after generation to enrich tags)
        tool_context:  ADK ToolContext — used to write to session state

    Returns:
        dict with:
            - skills (list[str]): Taxonomy tags
            - join_paths (list[str]): Join paths extracted from SQL
            - difficulty (str): "simple" | "moderate" | "complex"
            - stored_in_state (bool): Whether session state was updated
    """
    from core.skill_classifier import classify_query, extract_join_paths, get_skill_difficulty

    skills = classify_query(nl_query, generated_sql)
    join_paths = extract_join_paths(generated_sql)
    difficulty = get_skill_difficulty(skills)

    stored = False
    if tool_context is not None:
        try:
            tool_context.state[STATE_SKILL_TAGS] = skills
            if join_paths:
                existing = tool_context.state.get(STATE_JOIN_PATHS, [])
                tool_context.state[STATE_JOIN_PATHS] = list(set(existing + join_paths))
            stored = True
        except Exception as exc:
            logger.warning("classify_query_skills: state write failed", error=str(exc))

    return {
        "skills": skills,
        "join_paths": join_paths,
        "difficulty": difficulty,
        "stored_in_state": stored,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill 2: Correction Examples
# ─────────────────────────────────────────────────────────────────────────────

def find_correction_examples(
    nl_query: str,
    top_k: int = 3,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Retrieve past SQL corrections for queries similar to the current one.

    Call this before generating SQL.  The returned examples help the agent
    avoid patterns that previously failed: "Previous attempt used X which
    was wrong — correct form is Y."

    Args:
        nl_query:     Current natural language query
        top_k:        Maximum corrections to return
        tool_context: ADK ToolContext (for tenant_id extraction)

    Returns:
        dict with:
            - corrections (list[dict]): Similar past corrections
            - count (int)
            - note (str): Agent guidance
    """
    from core.correction_store import find_similar_corrections

    tenant_id = "default"
    if tool_context is not None:
        try:
            tenant_id = tool_context.state.get("tenant_id", "default")
        except Exception:
            pass

    corrections = find_similar_corrections(nl_query, tenant_id=tenant_id, top_k=top_k)

    note = (
        "No past corrections found for this query type."
        if not corrections
        else (
            f"Found {len(corrections)} similar past corrections. "
            "Review the wrong_sql examples and AVOID those patterns. "
            "Use correct_sql as reference for the correct approach."
        )
    )

    return {"corrections": corrections, "count": len(corrections), "note": note}


def store_query_correction(
    original_query: str,
    wrong_sql: str,
    corrected_sql: str,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Persist a SQL correction for future learning.

    Call this when a user provides corrected SQL via feedback.

    Args:
        original_query: The natural language question
        wrong_sql:      SQL that was generated incorrectly
        corrected_sql:  User-provided correct SQL
        tool_context:   ADK ToolContext (for skill tags and tenant ID)

    Returns:
        {"stored": True, "skill_tags": [...]}
    """
    from core.correction_store import store_correction

    skill_tags: list[str] = []
    tenant_id = "default"
    user_id = "anonymous"

    if tool_context is not None:
        try:
            skill_tags = tool_context.state.get(STATE_SKILL_TAGS, [])
            tenant_id = tool_context.state.get("tenant_id", "default")
            user_id = tool_context.state.get("user_id", "anonymous")
        except Exception:
            pass

    # Auto-classify if no tags in state yet
    if not skill_tags:
        from core.skill_classifier import classify_query
        skill_tags = classify_query(original_query, wrong_sql)

    store_correction(
        nl_query=original_query,
        wrong_sql=wrong_sql,
        correct_sql=corrected_sql,
        user_id=user_id,
        tenant_id=tenant_id,
        skill_tags=skill_tags,
    )

    return {
        "stored": True,
        "skill_tags": skill_tags,
        "message": "Correction stored. Will influence future SQL generation for similar queries.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill 3: Schema Expertise / Proven Join Paths
# ─────────────────────────────────────────────────────────────────────────────

def get_proven_join_paths(
    skill_tags: list[str] | None = None,
    tool_context: Any = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Retrieve table join paths that have historically produced high-rated
    results for queries with the given skill tags.

    Call this before writing JOIN clauses — use the returned paths as
    soft hints ("these join patterns have worked well for similar queries").

    Args:
        skill_tags:   Taxonomy tags (if None, reads from session state)
        tool_context: ADK ToolContext (for reading session state)
        top_k:        Max paths to return

    Returns:
        dict with:
            - proven_paths (list[dict]): join_path, skill_tag, avg_rating, success_count
            - count (int)
            - note (str)
    """
    from core.schema_expertise import get_proven_joins

    tags = skill_tags or []
    if not tags and tool_context is not None:
        try:
            tags = tool_context.state.get(STATE_SKILL_TAGS, [])
        except Exception:
            pass

    if not tags:
        return {
            "proven_paths": [],
            "count": 0,
            "note": "No skill tags available — call classify_query_skills first.",
        }

    paths = get_proven_joins(tags, top_k=top_k)

    note = (
        "No proven join paths found for these skill types yet."
        if not paths
        else (
            f"Found {len(paths)} proven join path(s) for {tags}. "
            "Prefer these join orderings when they match the required tables."
        )
    )

    return {"proven_paths": paths, "count": len(paths), "note": note}


# ─────────────────────────────────────────────────────────────────────────────
# Skill 4: Adaptive Confidence Threshold
# ─────────────────────────────────────────────────────────────────────────────

def get_effective_confidence_threshold(
    skill_tags: list[str] | None = None,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Return the calibrated confidence threshold for the current query's skills.

    The threshold is higher for skill types where the pipeline has been
    over-confident (predicted high but users rated low) and lower where
    predictions are well-calibrated.

    Call this to decide whether to trigger a re-think loop.

    Args:
        skill_tags:   Taxonomy tags (if None, reads from session state)
        tool_context: ADK ToolContext

    Returns:
        dict with:
            - threshold (float): Effective threshold for this skill combination
            - base_threshold (float): Configured base threshold
            - skill_tags (list[str])
            - calibrated (bool): True if enough data for calibration
    """
    from core.confidence_calibrator import get_calibrated_threshold, _base_threshold, MIN_SAMPLES

    tags = skill_tags or []
    if not tags and tool_context is not None:
        try:
            tags = tool_context.state.get(STATE_SKILL_TAGS, [])
        except Exception:
            pass

    base = _base_threshold()
    effective = get_calibrated_threshold(tags) if tags else base

    return {
        "threshold": effective,
        "base_threshold": base,
        "skill_tags": tags,
        "calibrated": effective != base,
        "note": (
            f"Use confidence ≥ {effective:.2f} to decide if re-thinking is needed."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill 5: Clarification Memory
# ─────────────────────────────────────────────────────────────────────────────

def find_known_clarification(
    nl_query: str,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Check whether a similar ambiguous query has been clarified before and
    return the stored answer, so agents can auto-apply it without asking.

    Call this BEFORE generating a clarification request.
    If this returns a result, use the answer directly instead of asking.

    Args:
        nl_query:     Current natural language query
        tool_context: ADK ToolContext (for tenant_id)

    Returns:
        dict with:
            - found (bool)
            - question (str | None): The clarifying question previously asked
            - answer (str | None): The stored answer to auto-apply
            - similarity (float): How similar the stored query is (0-1)
            - auto_apply (bool): True if similarity is high enough to auto-apply
    """
    from core.clarification_memory import find_known_clarification as _find

    tenant_id = "default"
    if tool_context is not None:
        try:
            tenant_id = tool_context.state.get("tenant_id", "default")
        except Exception:
            pass

    result = _find(nl_query, tenant_id=tenant_id)
    if result is None:
        return {
            "found": False,
            "question": None,
            "answer": None,
            "similarity": 0.0,
            "auto_apply": False,
            "note": "No prior clarification found — proceed with clarification request if needed.",
        }

    auto_apply = result["similarity"] >= 0.8
    return {
        "found": True,
        "question": result["question"],
        "answer": result["answer"],
        "similarity": result["similarity"],
        "auto_apply": auto_apply,
        "original_query": result["nl_query"],
        "use_count": result.get("use_count", 0),
        "note": (
            f"Known answer (similarity {result['similarity']:.0%}): {result['answer']!r}. "
            + ("AUTO-APPLY this answer — do not ask the user again."
               if auto_apply
               else "Low similarity — confirm with user before applying.")
        ),
    }


def record_clarification_answer(
    nl_query: str,
    question: str,
    answer: str,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Store a (question, answer) clarification pair for future auto-application.

    Call this after the user answers a clarifying question.

    Args:
        nl_query:     The ambiguous natural language query
        question:     The clarifying question the agent asked
        answer:       The user's answer
        tool_context: ADK ToolContext (for tenant_id)

    Returns:
        {"stored": True}
    """
    from core.clarification_memory import store_clarification

    tenant_id = "default"
    if tool_context is not None:
        try:
            tenant_id = tool_context.state.get("tenant_id", "default")
        except Exception:
            pass

    store_clarification(nl_query=nl_query, question=question, answer=answer, tenant_id=tenant_id)
    return {
        "stored": True,
        "message": "Clarification stored. Will be auto-applied for similar future queries.",
    }
