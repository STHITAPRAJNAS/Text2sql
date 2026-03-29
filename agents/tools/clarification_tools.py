"""
Clarification Tools
===================
When the Deep Think analyzer determines that a query is too ambiguous
to generate reliable SQL (confidence < threshold, multiple plausible
interpretations, missing required filter values, etc.), it calls
`request_clarification` instead of proceeding.

This tool exits the PRISM pipeline early and returns a structured
clarification request to the API caller.  The response includes:
  - A clear question for the user
  - Optional answer choices for guided disambiguation
  - The confidence score that triggered clarification
  - The list of ambiguities detected

The API response model includes `needs_clarification: true` so clients
can render a disambiguation UI before re-submitting.

ADK integration:
  request_clarification sets tool_context.actions.escalate = True
  which terminates the LoopAgent or SequentialAgent iteration and
  bubbles the clarification payload up to the root orchestrator.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger = structlog.get_logger(__name__)

# Sentinel key written to session state so the runner can detect
# that a clarification was requested
CLARIFICATION_STATE_KEY = "clarification_request"


def request_clarification(
    question: str,
    options: list[str] | None = None,
    ambiguities: list[str] | None = None,
    confidence: float = 0.0,
    tool_context: Any = None,
) -> dict[str, Any]:
    """
    Signal that the query is too ambiguous to answer without user input.

    Call this when:
    - Confidence score < deep_think_confidence_threshold
    - Multiple tables could plausibly answer the question
    - A required filter value is missing (e.g., "show sales" — which year?)
    - Business term is undefined (e.g., "show active customers" — what is 'active'?)

    Args:
        question: The clarifying question to ask the user
        options: Optional list of answer choices (for guided disambiguation)
        ambiguities: List of specific ambiguities detected
        confidence: Current confidence score that triggered clarification
        tool_context: ADK ToolContext — used to set escalate and write state

    Returns:
        Structured clarification payload (also written to session state)

    Example:
        request_clarification(
            question="Which time period should the revenue calculation cover?",
            options=["This month", "This quarter", "This year", "All time"],
            ambiguities=["No date filter specified", "Could be order_date or ship_date"],
            confidence=0.45,
        )
    """
    payload = {
        "needs_clarification": True,
        "question": question,
        "options": options or [],
        "ambiguities": ambiguities or [],
        "confidence": confidence,
        "status": "clarification_needed",
    }

    # Write to session state so runner can detect and return it
    if tool_context is not None:
        try:
            tool_context.state[CLARIFICATION_STATE_KEY] = payload
        except Exception:
            pass

        # Escalate — stops SequentialAgent/LoopAgent iteration
        try:
            tool_context.actions.escalate = True
        except AttributeError:
            pass

    logger.info(
        "Clarification requested",
        question=question[:80],
        options_count=len(options or []),
        confidence=confidence,
    )

    return payload


def check_needs_clarification(tool_context: Any = None) -> dict[str, Any] | None:
    """
    Check session state for a pending clarification request.

    Called by the runner/formatter to detect if the pipeline exited
    via clarification rather than SQL generation.

    Returns the clarification payload or None.
    """
    if tool_context is None:
        return None
    try:
        return tool_context.state.get(CLARIFICATION_STATE_KEY)
    except Exception:
        return None


def clear_clarification(tool_context: Any = None) -> None:
    """Clear a resolved clarification from session state."""
    if tool_context is None:
        return
    try:
        tool_context.state.pop(CLARIFICATION_STATE_KEY, None)
    except Exception:
        pass
