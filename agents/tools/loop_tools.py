"""
Loop Control Tools for PRISM LoopAgent (Phase S)
Provides exit_loop tool that agents call to terminate the validation loop
when the SQL passes all checks — following Google ADK LoopAgent patterns.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.tools import ToolContext


def exit_validation_loop(tool_context: Any) -> dict[str, str]:
    """
    Signal the PRISM validation loop to stop iterating.

    Call this tool when the SQL query has passed ALL validation layers
    (syntax, schema, security, performance) AND has been optimized.
    The LoopAgent will stop after this call and proceed to Phase M.

    Args:
        tool_context: ADK ToolContext (injected automatically by the runner)

    Returns:
        Confirmation message dict
    """
    # Signal ADK LoopAgent to exit by setting end_of_agent action
    tool_context.actions.escalate = True
    return {
        "status": "loop_complete",
        "message": "SQL validation and optimization complete. Proceeding to execution.",
    }
