"""
Tests for the clarification flow tools.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.tools.clarification_tools import (
    CLARIFICATION_STATE_KEY,
    check_needs_clarification,
    clear_clarification,
    request_clarification,
)


class TestRequestClarification:

    def test_returns_payload(self):
        result = request_clarification(
            question="Which time period?",
            options=["This month", "All time"],
            ambiguities=["No date filter"],
            confidence=0.45,
        )
        assert result["needs_clarification"] is True
        assert result["question"] == "Which time period?"
        assert result["options"] == ["This month", "All time"]
        assert result["ambiguities"] == ["No date filter"]
        assert result["confidence"] == 0.45
        assert result["status"] == "clarification_needed"

    def test_empty_options_defaults_to_empty_list(self):
        result = request_clarification(question="Which region?")
        assert result["options"] == []
        assert result["ambiguities"] == []

    def test_writes_to_session_state(self):
        mock_ctx = MagicMock()
        mock_ctx.state = {}
        mock_ctx.actions = MagicMock()
        request_clarification(
            question="Which date?",
            tool_context=mock_ctx,
        )
        assert CLARIFICATION_STATE_KEY in mock_ctx.state
        assert mock_ctx.state[CLARIFICATION_STATE_KEY]["needs_clarification"] is True

    def test_sets_escalate(self):
        mock_ctx = MagicMock()
        mock_ctx.state = {}
        mock_ctx.actions = MagicMock()
        request_clarification(question="What?", tool_context=mock_ctx)
        assert mock_ctx.actions.escalate is True

    def test_works_without_tool_context(self):
        # No tool_context → should not raise
        result = request_clarification(question="What region?", tool_context=None)
        assert result["needs_clarification"] is True


class TestCheckNeedsClarification:

    def test_returns_none_without_context(self):
        assert check_needs_clarification(None) is None

    def test_returns_none_when_no_key_in_state(self):
        mock_ctx = MagicMock()
        # Use a real dict so .get() works correctly
        mock_ctx.state = {}
        assert check_needs_clarification(mock_ctx) is None

    def test_returns_payload_when_set(self):
        mock_ctx = MagicMock()
        payload = {"needs_clarification": True, "question": "What region?"}
        mock_ctx.state = {CLARIFICATION_STATE_KEY: payload}
        result = check_needs_clarification(mock_ctx)
        assert result is not None
        assert result["question"] == "What region?"


class TestClearClarification:

    def test_clears_key(self):
        mock_ctx = MagicMock()
        # Use a real dict directly as state so pop() works
        mock_ctx.state = {CLARIFICATION_STATE_KEY: {"needs_clarification": True}}
        clear_clarification(mock_ctx)
        assert CLARIFICATION_STATE_KEY not in mock_ctx.state

    def test_no_error_without_context(self):
        clear_clarification(None)  # Should not raise
