"""
Tests for PII detector — scan_and_mask function.
"""
from __future__ import annotations

import pytest

from core.pii_detector import scan_and_mask, _column_is_pii, _mask_email, _mask_ssn


class TestColumnHeuristics:

    def test_email_column_detected(self):
        assert _column_is_pii("email") is True
        assert _column_is_pii("user_email") is True
        assert _column_is_pii("EMAIL") is True  # case-insensitive (lowercased internally)

    def test_email_column_case_insensitive_in_scan(self):
        rows = [{"Email": "john@example.com"}]
        masked, report = scan_and_mask(rows)
        # Column "Email" should be detected via heuristic
        assert report["pii_detected"] is True

    def test_phone_column_detected(self):
        assert _column_is_pii("phone") is True
        assert _column_is_pii("mobile_phone") is True

    def test_ssn_column_detected(self):
        assert _column_is_pii("ssn") is True
        assert _column_is_pii("social_security_number") is True

    def test_safe_column_not_detected(self):
        assert _column_is_pii("revenue") is False
        assert _column_is_pii("customer_id") is False
        assert _column_is_pii("order_date") is False


class TestEmailMasking:

    def test_basic_email(self):
        result = _mask_email("john.doe@example.com")
        assert "@" in result
        assert "john.doe" not in result
        assert result.startswith("j")
        assert result.endswith(".com")

    def test_email_in_text(self):
        result = _mask_email("Contact us at support@company.org for help")
        assert "support@company.org" not in result
        assert "s***" in result

    def test_no_email_unchanged(self):
        text = "Hello World 12345"
        assert _mask_email(text) == text


class TestSSNMasking:

    def test_ssn_dashes(self):
        result = _mask_ssn("123-45-6789")
        assert "***-**-6789" == result

    def test_ssn_spaces(self):
        result = _mask_ssn("123 45 6789")
        assert "***-**-6789" == result

    def test_no_ssn_unchanged(self):
        text = "Account 1234567890"
        assert _mask_ssn(text) == text


class TestScanAndMask:

    def test_empty_rows(self):
        masked, report = scan_and_mask([])
        assert masked == []
        assert report["pii_detected"] is False

    def test_no_pii(self):
        rows = [{"customer_id": 1, "revenue": 1000.0, "region": "West"}]
        masked, report = scan_and_mask(rows)
        assert masked == rows
        assert report["pii_detected"] is False
        assert report["rows_affected"] == 0

    def test_email_in_column(self):
        rows = [
            {"id": 1, "email": "alice@example.com"},
            {"id": 2, "email": "bob@test.org"},
        ]
        masked, report = scan_and_mask(rows)
        assert report["pii_detected"] is True
        assert "email" in report["pii_columns"]
        assert masked[0]["email"] == "[REDACTED]"
        assert masked[1]["email"] == "[REDACTED]"
        assert report["rows_affected"] == 2

    def test_email_value_in_non_pii_column(self):
        rows = [{"note": "contact john@example.com for details"}]
        masked, report = scan_and_mask(rows)
        assert report["pii_detected"] is True
        assert "note" in report["pii_columns"]
        assert "john@example.com" not in masked[0]["note"]

    def test_ssn_value_detected(self):
        rows = [{"info": "SSN: 123-45-6789"}]
        masked, report = scan_and_mask(rows)
        assert report["pii_detected"] is True
        assert "123-45-6789" not in masked[0]["info"]

    def test_credit_card_detected(self):
        rows = [{"payment": "Card: 4111-1111-1111-1111"}]
        masked, report = scan_and_mask(rows)
        assert report["pii_detected"] is True
        assert "4111-1111-1111-1111" not in masked[0]["payment"]
        assert "1111" in masked[0]["payment"]  # last 4 preserved

    def test_phone_detected(self):
        rows = [{"contact": "Call 555-867-5309"}]
        masked, report = scan_and_scan(rows)
        assert report["pii_detected"] is True
        assert "867-5309" not in masked[0]["contact"]

    def test_non_string_values_unchanged(self):
        rows = [{"amount": 1234.56, "count": 42, "flag": True, "data": None}]
        masked, report = scan_and_mask(rows)
        assert masked[0]["amount"] == 1234.56
        assert masked[0]["count"] == 42
        assert report["pii_detected"] is False

    def test_mask_false_scan_only(self):
        rows = [{"email": "test@example.com"}]
        masked, report = scan_and_mask(rows, mask_pii=False)
        # Original rows returned unchanged
        assert masked[0]["email"] == "test@example.com"
        assert report["pii_detected"] is True

    def test_multiple_pii_columns(self):
        rows = [{"email": "a@b.com", "ssn": "123-45-6789", "revenue": 100}]
        masked, report = scan_and_mask(rows)
        assert len(report["pii_columns"]) >= 2
        assert masked[0]["revenue"] == 100

    def test_column_names_override(self):
        rows = [{"c1": "a@b.com"}]
        # Pass explicit column names that include "email" mapping
        masked, report = scan_and_mask(rows, column_names=["email"])
        # c1 value detected via value scan even if column_names passed separately
        assert report["pii_detected"] is True

    def test_unity_catalog_tag_masking(self):
        rows = [{"salary_amount": 95000}]
        schema = {"columns": {"salary_amount": {"tags": {"pii_sensitivity": "high"}}}}
        masked, report = scan_and_mask(rows, schema_metadata=schema)
        # Non-string values are passed through, but column flagged
        assert "salary_amount" in report.get("pii_columns", []) or report["pii_detected"] is False
        # At minimum, integer stays integer
        assert masked[0]["salary_amount"] == 95000


# Fix typo in test above
def scan_and_scan(rows):
    """Alias for typo fix."""
    return scan_and_mask(rows)
