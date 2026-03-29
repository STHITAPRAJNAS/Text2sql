"""
PII Detector — Fast regex-based detection and masking
=====================================================
Scans query result rows for personally identifiable information before
returning data to the caller. Pure Python regex — no LLM call, <1ms for
typical result sets (<1000 rows).

Detection methods:
  1. Column-name heuristics: columns named "email", "ssn", "phone" etc.
     are always treated as PII regardless of content.
  2. Value-pattern regex: scans cell values for known PII patterns.
  3. Unity Catalog column tags: if metadata is provided and a column has
     `pii_sensitivity` tag, it is masked automatically.

Masking strategy:
  - Email: john.doe@example.com → j***@***.com
  - SSN: 123-45-6789 → ***-**-6789
  - Credit card: 4111-1111-1111-1111 → ****-****-****-1111
  - Phone: +1 (555) 123-4567 → ***-***-4567
  - Generic (column-name match): value → [REDACTED]

Usage:
    from core.pii_detector import scan_and_mask
    masked_rows, report = scan_and_mask(rows, column_names, schema_metadata)
"""
from __future__ import annotations

import re
from typing import Any

# ------------------------------------------------------------------ #
# PII patterns                                                         #
# ------------------------------------------------------------------ #

_EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
)
_SSN_RE = re.compile(
    r'\b\d{3}[- ]\d{2}[- ]\d{4}\b'
)
_CREDIT_CARD_RE = re.compile(
    r'\b(?:\d{4}[\s\-]){3}\d{4}\b'
)
_PHONE_RE = re.compile(
    r'\b(?:\+?1[\s\-.])?(?:\(?\d{3}\)?[\s\-.])\d{3}[\s\-\.]\d{4}\b'
)
_IP_RE = re.compile(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
)

# Column name keywords that always trigger masking
_PII_COLUMN_KEYWORDS = frozenset({
    "email", "e_mail", "mail",
    "ssn", "social_security", "sin",
    "phone", "mobile", "cell", "telephone", "tel",
    "credit_card", "card_number", "cvv", "cvc",
    "password", "passwd", "pwd", "secret", "token", "api_key",
    "date_of_birth", "dob", "birth_date",
    "address", "street", "zip", "postal",
    "passport", "license", "licence",
    "ip_address", "ip_addr",
    "national_id", "tax_id", "vat",
    "salary", "wage", "compensation",
})


def _column_is_pii(col_name: str) -> bool:
    """Return True if the column name suggests PII data."""
    name = col_name.lower().replace(" ", "_")
    if name in _PII_COLUMN_KEYWORDS:
        return True
    # Partial match
    return any(kw in name for kw in _PII_COLUMN_KEYWORDS)


def _mask_email(value: str) -> str:
    def _replace(m: re.Match) -> str:
        parts = m.group(0).split("@")
        local = parts[0][0] + "***" if parts[0] else "***"
        domain_parts = parts[1].rsplit(".", 1)
        domain = "***." + domain_parts[-1]
        return f"{local}@{domain}"
    return _EMAIL_RE.sub(_replace, value)


def _mask_ssn(value: str) -> str:
    return _SSN_RE.sub(lambda m: "***-**-" + m.group(0)[-4:], value)


def _mask_credit_card(value: str) -> str:
    def _replace(m: re.Match) -> str:
        digits = re.sub(r'\D', '', m.group(0))
        return "****-****-****-" + digits[-4:]
    return _CREDIT_CARD_RE.sub(_replace, value)


def _mask_phone(value: str) -> str:
    def _replace(m: re.Match) -> str:
        digits = re.sub(r'\D', '', m.group(0))
        return "***-***-" + digits[-4:]
    return _PHONE_RE.sub(_replace, value)


def _mask_value(value: str) -> str:
    """Apply all pattern-based masks to a string value."""
    value = _mask_email(value)
    value = _mask_ssn(value)
    value = _mask_credit_card(value)
    value = _mask_phone(value)
    return value


def _detect_patterns_in_value(value: str) -> list[str]:
    """Return list of PII type names detected in a string value."""
    found = []
    if _EMAIL_RE.search(value):
        found.append("email")
    if _SSN_RE.search(value):
        found.append("ssn")
    if _CREDIT_CARD_RE.search(value):
        found.append("credit_card")
    if _PHONE_RE.search(value):
        found.append("phone")
    return found


def scan_and_mask(
    rows: list[dict[str, Any]],
    column_names: list[str] | None = None,
    schema_metadata: dict[str, Any] | None = None,
    mask_pii: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Scan result rows for PII and optionally mask detected values.

    Runs in-place scan with two passes:
      Pass 1 (column heuristic): O(columns) — checks column names
      Pass 2 (value scan): O(rows × pii_columns) — only for non-heuristic columns

    For 1000 rows × 20 columns, typical latency < 2ms.

    Args:
        rows: List of result row dicts
        column_names: All column names (used for heuristic detection)
        schema_metadata: Optional Unity Catalog metadata with column tags
        mask_pii: If True, replace PII with masked version; if False, just report

    Returns:
        (masked_rows, report) where report = {
            "pii_detected": bool,
            "pii_columns": list of column names containing PII,
            "pii_types_found": dict {column: [pii_types]},
            "rows_affected": int,
        }
    """
    if not rows:
        return rows, {"pii_detected": False, "pii_columns": [], "pii_types_found": {}, "rows_affected": 0}

    all_cols = column_names or (list(rows[0].keys()) if rows else [])

    # Determine PII columns by name heuristic
    pii_cols_by_name: set[str] = {col for col in all_cols if _column_is_pii(col)}

    # Check Unity Catalog column tags
    if schema_metadata:
        for col_name, col_meta in schema_metadata.get("columns", {}).items():
            tags = col_meta.get("tags", {})
            if tags.get("pii_sensitivity") or tags.get("mask"):
                pii_cols_by_name.add(col_name)

    pii_types_found: dict[str, list[str]] = {}
    rows_affected = 0

    # Build masked rows
    if not mask_pii:
        # Scan only — no masking
        for row in rows:
            for col, val in row.items():
                if not isinstance(val, str):
                    continue
                if col in pii_cols_by_name:
                    pii_types_found.setdefault(col, ["column_name_match"])
                else:
                    detected = _detect_patterns_in_value(val)
                    if detected:
                        pii_types_found.setdefault(col, []).extend(detected)
        return rows, {
            "pii_detected": bool(pii_types_found),
            "pii_columns": list(pii_types_found.keys()),
            "pii_types_found": pii_types_found,
            "rows_affected": 0,
        }

    masked_rows = []
    for row in rows:
        masked_row = {}
        row_had_pii = False
        for col, val in row.items():
            if not isinstance(val, str):
                masked_row[col] = val
                continue

            if col in pii_cols_by_name:
                masked_row[col] = "[REDACTED]"
                pii_types_found.setdefault(col, ["column_name_match"])
                row_had_pii = True
            else:
                masked_val = _mask_value(val)
                if masked_val != val:
                    detected = _detect_patterns_in_value(val)
                    pii_types_found.setdefault(col, []).extend(detected)
                    row_had_pii = True
                masked_row[col] = masked_val

        masked_rows.append(masked_row)
        if row_had_pii:
            rows_affected += 1

    return masked_rows, {
        "pii_detected": bool(pii_types_found),
        "pii_columns": list(pii_types_found.keys()),
        "pii_types_found": {k: list(set(v)) for k, v in pii_types_found.items()},
        "rows_affected": rows_affected,
    }
