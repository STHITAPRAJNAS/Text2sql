"""
Query Skill Classifier
=======================
Classifies natural language queries and their generated SQL into a
skill taxonomy.  Completely synchronous — no I/O, no LLM calls.

Skill taxonomy:
  cte                     — WITH … AS (…) common table expression
  multi_table_join        — two or more table JOINs
  window_function         — OVER clause, ROW_NUMBER, RANK, LAG, LEAD
  subquery                — nested (SELECT …) inside the statement
  conditional_aggregation — CASE WHEN … SUM/COUNT
  date_aggregation        — grouping / truncating by date / time
  set_operation           — UNION, INTERSECT, EXCEPT
  text_search             — LIKE, ILIKE, CONTAINS, REGEXP
  pivot                   — PIVOT or manual cross-tab
  aggregation             — GROUP BY with aggregate functions
  simple_filter           — plain WHERE without joins or aggregation
  single_table_select     — single table, basic SELECT

Skills are used to:
  1. Tag few-shot examples for better semantic retrieval
  2. Route queries to the most capable agent config
  3. Track per-skill accuracy for confidence calibration
  4. Surface proven join paths for the same skill type
"""
from __future__ import annotations

import re
from typing import Any


# ── Regex patterns keyed by skill name ───────────────────────────────────────
_SKILL_PATTERNS: dict[str, list[str]] = {
    "cte": [
        r"\bWITH\s+\w+\s+AS\s*\(",
    ],
    "multi_table_join": [
        r"\bJOIN\b",
    ],
    "window_function": [
        r"\bOVER\s*\(",
        r"\bROW_NUMBER\s*\(",
        r"\bRANK\s*\(",
        r"\bDENSE_RANK\s*\(",
        r"\bLAG\s*\(",
        r"\bLEAD\s*\(",
        r"\bNTILE\s*\(",
        r"\bFIRST_VALUE\s*\(",
        r"\bLAST_VALUE\s*\(",
    ],
    "subquery": [
        r"\(\s*SELECT\b",
    ],
    "conditional_aggregation": [
        r"\bCASE\s+WHEN\b",
        r"\bIF\s*\(",
        r"\bIFF\s*\(",
    ],
    "date_aggregation": [
        r"\bDATE_TRUNC\s*\(",
        r"\bTRUNC\s*\(\s*\w+\s*,\s*['\"]",
        r"\bSTRFTIME\s*\(",
        r"\bEXTRACT\s*\(",
        r"\bTO_DATE\s*\(",
        r"\bDATE_PART\s*\(",
        r"\bYEAR\s*\(",
        r"\bMONTH\s*\(",
        r"\bQUARTER\s*\(",
        r"\bWEEK\s*\(",
    ],
    "set_operation": [
        r"\bUNION\b",
        r"\bINTERSECT\b",
        r"\bEXCEPT\b",
        r"\bMINUS\b",
    ],
    "text_search": [
        r"\bLIKE\b",
        r"\bILIKE\b",
        r"\bREGEXP\b",
        r"\bRLIKE\b",
        r"\bCONTAINS\b",
        r"\bMATCHES\b",
    ],
    "pivot": [
        r"\bPIVOT\b",
        r"\bUNPIVOT\b",
    ],
    "aggregation": [
        r"\bGROUP\s+BY\b",
        r"\bSUM\s*\(",
        r"\bCOUNT\s*\(",
        r"\bAVG\s*\(",
        r"\bMIN\s*\(",
        r"\bMAX\s*\(",
        r"\bMEDIAN\s*\(",
        r"\bPERCENTILE\s*\(",
    ],
}

# NL-level hints (matched against the natural language query)
_NL_PATTERNS: dict[str, list[str]] = {
    "date_aggregation": [
        r"\bby\s+(day|week|month|quarter|year)\b",
        r"\btrend\b",
        r"\bover\s+time\b",
        r"\btime\s+series\b",
        r"\bhistorical\b",
    ],
    "window_function": [
        r"\brunning\s+(total|sum|average|count)\b",
        r"\bcumulative\b",
        r"\branking\b",
        r"\btop\s+\d+\s+per\b",
        r"\bprevious\s+(period|month|quarter|year)\b",
    ],
    "text_search": [
        r"\bcontains\b",
        r"\bmatching\b",
        r"\bsearch\b",
        r"\blike\b",
        r"\bstarting\s+with\b",
        r"\bending\s+with\b",
    ],
}

# Difficulty mapping
_DIFFICULTY_MAP: dict[str, str] = {
    "cte": "complex",
    "window_function": "complex",
    "pivot": "complex",
    "subquery": "moderate",
    "conditional_aggregation": "moderate",
    "multi_table_join": "moderate",
    "date_aggregation": "moderate",
    "set_operation": "moderate",
    "aggregation": "simple",
    "text_search": "simple",
    "simple_filter": "simple",
    "single_table_select": "simple",
}

# Difficulty ranking
_DIFFICULTY_RANK = {"simple": 0, "moderate": 1, "complex": 2}


def classify_query(nl_query: str, generated_sql: str = "") -> list[str]:
    """
    Classify a query into skill taxonomy tags.

    Args:
        nl_query:      Natural language question from the user
        generated_sql: SQL produced by the pipeline (empty string if not yet generated)

    Returns:
        List of skill tags, e.g. ["multi_table_join", "date_aggregation", "aggregation"]
        Always contains at least one tag.
    """
    sql_upper = generated_sql.upper() if generated_sql else ""
    nl_lower = nl_query.lower() if nl_query else ""

    detected: set[str] = set()

    # Match SQL patterns
    for skill, patterns in _SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                detected.add(skill)
                break

    # Match NL patterns (supplement SQL detection)
    for skill, patterns in _NL_PATTERNS.items():
        if skill not in detected:
            for pattern in patterns:
                if re.search(pattern, nl_lower, re.IGNORECASE):
                    detected.add(skill)
                    break

    # Fallback classification
    if not detected:
        if sql_upper and "WHERE" in sql_upper and "JOIN" not in sql_upper:
            detected.add("simple_filter")
        else:
            detected.add("single_table_select")

    return sorted(detected)


def extract_join_paths(sql: str) -> list[str]:
    """
    Extract ordered join paths from SQL.

    Returns a list of strings like "orders→order_items→products"
    representing the sequence of tables joined in the query.
    """
    if not sql:
        return []

    # Find table names after FROM and JOIN keywords
    pattern = re.compile(
        r"(?:FROM|JOIN)\s+([a-zA-Z_][\w.]*)",
        re.IGNORECASE,
    )
    tables_raw = pattern.findall(sql)

    # Deduplicate while preserving order, strip schema prefixes for readability
    seen: set[str] = set()
    tables: list[str] = []
    for t in tables_raw:
        name = t.split(".")[-1].lower()  # strip catalog.schema.
        if name not in seen:
            seen.add(name)
            tables.append(name)

    if len(tables) < 2:
        return []

    return ["→".join(tables)]


def get_skill_difficulty(skills: list[str]) -> str:
    """
    Return the highest difficulty level among the given skills.

    Returns: "simple" | "moderate" | "complex"
    """
    if not skills:
        return "simple"

    max_rank = 0
    max_difficulty = "simple"
    for skill in skills:
        difficulty = _DIFFICULTY_MAP.get(skill, "simple")
        rank = _DIFFICULTY_RANK.get(difficulty, 0)
        if rank > max_rank:
            max_rank = rank
            max_difficulty = difficulty

    return max_difficulty
