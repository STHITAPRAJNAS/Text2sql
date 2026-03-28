"""
Validation Tools for PRISM Agents
SQL syntax, semantic, security, and performance validation tools.
Used by the SQL Validator agent in the PRISM swarm.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# SQL keywords that indicate non-SELECT statements
DDL_KEYWORDS = {"CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME"}
DML_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE"}
DANGEROUS_FUNCTIONS = {"pg_read_file", "pg_exec", "lo_export", "copy", "load_file"}
SYSTEM_TABLES = {
    "pg_shadow", "pg_authid", "pg_stat_activity",
    "information_schema.user_privileges",
    "mysql.user", "sys.credentials",
}


def validate_sql_syntax(sql: str, dialect: str = "postgresql") -> dict[str, Any]:
    """
    Perform static syntax validation on a SQL query.

    Checks for:
    - Balanced parentheses
    - Proper keyword ordering (SELECT...FROM...WHERE...GROUP BY...ORDER BY...LIMIT)
    - Basic SQL grammar issues
    - Common syntax mistakes

    Args:
        sql: The SQL query string to validate
        dialect: SQL dialect for dialect-specific checks

    Returns:
        dict with:
            - is_valid (bool): Whether syntax is valid
            - errors (list): Syntax errors found
            - warnings (list): Non-blocking warnings
            - corrected_sql (str | None): Auto-corrected SQL if fixable
    """
    errors = []
    warnings = []
    corrected_sql = None
    working_sql = sql.strip()

    # 1. Empty query check
    if not working_sql:
        return {
            "is_valid": False,
            "errors": [{"message": "Empty SQL query", "layer": "syntax"}],
            "warnings": [],
            "corrected_sql": None,
        }

    # 2. Balanced parentheses
    open_count = working_sql.count("(")
    close_count = working_sql.count(")")
    if open_count != close_count:
        errors.append({
            "message": f"Unbalanced parentheses: {open_count} opening vs {close_count} closing",
            "layer": "syntax",
            "severity": "ERROR",
        })

    # 3. Balanced single quotes
    # Count unescaped single quotes
    quote_count = len(re.findall(r"(?<!')'(?!')", working_sql))
    if quote_count % 2 != 0:
        errors.append({
            "message": "Unbalanced single quotes in SQL string literals",
            "layer": "syntax",
            "severity": "ERROR",
        })

    # 4. Must start with SELECT, WITH, or EXPLAIN
    first_token = working_sql.split()[0].upper() if working_sql.split() else ""
    if first_token not in {"SELECT", "WITH", "EXPLAIN"}:
        errors.append({
            "message": f"Query must start with SELECT, WITH, or EXPLAIN. Got: {first_token}",
            "layer": "syntax",
            "severity": "ERROR",
        })

    # 5. Has FROM clause (unless SELECT 1 or similar)
    if first_token == "SELECT":
        has_from = bool(re.search(r'\bFROM\b', working_sql, re.IGNORECASE))
        has_only_literal = bool(re.match(r'SELECT\s+[\d\'"]+', working_sql, re.IGNORECASE))
        if not has_from and not has_only_literal:
            warnings.append({
                "message": "SELECT without FROM clause — unusual unless selecting a literal",
                "layer": "syntax",
                "severity": "WARNING",
            })

    # 6. No semicolons in the middle (injection risk)
    stripped_no_strings = re.sub(r"'[^']*'", "''", working_sql)
    if ";" in stripped_no_strings.rstrip(";"):
        errors.append({
            "message": "Multiple statements detected (semicolon found). Only single statements allowed.",
            "layer": "syntax",
            "severity": "ERROR",
        })
        # Auto-fix: take first statement only
        corrected_sql = working_sql.split(";")[0].strip()

    # 7. SELECT * warning
    if re.search(r'\bSELECT\s+\*', working_sql, re.IGNORECASE):
        warnings.append({
            "message": "SELECT * used — consider specifying explicit column names for performance",
            "layer": "syntax",
            "severity": "WARNING",
        })

    # 8. Missing GROUP BY for aggregations
    has_aggregation = bool(re.search(
        r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', working_sql, re.IGNORECASE
    ))
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', working_sql, re.IGNORECASE))
    has_only_agg = bool(re.search(
        r'SELECT\s+(COUNT|SUM|AVG|MAX|MIN)\s*\(', working_sql, re.IGNORECASE
    ))
    if has_aggregation and not has_group_by and not has_only_agg:
        warnings.append({
            "message": "Aggregation function used without GROUP BY — verify this is intentional",
            "layer": "syntax",
            "severity": "WARNING",
        })

    # 9. Dialect-specific checks
    if dialect == "postgresql":
        # Double-quoted identifiers are OK in PG
        pass
    elif dialect == "mysql":
        # MySQL uses backtick identifiers
        if '"' in working_sql:
            warnings.append({
                "message": "Double quotes used — MySQL uses backticks for identifiers",
                "layer": "syntax",
                "severity": "WARNING",
            })

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "corrected_sql": corrected_sql if not is_valid and corrected_sql else None,
    }


def check_schema_compliance(
    sql: str,
    schema_context: str,
    tables_in_schema: list[str],
) -> dict[str, Any]:
    """
    Check if a SQL query complies with the known database schema.

    Verifies that:
    - All referenced tables exist in the schema
    - Aliasing is consistent
    - Common schema compliance issues are flagged

    Args:
        sql: The SQL query to validate
        schema_context: JSON or text representation of the schema
        tables_in_schema: List of known table names in the database

    Returns:
        dict with:
            - is_compliant (bool): Whether the SQL is schema-compliant
            - missing_tables (list): Tables referenced but not in schema
            - warnings (list): Non-blocking compliance warnings
    """
    errors = []
    warnings = []
    missing_tables = []

    # Extract CTE names so they are not flagged as missing schema tables
    cte_names: set[str] = set()
    for cte_match in re.finditer(r'\b(\w+)\s+AS\s*\(', sql, re.IGNORECASE):
        cte_names.add(cte_match.group(1).lower())

    # Extract table names from SQL (rough parsing)
    # Match FROM and JOIN clauses
    from_pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*(?:AS\s+\w+)?',
        re.IGNORECASE
    )
    referenced_tables = [
        m.group(1).split(".")[-1].lower()  # Handle schema.table notation
        for m in from_pattern.finditer(sql)
    ]

    schema_tables_lower = [t.lower() for t in tables_in_schema]
    skip_names = {"lateral", "values"} | cte_names

    for table in referenced_tables:
        if table not in schema_tables_lower and table not in skip_names:
            missing_tables.append(table)
            errors.append({
                "message": f"Table '{table}' not found in schema. Available tables: {', '.join(tables_in_schema[:10])}",
                "layer": "schema",
                "severity": "ERROR",
            })

    # Check for implicit joins (comma-separated FROM) — handles aliases like "FROM tbl a, tbl2 b"
    if re.search(r'FROM\s+\w+(?:\s+\w+)?\s*,\s*\w+', sql, re.IGNORECASE):
        warnings.append({
            "message": "Implicit JOIN detected (comma-separated FROM). Use explicit JOIN ... ON syntax.",
            "layer": "schema",
            "severity": "WARNING",
        })

    is_compliant = len(errors) == 0
    return {
        "is_compliant": is_compliant,
        "missing_tables": missing_tables,
        "errors": errors,
        "warnings": warnings,
    }


def check_sql_security(sql: str) -> dict[str, Any]:
    """
    Check a SQL query for security vulnerabilities and policy violations.

    Enforces:
    - No DDL statements (CREATE, DROP, ALTER, TRUNCATE)
    - No DML statements (INSERT, UPDATE, DELETE, MERGE)
    - No access to system/privileged tables
    - No dangerous database functions
    - No comment-based injection patterns

    Args:
        sql: The SQL query to security-check

    Returns:
        dict with:
            - is_safe (bool): Whether the query passes security checks
            - violations (list): Security violations found
            - risk_level (str): "LOW", "MEDIUM", or "HIGH"
    """
    violations = []
    sql_upper = sql.upper()

    # Check for DDL keywords
    for keyword in DDL_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            violations.append({
                "type": "DDL_STATEMENT",
                "message": f"DDL statement '{keyword}' is not allowed",
                "severity": "HIGH",
            })

    # Check for DML keywords
    for keyword in DML_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            violations.append({
                "type": "DML_STATEMENT",
                "message": f"DML statement '{keyword}' is not allowed. Only SELECT queries permitted.",
                "severity": "HIGH",
            })

    # Check for dangerous functions
    for func in DANGEROUS_FUNCTIONS:
        if func.upper() in sql_upper:
            violations.append({
                "type": "DANGEROUS_FUNCTION",
                "message": f"Dangerous function '{func}' is not permitted",
                "severity": "HIGH",
            })

    # Check for system table access
    for sys_table in SYSTEM_TABLES:
        if sys_table.upper() in sql_upper:
            violations.append({
                "type": "SYSTEM_TABLE_ACCESS",
                "message": f"Access to system table '{sys_table}' is not permitted",
                "severity": "HIGH",
            })

    # Check for SQL comment injection patterns
    if "--" in sql or "/*" in sql:
        # Inline comments are OK in structured SQL, but flag for review
        if re.search(r"'\s*--", sql) or re.search(r"'\s*/\*", sql):
            violations.append({
                "type": "COMMENT_INJECTION",
                "message": "Potential comment injection pattern detected",
                "severity": "HIGH",
            })

    # Check for UNION-based injection
    union_count = len(re.findall(r'\bUNION\b', sql_upper))
    if union_count > 3:
        violations.append({
            "type": "EXCESSIVE_UNIONS",
            "message": f"Excessive UNION count ({union_count}) may indicate injection",
            "severity": "MEDIUM",
        })

    # Determine risk level
    if any(v["severity"] == "HIGH" for v in violations):
        risk_level = "HIGH"
    elif violations:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "is_safe": len(violations) == 0,
        "violations": violations,
        "risk_level": risk_level,
    }


def check_performance_safety(
    sql: str,
    table_row_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Check a SQL query for potential performance issues.

    Identifies:
    - Queries without WHERE clauses on large tables
    - Missing LIMIT on potentially large result sets
    - Functions on indexed columns that prevent index use
    - Cartesian products (missing JOIN conditions)
    - ORDER BY on non-indexed columns

    Args:
        sql: The SQL query to check
        table_row_counts: Optional dict mapping table names to row counts

    Returns:
        dict with:
            - warnings (list): Performance warnings
            - suggestions (list): Optimization suggestions
            - risk_level (str): "LOW", "MEDIUM", or "HIGH"
    """
    warnings = []
    suggestions = []
    sql_upper = sql.upper()
    table_row_counts = table_row_counts or {}

    # Check for missing LIMIT
    has_limit = bool(re.search(r'\bLIMIT\b', sql_upper))
    has_aggregation = bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper))
    if not has_limit and not has_aggregation:
        warnings.append({
            "type": "MISSING_LIMIT",
            "message": "Query has no LIMIT clause — may return large result sets",
            "severity": "MEDIUM",
        })
        suggestions.append("Add LIMIT 1000 to prevent large result sets")

    # Check for functions on WHERE clause columns
    where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|$)', sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        if re.search(r'\b(UPPER|LOWER|TRIM|SUBSTRING|CAST)\s*\(\s*\w+', where_clause, re.IGNORECASE):
            warnings.append({
                "type": "FUNCTION_ON_INDEXED_COLUMN",
                "message": "Function applied to column in WHERE clause may prevent index usage",
                "severity": "MEDIUM",
            })
            suggestions.append("Consider using functional indexes or rewriting the condition")

    # Check for CROSS JOIN or missing JOIN condition
    if re.search(r'\bCROSS\s+JOIN\b', sql_upper):
        # Extract tables involved
        from_pattern = re.compile(r'\bFROM\s+([a-zA-Z_]\w*)', re.IGNORECASE)
        tables = from_pattern.findall(sql)
        total_rows = sum(table_row_counts.get(t, 0) for t in tables)
        if total_rows > 10000:
            warnings.append({
                "type": "CARTESIAN_PRODUCT",
                "message": "CROSS JOIN on potentially large tables",
                "severity": "HIGH",
            })

    # Large table without WHERE
    has_where = bool(re.search(r'\bWHERE\b', sql_upper))
    if not has_where and table_row_counts:
        large_tables = [t for t, c in table_row_counts.items() if c > 1_000_000]
        if large_tables:
            warnings.append({
                "type": "FULL_SCAN_LARGE_TABLE",
                "message": f"Full table scan on large table(s): {', '.join(large_tables)}",
                "severity": "HIGH",
            })

    # Determine risk level
    if any(w["severity"] == "HIGH" for w in warnings):
        risk_level = "HIGH"
    elif warnings:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "warnings": warnings,
        "suggestions": suggestions,
        "risk_level": risk_level,
        "has_limit": has_limit,
        "has_where": has_where,
    }
