"""
PRISM Response Formatter Agent
Phase M (Monitoring) of the PRISM pipeline — query execution, PII masking,
result formatting, and data storytelling.

Executes the final SQL, applies PII masking, formats results appropriately,
generates a data-story narrative, and assembles the complete QueryResponse.
"""
from __future__ import annotations

from google.adk.agents import Agent

from config.prompts import PromptLibrary
from config.settings import get_settings
from agents.tools.database_tools import execute_sql_query


def _make_interpret_results_tool():
    """Create a result interpretation tool: PII scan + data storytelling."""

    def interpret_results(
        rows: list,
        columns: list,
        sql: str,
        query: str,
        execution_time_ms: float = 0.0,
    ) -> dict:
        """
        Post-process execution results: apply PII masking and generate data story.

        Call this AFTER execute_sql_query to mask PII and generate insight narrative.

        Steps (all pure Python, < 2ms for typical result sets):
          1. Scan rows for PII (emails, SSNs, credit cards, phone numbers, PII-named columns)
          2. Mask detected PII values
          3. Detect anomalies: outliers (>3x avg), high null rates, empty sets
          4. Generate key_finding and narrative story

        Args:
            rows: List of row dicts from execute_sql_query
            columns: Column name list
            sql: Executed SQL (for context)
            query: Original natural language question
            execution_time_ms: Execution latency

        Returns:
            {
              "masked_rows": list,       # rows with PII masked
              "pii_report": dict,        # pii_detected, pii_columns, pii_types_found
              "story": str,              # plain-English narrative
              "key_finding": str,        # one-sentence lead
              "anomalies": list[str],    # notable patterns / outliers
            }
        """
        from core.pii_detector import scan_and_mask

        if isinstance(rows, list) and rows:
            masked_rows, pii_report = scan_and_mask(
                rows=[r if isinstance(r, dict) else {} for r in rows],
                column_names=list(columns) if columns else None,
            )
        else:
            masked_rows = rows or []
            pii_report = {"pii_detected": False, "pii_columns": [], "pii_types_found": {}, "rows_affected": 0}

        anomalies = []
        story_parts = []
        row_count = len(masked_rows)

        if row_count == 0:
            return {
                "masked_rows": [],
                "pii_report": pii_report,
                "story": "The query returned no results. The filter criteria may be too restrictive or the data does not exist yet.",
                "key_finding": "No results found.",
                "anomalies": ["Empty result set — check filter conditions"],
            }

        # Numeric analysis per column
        if masked_rows and isinstance(masked_rows[0], dict):
            for col in (columns or list(masked_rows[0].keys())):
                values = [r.get(col) for r in masked_rows if r.get(col) is not None]
                numeric_vals = []
                for v in values:
                    try:
                        numeric_vals.append(float(v))
                    except (TypeError, ValueError):
                        pass

                if numeric_vals and len(numeric_vals) > 1:
                    avg_val = sum(numeric_vals) / len(numeric_vals)
                    max_val = max(numeric_vals)
                    min_val = min(numeric_vals)
                    story_parts.append(f"{col}: avg={avg_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
                    # Outlier detection
                    if avg_val > 0:
                        outliers = [v for v in numeric_vals if v > avg_val * 3]
                        if outliers:
                            anomalies.append(
                                f"'{col}' has {len(outliers)} outlier(s) significantly above average ({avg_val:.2f})"
                            )

                # Null rate check
                null_count = sum(1 for r in masked_rows if r.get(col) is None)
                if null_count > row_count * 0.15:
                    anomalies.append(
                        f"'{col}' is {null_count * 100 // row_count}% null ({null_count}/{row_count} rows)"
                    )

        story = f"Returned {row_count} row(s) in {execution_time_ms:.0f}ms."
        if story_parts:
            story += " Summary — " + " | ".join(story_parts[:3])  # limit to 3 columns
        if pii_report.get("pii_detected"):
            story += f" [PII masked: {', '.join(pii_report['pii_columns'])}]"

        key_finding = story.split(".")[0] + "."

        return {
            "masked_rows": masked_rows,
            "pii_report": pii_report,
            "story": story,
            "key_finding": key_finding,
            "anomalies": anomalies,
        }

    return interpret_results


# Module-level tool instance
interpret_results = _make_interpret_results_tool()


def create_response_formatter_agent() -> Agent:
    """
    Create the Response Formatter Agent.

    Responsibilities:
    1. Execute the validated, optimized SQL query via execute_sql_query
    2. Call interpret_results for PII masking + data story generation
    3. Format results based on query type (table, single value, time series)
    4. Generate a data narrative leading with the key finding
    5. Surface anomalies (outliers, high null rates, empty result sets)
    6. Include metadata: execution time, row count, SQL, confidence, PII report
    """
    settings = get_settings()

    return Agent(
        name="response_formatter_agent",
        model=settings.llm.formatter_model,
        description=(
            "Response formatter that executes the final SQL query, applies PII masking, "
            "generates a data narrative with anomaly detection, and assembles the "
            "complete structured QueryResponse."
        ),
        instruction=PromptLibrary.RESPONSE_FORMATTER_AGENT,
        tools=[
            execute_sql_query,
            interpret_results,
        ],
    )
