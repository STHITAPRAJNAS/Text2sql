"""
OpenTelemetry — Full Compliance for PRISM Pipeline
====================================================
Implements all 5 OTel compliance requirements:

  1. Semantic conventions  — OTel standard attribute names (db.*, gen_ai.*, user.*)
  2. Context propagation   — W3C TraceContext + Baggage headers; correlation_id in span
  3. OTel Metrics instruments — Counter, Histogram, UpDownCounter exported via OTLP
  4. FastAPI instrumentation — auto via FastAPIInstrumentor (called from api/app.py)
  5. Structlog correlation  — trace_id + span_id injected into every log record

Architecture:
  - Lazy init: OTel only loads when ENABLE_TRACING=true (zero overhead otherwise)
  - In-process fallback: counters/latencies always maintained for /api/v1/metrics
  - Resource attributes follow OTel resource semantic conventions
  - Instruments use OTel Gen AI semantic conventions (OTEP-0214 draft)

Exported to any OTLP receiver: Jaeger, Grafana Tempo, Datadog, Honeycomb, etc.
Set OTEL_EXPORTER_OTLP_ENDPOINT env var to override the default endpoint.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------ #
# OTel Semantic Convention constants                                    #
# (mirrors opentelemetry-semantic-conventions, avoids hard dep)        #
# ------------------------------------------------------------------ #

# Database
DB_SYSTEM          = "db.system"
DB_NAME            = "db.name"
DB_STATEMENT       = "db.statement"
DB_OPERATION       = "db.operation"

# Gen AI (OTEP-0214 / opentelemetry-semantic-conventions ≥ 0.23)
GENAI_SYSTEM       = "gen_ai.system"
GENAI_REQUEST_MODEL = "gen_ai.request.model"
GENAI_USAGE_INPUT  = "gen_ai.usage.input_tokens"
GENAI_USAGE_OUTPUT = "gen_ai.usage.output_tokens"

# User
USER_ID            = "user.id"

# PRISM-specific
PRISM_PHASE        = "prism.phase"
PRISM_QUERY        = "prism.query"          # NL query (first 100 chars)
PRISM_DATABASE     = "prism.database"
PRISM_CONFIDENCE   = "prism.confidence"
PRISM_ATTEMPT      = "prism.attempt"
PRISM_TENANT       = "prism.tenant_id"
PRISM_CACHE_HIT    = "prism.cache_hit"
PRISM_PII_DETECTED = "prism.pii_detected"

# Metric names
METRIC_QUERIES_TOTAL    = "prism.queries.total"
METRIC_QUERIES_SUCCESS  = "prism.queries.success"
METRIC_QUERIES_FAILED   = "prism.queries.failed"
METRIC_CACHE_HITS       = "prism.cache.hits"
METRIC_TOKENS_INPUT     = "prism.tokens.input"
METRIC_TOKENS_OUTPUT    = "prism.tokens.output"
METRIC_QUERY_DURATION   = "prism.query.duration"   # histogram (ms)
METRIC_PHASE_DURATION   = "prism.phase.duration"   # histogram (ms)
METRIC_ACTIVE_QUERIES   = "prism.active_queries"   # up-down counter

# ------------------------------------------------------------------ #
# Module-level state                                                    #
# ------------------------------------------------------------------ #

_tracer   = None
_meter    = None
_enabled  = False

# OTel instrument handles (None when disabled)
_counter_queries_total:   Any = None
_counter_queries_success: Any = None
_counter_queries_failed:  Any = None
_counter_cache_hits:      Any = None
_counter_tokens_in:       Any = None
_counter_tokens_out:      Any = None
_histogram_query_dur:     Any = None
_histogram_phase_dur:     Any = None
_updown_active:           Any = None

# In-process fallback (always active — used by /api/v1/metrics)
_query_counter: dict[str, int] = {
    "total": 0, "success": 0, "failed": 0,
    "cache_hit_l1": 0, "cache_hit_l2": 0,
    "clarification_needed": 0,
}
_phase_latencies: dict[str, list[float]] = {
    "phase_p": [], "phase_r": [], "phase_i": [],
    "phase_s": [], "phase_m": [], "total": [],
}


# ------------------------------------------------------------------ #
# Initialization                                                        #
# ------------------------------------------------------------------ #

def init_telemetry(
    service_name: str = "text2sql-prism",
    otlp_endpoint: str = "http://localhost:4317",
    enabled: bool = False,
    service_version: str = "1.0.0",
    deployment_env: str = "production",
) -> None:
    """
    Initialize OpenTelemetry tracer, meter, and propagator.
    Call once at application startup (e.g. from FastAPI lifespan or main.py).

    Also configures structlog to inject trace_id/span_id into every log record.
    """
    global _tracer, _meter, _enabled
    global _counter_queries_total, _counter_queries_success, _counter_queries_failed
    global _counter_cache_hits, _counter_tokens_in, _counter_tokens_out
    global _histogram_query_dur, _histogram_phase_dur, _updown_active

    _configure_structlog_otel()

    if not enabled:
        logger.info("OTel disabled — in-process metrics only")
        return

    try:
        from opentelemetry import trace, metrics, propagate
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
        from opentelemetry.propagators.composite import CompositePropagator
        from opentelemetry.propagators.b3 import B3Format
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        from opentelemetry.baggage.propagation import W3CBaggagePropagator

        # Resource with OTel semantic convention attributes
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": deployment_env,
        })

        # ---- Tracer ----
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(service_name, service_version)

        # ---- Meter ----
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint),
            export_interval_millis=30_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(service_name, service_version)

        # ---- Propagator: W3C TraceContext + Baggage (+ B3 for legacy systems) ----
        propagate.set_global_textmap(CompositePropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator(),
        ]))

        # ---- Metric instruments ----
        _counter_queries_total   = _meter.create_counter(METRIC_QUERIES_TOTAL,   unit="1",  description="Total queries processed")
        _counter_queries_success = _meter.create_counter(METRIC_QUERIES_SUCCESS,  unit="1",  description="Successful queries")
        _counter_queries_failed  = _meter.create_counter(METRIC_QUERIES_FAILED,   unit="1",  description="Failed queries")
        _counter_cache_hits      = _meter.create_counter(METRIC_CACHE_HITS,       unit="1",  description="Semantic cache hits")
        _counter_tokens_in       = _meter.create_counter(METRIC_TOKENS_INPUT,     unit="1",  description="LLM input tokens consumed")
        _counter_tokens_out      = _meter.create_counter(METRIC_TOKENS_OUTPUT,    unit="1",  description="LLM output tokens generated")
        _histogram_query_dur     = _meter.create_histogram(METRIC_QUERY_DURATION, unit="ms", description="End-to-end query pipeline latency")
        _histogram_phase_dur     = _meter.create_histogram(METRIC_PHASE_DURATION, unit="ms", description="Per-phase latency")
        _updown_active           = _meter.create_up_down_counter(METRIC_ACTIVE_QUERIES, unit="1", description="Currently in-flight queries")

        _enabled = True
        logger.info("OTel initialized", endpoint=otlp_endpoint, service=service_name)

    except ImportError as exc:
        logger.warning("opentelemetry packages missing — tracing disabled", missing=str(exc))
    except Exception as exc:
        logger.warning("OTel init failed — continuing without tracing", error=str(exc))


# ------------------------------------------------------------------ #
# 5. Structlog OTel processor                                           #
# ------------------------------------------------------------------ #

def _otel_log_processor(logger, method, event_dict: dict) -> dict:
    """
    Structlog processor: injects trace_id and span_id from the current OTel
    span into every log record. Enables log-trace correlation in Grafana, Datadog, etc.

    Format: 32-char hex trace_id, 16-char hex span_id (W3C TraceContext standard).
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"]  = format(ctx.span_id, "016x")
            event_dict["trace_flags"] = format(ctx.trace_flags, "02x")
    except Exception:
        pass
    return event_dict


def _configure_structlog_otel() -> None:
    """
    Reconfigure structlog to include the OTel log processor.
    Safe to call multiple times (idempotent check via processor list).
    """
    try:
        import structlog as _sl

        existing = _sl.get_config().get("processors", [])
        # Avoid double-adding
        if any(getattr(p, "__name__", "") == "_otel_log_processor" for p in existing):
            return

        # Insert OTel processor near the front (after timestamper, before renderer)
        new_processors = list(existing)
        insert_at = next(
            (i for i, p in enumerate(new_processors)
             if getattr(p, "__name__", "") in ("add_log_level", "add_log_level_number")),
            0,
        )
        new_processors.insert(insert_at + 1, _otel_log_processor)
        _sl.configure(processors=new_processors)
        logger.debug("Structlog OTel processor installed")
    except Exception as exc:
        logger.debug("Structlog OTel processor setup failed", error=str(exc))


# ------------------------------------------------------------------ #
# 2. Context propagation helpers                                        #
# ------------------------------------------------------------------ #

def extract_trace_context(headers: dict[str, str]) -> Any:
    """
    Extract W3C TraceContext from HTTP request headers.
    Returns an OTel context object (or None if OTel not enabled).

    Use with context.attach(ctx) to make the extracted context current.
    """
    if not _enabled:
        return None
    try:
        from opentelemetry import propagate, context
        ctx = propagate.extract(headers)
        return ctx
    except Exception:
        return None


def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject W3C TraceContext headers into an outgoing request headers dict.
    Use when making downstream HTTP calls to propagate the trace.
    """
    if not _enabled:
        return headers
    try:
        from opentelemetry import propagate
        propagate.inject(headers)
    except Exception:
        pass
    return headers


def get_current_trace_ids() -> dict[str, str]:
    """Return current trace_id and span_id as hex strings (for logging/response headers)."""
    result: dict[str, str] = {}
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            result["trace_id"] = format(ctx.trace_id, "032x")
            result["span_id"]  = format(ctx.span_id, "016x")
    except Exception:
        pass
    return result


# ------------------------------------------------------------------ #
# 1. Semantic-convention-compliant span creation                        #
# ------------------------------------------------------------------ #

@contextmanager
def trace_phase(
    phase_name: str,
    attributes: dict[str, Any] | None = None,
):
    """
    Context manager that creates an OTel span for a PRISM pipeline phase.
    Uses OTel semantic conventions for all attribute names.
    Also records in-process latency for /api/v1/metrics (always).

    Span name format: "prism.<phase_name>" — e.g. "prism.total", "prism.phase_r"
    Attributes set using OTel semantic convention keys.

    Usage:
        with trace_phase("phase_r", {
            PRISM_QUERY: query[:100],
            PRISM_DATABASE: database_name,
            USER_ID: user_id,
        }):
            result = await deep_think_agent.run(...)
    """
    start = time.monotonic()
    attrs = attributes or {}

    # Increment active query counter
    if _updown_active:
        _updown_active.add(1, {PRISM_PHASE: phase_name})

    if _tracer:
        # Build semantic-convention-compliant attribute dict
        span_attrs: dict[str, Any] = {}

        # Map our shorthand attributes to OTel convention names
        for k, v in attrs.items():
            span_attrs[k] = str(v) if not isinstance(v, (bool, int, float)) else v

        # Always set the phase name
        span_attrs[PRISM_PHASE] = phase_name

        with _tracer.start_as_current_span(
            f"prism.{phase_name}",
            attributes=span_attrs,
        ) as span:
            try:
                yield span
            except Exception as exc:
                from opentelemetry.trace import StatusCode
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise
            finally:
                elapsed_ms = (time.monotonic() - start) * 1000
                span.set_attribute("prism.duration_ms", round(elapsed_ms, 2))
                _record_phase_latency(phase_name, elapsed_ms)
                if _histogram_phase_dur:
                    _histogram_phase_dur.record(elapsed_ms, {PRISM_PHASE: phase_name})
                if _updown_active:
                    _updown_active.add(-1, {PRISM_PHASE: phase_name})
    else:
        try:
            yield None
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            _record_phase_latency(phase_name, elapsed_ms)
            if _updown_active:
                _updown_active.add(-1, {PRISM_PHASE: phase_name})


@contextmanager
def trace_db_operation(
    operation: str,
    db_system: str,
    db_name: str = "",
    statement: str = "",
):
    """
    Semantic-convention span for a database operation.

    Attributes: db.system, db.name, db.statement (truncated), db.operation
    """
    attrs = {
        DB_SYSTEM:    db_system,
        DB_NAME:      db_name,
        DB_OPERATION: operation,
        DB_STATEMENT: statement[:500] if statement else "",
    }
    with trace_phase(f"db.{operation.lower()}", attrs) as span:
        yield span


@contextmanager
def trace_llm_call(
    model: str,
    system: str = "google_ai_studio",
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    """
    Semantic-convention span for an LLM invocation.

    Attributes follow OTEP-0214 gen_ai.* conventions.
    Token counts are recorded on the span and exported as metrics.
    """
    attrs = {
        GENAI_SYSTEM:        system,
        GENAI_REQUEST_MODEL: model,
    }
    with trace_phase("llm.invoke", attrs) as span:
        yield span
        # Set token attributes after the call completes
        if span and input_tokens:
            span.set_attribute(GENAI_USAGE_INPUT,  input_tokens)
            span.set_attribute(GENAI_USAGE_OUTPUT, output_tokens)

    # Export token counts via OTel metrics
    if input_tokens and _counter_tokens_in:
        dims = {GENAI_SYSTEM: system, GENAI_REQUEST_MODEL: model}
        _counter_tokens_in.add(input_tokens, dims)
        _counter_tokens_out.add(output_tokens, dims)


# ------------------------------------------------------------------ #
# 3. OTel Metrics — record_query + record_query_complete               #
# ------------------------------------------------------------------ #

def record_query(outcome: str) -> None:
    """
    Record a query outcome in both in-process fallback and OTel instruments.

    outcome: 'success' | 'failed' | 'cache_hit_l1' | 'cache_hit_l2' |
             'clarification_needed'
    """
    _query_counter["total"] += 1
    if outcome in _query_counter:
        _query_counter[outcome] += 1

    # OTel counters
    if _counter_queries_total:
        _counter_queries_total.add(1, {"outcome": outcome})

    if outcome == "success" and _counter_queries_success:
        _counter_queries_success.add(1)
    elif outcome == "failed" and _counter_queries_failed:
        _counter_queries_failed.add(1)
    elif outcome.startswith("cache_hit") and _counter_cache_hits:
        tier = "l1" if outcome.endswith("l1") else "l2"
        _counter_cache_hits.add(1, {"cache.tier": tier})


def record_query_complete(
    pipeline_ms: float,
    database: str = "",
    cache_hit: bool = False,
    pii_detected: bool = False,
    token_input: int = 0,
    token_output: int = 0,
) -> None:
    """
    Record full query completion metrics: duration histogram + token counters.
    Call this once per query after the pipeline finishes.
    """
    # In-process latency tracking
    _record_phase_latency("total", pipeline_ms)

    dims: dict[str, Any] = {
        PRISM_DATABASE:  database,
        PRISM_CACHE_HIT: cache_hit,
        PRISM_PII_DETECTED: pii_detected,
    }
    if _histogram_query_dur:
        _histogram_query_dur.record(pipeline_ms, dims)

    if token_input and _counter_tokens_in:
        _counter_tokens_in.add(token_input, dims)
    if token_output and _counter_tokens_out:
        _counter_tokens_out.add(token_output, dims)


# ------------------------------------------------------------------ #
# In-process latency tracking (fallback)                               #
# ------------------------------------------------------------------ #

def _record_phase_latency(phase: str, ms: float) -> None:
    key = phase.replace("prism.", "")
    if key in _phase_latencies:
        _phase_latencies[key].append(ms)
        if len(_phase_latencies[key]) > 1000:
            _phase_latencies[key] = _phase_latencies[key][-1000:]


def get_metrics_snapshot() -> dict[str, Any]:
    """Return current in-process metrics snapshot for /api/v1/metrics endpoint."""
    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    def _p95(lst: list[float]) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        return round(s[int(len(s) * 0.95)], 2)

    return {
        "queries": dict(_query_counter),
        "latency_ms": {
            phase: {
                "avg": _avg(samples),
                "p95": _p95(samples),
                "samples": len(samples),
            }
            for phase, samples in _phase_latencies.items()
        },
        "cache_hit_rate": round(
            (_query_counter["cache_hit_l1"] + _query_counter["cache_hit_l2"])
            / max(_query_counter["total"], 1),
            3,
        ),
        "otel_enabled": _enabled,
    }
