"""
OpenTelemetry Tracing — PRISM Pipeline Spans
Each PRISM phase gets a dedicated span so you can see exactly where
latency goes: schema discovery vs deep-think reasoning vs SQL generation
vs validation loops vs execution.

Exported to OTLP (Jaeger / Grafana Tempo / Datadog) when enabled.
Zero-overhead when disabled — all calls are no-ops.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

# Lazy OTel import — only loads when ENABLE_TRACING=true
_tracer = None
_meter  = None

# In-process metrics (always active, low overhead)
_query_counter: dict[str, int] = {
    "total": 0, "success": 0, "failed": 0,
    "cache_hit_l1": 0, "cache_hit_l2": 0,
    "clarification_needed": 0,
}
_phase_latencies: dict[str, list[float]] = {
    "phase_p": [], "phase_r": [], "phase_i": [],
    "phase_s": [], "phase_m": [], "total": [],
}


def init_telemetry(
    service_name: str = "text2sql-prism",
    otlp_endpoint: str = "http://localhost:4317",
    enabled: bool = False,
) -> None:
    """Initialize OpenTelemetry tracer and meter. Call once at startup."""
    global _tracer, _meter
    if not enabled:
        logger.info("Telemetry disabled — using in-process metrics only")
        return

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource(attributes={"service.name": service_name})

        # Tracer
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(service_name)

        # Meter
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint),
            export_interval_millis=30_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(service_name)

        logger.info("OpenTelemetry initialized", endpoint=otlp_endpoint)
    except ImportError:
        logger.warning("opentelemetry packages not installed — tracing disabled")


@contextmanager
def trace_phase(
    phase_name: str,
    attributes: dict[str, Any] | None = None,
):
    """
    Context manager that creates an OTel span for a PRISM phase.
    Also records in-process latency regardless of OTel availability.

    Usage:
        with trace_phase("phase_r_deep_think", {"query": query[:50]}):
            result = await deep_think_agent.run(...)
    """
    start = time.monotonic()
    attrs = attributes or {}

    if _tracer:
        with _tracer.start_as_current_span(
            f"prism.{phase_name}",
            attributes={f"prism.{k}": str(v) for k, v in attrs.items()},
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    __import__("opentelemetry.trace", fromlist=["StatusCode"]).StatusCode.ERROR,
                    str(exc),
                )
                raise
            finally:
                elapsed = (time.monotonic() - start) * 1000
                _record_phase_latency(phase_name, elapsed)
    else:
        try:
            yield None
        finally:
            elapsed = (time.monotonic() - start) * 1000
            _record_phase_latency(phase_name, elapsed)


def _record_phase_latency(phase: str, ms: float) -> None:
    key = phase.replace("prism.", "")
    if key in _phase_latencies:
        _phase_latencies[key].append(ms)
        # Keep last 1000 samples
        if len(_phase_latencies[key]) > 1000:
            _phase_latencies[key] = _phase_latencies[key][-1000:]


def record_query(outcome: str) -> None:
    """Record a query outcome: 'success', 'failed', 'cache_hit_l1', 'cache_hit_l2',
    'clarification_needed'."""
    _query_counter["total"] += 1
    if outcome in _query_counter:
        _query_counter[outcome] += 1


def get_metrics_snapshot() -> dict[str, Any]:
    """Return current in-process metrics snapshot for /api/v1/metrics."""
    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    def _p95(lst: list[float]) -> float:
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        idx = int(len(sorted_lst) * 0.95)
        return round(sorted_lst[idx], 2)

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
        "cache_hit_rate": (
            round(
                (_query_counter["cache_hit_l1"] + _query_counter["cache_hit_l2"])
                / max(_query_counter["total"], 1),
                3,
            )
        ),
    }
