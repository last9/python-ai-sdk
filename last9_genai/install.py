"""
One-call setup helper that wires up the full Last9 GenAI observability stack.

Collapses the six-line boilerplate (TracerProvider + Last9SpanProcessor +
LoggerProvider + Last9LogToSpanProcessor + capture-content env var +
OpenAI instrumentation) into a single ``install()`` call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from opentelemetry import _logs, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider

from .log_processor import Last9LogToSpanProcessor
from .processor import Last9SpanProcessor


@dataclass
class InstallHandle:
    """Handles returned by :func:`install` so callers can flush or tear down."""

    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    span_processor: Last9SpanProcessor
    log_processor: Last9LogToSpanProcessor

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.logger_provider.shutdown()


def install(
    *,
    tracer_provider: Optional[TracerProvider] = None,
    logger_provider: Optional[LoggerProvider] = None,
    instrument_openai: bool = True,
    capture_content: bool = True,
    set_global: bool = True,
    **span_processor_kwargs,
) -> InstallHandle:
    """Wire up Last9 GenAI observability in one call.

    Args:
        tracer_provider: Existing provider to enrich. A new one is created if
            omitted and, when ``set_global`` is true, registered as the global
            tracer provider.
        logger_provider: Existing logger provider to attach the log-to-span
            bridge to. A new one is created if omitted and, when ``set_global``
            is true, registered as the global logger provider.
        instrument_openai: If true and ``opentelemetry-instrumentation-openai-v2``
            is importable, call ``OpenAIInstrumentor().instrument()`` with the
            logger provider so message / completion / tool-call events flow
            through the bridge.
        capture_content: If true, sets
            ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`` so the
            OTel GenAI instrumentations emit message bodies (required for the
            bridge to have something to promote).
        set_global: Register the freshly created provider(s) as OTel globals.
            Pass ``False`` when you manage providers yourself or when running
            tests in parallel.
        **span_processor_kwargs: Extra kwargs forwarded to
            :class:`Last9SpanProcessor` (``custom_pricing``,
            ``enable_cost_tracking``, ``workflow_tracker``).

    Returns:
        An :class:`InstallHandle` exposing both providers and both Last9
        processors so callers can add exporters or call ``shutdown()``.

    Example:
        ```python
        from last9_genai import install

        handle = install()
        # add OTLP exporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        handle.tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter())
        )
        ```
    """
    if capture_content:
        os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    tp_created = False
    if tracer_provider is None:
        tracer_provider = TracerProvider()
        tp_created = True

    lp_created = False
    if logger_provider is None:
        logger_provider = LoggerProvider()
        lp_created = True

    log_bridge = Last9LogToSpanProcessor()
    logger_provider.add_log_record_processor(log_bridge)

    span_processor = Last9SpanProcessor(log_processor=log_bridge, **span_processor_kwargs)
    tracer_provider.add_span_processor(span_processor)

    if set_global:
        if tp_created:
            trace.set_tracer_provider(tracer_provider)
        if lp_created:
            _logs.set_logger_provider(logger_provider)

    if instrument_openai:
        _maybe_instrument_openai(logger_provider)

    return InstallHandle(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        span_processor=span_processor,
        log_processor=log_bridge,
    )


def _maybe_instrument_openai(logger_provider: LoggerProvider) -> None:
    """Call OpenAIInstrumentor().instrument() if the package is installed."""
    try:
        from opentelemetry.instrumentation.openai_v2 import (  # type: ignore
            OpenAIInstrumentor,
        )
    except ImportError:
        return

    instrumentor = OpenAIInstrumentor()
    if getattr(instrumentor, "_is_instrumented_by_opentelemetry", False):
        # Already instrumented — don't double-wrap.
        return
    instrumentor.instrument(logger_provider=logger_provider)
