"""Pytest configuration and fixtures"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from last9_genai import Last9SpanProcessor


@pytest.fixture(scope="session")
def tracer_provider():
    """
    Create a single TracerProvider for the entire test session.

    OpenTelemetry doesn't allow overriding the global TracerProvider,
    so we create one at session start and reuse it for all tests.
    """
    memory_exporter = InMemorySpanExporter()

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    provider.add_span_processor(Last9SpanProcessor())

    # Set as global provider (only once per session)
    trace.set_tracer_provider(provider)

    return provider, memory_exporter


@pytest.fixture(scope="function")
def tracer_setup(tracer_provider):
    """
    Setup for each test - provides tracer and clears exporter.

    Returns a tuple of (tracer, memory_exporter)
    """
    provider, memory_exporter = tracer_provider

    # Clear spans from previous test
    memory_exporter.clear()

    # Get tracer from the global provider
    tracer = trace.get_tracer(__name__)

    yield tracer, memory_exporter

    # Cleanup after test
    memory_exporter.clear()
