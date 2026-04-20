"""Unit tests for last9_genai.install."""

import os

import pytest
from opentelemetry import _logs, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider

from last9_genai import (
    InstallHandle,
    Last9LogToSpanProcessor,
    Last9SpanProcessor,
    install,
)


def test_install_creates_providers_and_processors():
    handle = install(instrument_openai=False, set_global=False)

    assert isinstance(handle, InstallHandle)
    assert isinstance(handle.tracer_provider, TracerProvider)
    assert isinstance(handle.logger_provider, LoggerProvider)
    assert isinstance(handle.span_processor, Last9SpanProcessor)
    assert isinstance(handle.log_processor, Last9LogToSpanProcessor)
    assert handle.span_processor.log_processor is handle.log_processor


def test_install_uses_provided_tracer_provider():
    tp = TracerProvider()
    handle = install(tracer_provider=tp, instrument_openai=False, set_global=False)
    assert handle.tracer_provider is tp


def test_install_uses_provided_logger_provider():
    lp = LoggerProvider()
    handle = install(logger_provider=lp, instrument_openai=False, set_global=False)
    assert handle.logger_provider is lp


def test_install_sets_capture_content_env_var(monkeypatch):
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    install(instrument_openai=False, set_global=False)
    assert os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT") == "true"


def test_install_does_not_override_capture_content_env_var(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    install(instrument_openai=False, set_global=False)
    assert os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT") == "false"


def test_install_respects_capture_content_false(monkeypatch):
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    install(instrument_openai=False, set_global=False, capture_content=False)
    assert os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT") is None


def test_install_forwards_span_processor_kwargs():
    from last9_genai import ModelPricing

    handle = install(
        instrument_openai=False,
        set_global=False,
        custom_pricing={"gpt-4o": ModelPricing(input=2.5, output=10.0)},
        enable_cost_tracking=False,
    )
    assert handle.span_processor.enable_cost_tracking is False
    assert handle.span_processor.custom_pricing is not None
    assert "gpt-4o" in handle.span_processor.custom_pricing


def test_install_shutdown_does_not_raise():
    handle = install(instrument_openai=False, set_global=False)
    handle.shutdown()


def test_instrument_openai_skipped_when_package_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opentelemetry.instrumentation.openai_v2":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    handle = install(instrument_openai=True, set_global=False)
    assert isinstance(handle, InstallHandle)
