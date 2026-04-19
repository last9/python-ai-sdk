"""Unit tests for Last9LogToSpanProcessor."""

import json

import pytest
from opentelemetry import trace
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from last9_genai import Last9LogToSpanProcessor, Last9SpanProcessor


@pytest.fixture
def setup_providers():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    bridge = Last9LogToSpanProcessor(max_content_length=200)
    tracer_provider.add_span_processor(Last9SpanProcessor(log_processor=bridge))
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(bridge)

    tracer = tracer_provider.get_tracer("t")
    logger = logger_provider.get_logger("openai_v2")

    yield tracer, logger, exporter, bridge

    tracer_provider.shutdown()
    logger_provider.shutdown()


def test_prompt_event_sets_flat_and_indexed_attrs(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    with tracer.start_as_current_span("chat gpt-4o") as span:
        logger.emit(
            LogRecord(
                event_name="gen_ai.user.message",
                body={"role": "user", "content": "hello"},
                context=trace.set_span_in_context(span),
            )
        )

    spans = exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.prompt.0.role"] == "user"
    assert attrs["gen_ai.prompt.0.content"] == "hello"
    assert json.loads(attrs["gen_ai.prompt"]) == [{"role": "user", "content": "hello"}]

    events = {e.name: dict(e.attributes) for e in spans[0].events}
    assert "gen_ai.content.prompt" in events
    assert json.loads(events["gen_ai.content.prompt"]["content"]) == [
        {"role": "user", "content": "hello"}
    ]


def test_choice_event_sets_completion_with_tool_calls(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    tool_calls = [{"id": "call_1", "function": {"name": "get_weather"}}]
    with tracer.start_as_current_span("chat gpt-4o") as span:
        logger.emit(
            LogRecord(
                event_name="gen_ai.choice",
                body={
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "tool_calls": tool_calls},
                },
                context=trace.set_span_in_context(span),
            )
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert attrs["gen_ai.completion.0.finish_reason"] == "tool_calls"
    assert attrs["gen_ai.completion.0.index"] == 0
    assert json.loads(attrs["gen_ai.completion.0.tool_calls"]) == tool_calls
    assert json.loads(attrs["gen_ai.completion"])[0]["tool_calls"] == tool_calls


def test_multiple_prompts_accumulate_in_order(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    with tracer.start_as_current_span("chat") as span:
        ctx = trace.set_span_in_context(span)
        logger.emit(
            LogRecord(
                event_name="gen_ai.system.message",
                body={"role": "system", "content": "sys"},
                context=ctx,
            )
        )
        logger.emit(
            LogRecord(
                event_name="gen_ai.user.message",
                body={"role": "user", "content": "hi"},
                context=ctx,
            )
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.1.role"] == "user"
    prompts = json.loads(attrs["gen_ai.prompt"])
    assert [p["role"] for p in prompts] == ["system", "user"]


def test_unrelated_event_names_ignored(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    with tracer.start_as_current_span("chat") as span:
        logger.emit(
            LogRecord(
                event_name="app.debug",
                body={"content": "not a gen_ai event"},
                context=trace.set_span_in_context(span),
            )
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert "gen_ai.prompt" not in attrs
    assert not any(k.startswith("gen_ai.prompt.") for k in attrs)


def test_non_dict_body_ignored(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    with tracer.start_as_current_span("chat") as span:
        logger.emit(
            LogRecord(
                event_name="gen_ai.user.message",
                body="plain string",
                context=trace.set_span_in_context(span),
            )
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert "gen_ai.prompt" not in attrs


def test_truncation_applied_on_long_content(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    big = "x" * 500
    with tracer.start_as_current_span("chat") as span:
        logger.emit(
            LogRecord(
                event_name="gen_ai.user.message",
                body={"role": "user", "content": big},
                context=trace.set_span_in_context(span),
            )
        )

    content = dict(exporter.get_finished_spans()[0].attributes)["gen_ai.prompt.0.content"]
    assert content.endswith("...[truncated]")
    assert len(content) <= 200 + len("...[truncated]")


def test_cleanup_releases_state_after_span_ends(setup_providers):
    tracer, logger, exporter, bridge = setup_providers
    with tracer.start_as_current_span("chat") as span:
        span_id = span.get_span_context().span_id
        logger.emit(
            LogRecord(
                event_name="gen_ai.user.message",
                body={"role": "user", "content": "hi"},
                context=trace.set_span_in_context(span),
            )
        )
        assert span_id in bridge._state

    assert span_id not in bridge._state


def test_tool_message_captures_tool_call_id(setup_providers):
    tracer, logger, exporter, _ = setup_providers
    with tracer.start_as_current_span("chat") as span:
        logger.emit(
            LogRecord(
                event_name="gen_ai.tool.message",
                body={"role": "tool", "content": "sunny", "id": "call_abc"},
                context=trace.set_span_in_context(span),
            )
        )

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.prompt.0.role"] == "tool"
    assert attrs["gen_ai.prompt.0.content"] == "sunny"
    assert attrs["gen_ai.prompt.0.tool_call.id"] == "call_abc"


def test_no_active_span_drops_event(setup_providers):
    _, logger, exporter, _ = setup_providers
    logger.emit(
        LogRecord(
            event_name="gen_ai.user.message",
            body={"role": "user", "content": "orphan"},
        )
    )
    assert exporter.get_finished_spans() == ()
