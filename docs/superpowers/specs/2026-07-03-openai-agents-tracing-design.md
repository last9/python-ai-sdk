# OpenAI Agents SDK tracing support — design

**Date:** 2026-07-03
**Status:** Approved, pending plan/implementation

## Problem

`last9_genai` has no support for the OpenAI Agents SDK's native tracing
system. The SDK does not emit OTel spans — it produces its own `Trace`/`Span`
object graph, consumable only via its `TracingProcessor` interface
(`agents.tracing.add_trace_processor`). Every existing `last9_genai`
integration (OpenAI, Anthropic, LangChain, `agent_context()`) assumes spans
already arrive as OTel spans; none of them bridge the Agents SDK's own
tracing system.

This also blocks a secondary goal: getting Last9 listed on
https://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list,
which requires shipping a processor that plugs into the SDK's own
`add_trace_processor`/`set_trace_processors` mechanism — not merely being
OTel-compatible via some other bridge.

## Goal

Ship a `TracingProcessor` implementation in `last9_genai` that converts
Agents SDK traces/spans into real OTel spans, flowing through the existing
`Last9SpanProcessor` → OTLP export pipeline unchanged. Wire it into the
existing `install()` one-call setup helper.

## Non-goals

- Preserving the Agents SDK's original trace_id/span_id as the actual OTel
  IDs. OTel gets fresh IDs; original IDs are preserved as span attributes for
  cross-referencing.
- Submitting the PR to `openai/openai-agents-python`'s docs — tracked as a
  follow-up after this ships and is verified end-to-end.

## Architecture

### New module: `last9_genai/agents_sdk.py`

`Last9AgentsTracingProcessor(TracingProcessor)`:

- `on_trace_start(trace)` / `on_trace_end(trace)` — creates/ends a **root
  OTel span** per Agents `Trace`, keyed by `trace_id` in an `_active_traces`
  dict. Span name is `trace.name` (the workflow name, e.g. "Customer
  Service"); `trace.group_id` and `trace.metadata` (when present) are set as
  `gen_ai.agents.group_id` and `custom.*` attributes on the root span. This
  gives each Agents SDK run a single-root trace tree in the Last9 UI instead
  of one root per top-level agent span.
- `on_span_start(span)` / `on_span_end(span)` — creates/ends a real OTel span
  per Agents `Span` via `tracer.start_span()`. Parent context resolved by
  looking up `span.parent_id` in an `_active` dict (keyed by `span_id`),
  falling back to `_active_traces[span.trace_id]` for top-level spans.
- Both dicts guarded by a single `threading.Lock` — Agents SDK traces can run
  concurrently across threads (e.g. concurrent FastAPI request handlers).
- Original Agents SDK identifiers preserved as span attributes:
  `gen_ai.agents.trace_id`, `gen_ai.agents.span_id`,
  `gen_ai.agents.parent_span_id`. OTel does not reuse or forge IDs from
  these — trace/span identity within OTel is entirely OTel-native.
- `started_at` / `ended_at` (ISO datetime strings) converted to epoch
  nanoseconds for `start_time` / `end_time` params.
- `span.error` (when present) → `span.set_status(StatusCode.ERROR, message)`.

Rejected alternative: hashing/padding original Agents IDs into valid OTel
trace/span IDs and hand-constructing `ReadableSpan` objects to bypass
`tracer.start_span()`, preserving exact ID fidelity. Rejected because it
requires touching unsupported SDK internals and bypasses
`Last9SpanProcessor.on_start`, which would need separate re-plumbing for no
clear benefit over attribute-based cross-referencing.

### Span type → attribute mapping

Source: `agents.tracing.span_data.*SpanData.export()`.

| Agents span type | OTel span name | Key attributes |
|---|---|---|
| `agent` | `agent {name}` | `gen_ai.agent.name`, handoffs/tools (JSON) |
| `generation` | `chat {model}` | `gen_ai.request.model`, `gen_ai.usage.*`, input/output as span events (same shape `Last9LogToSpanProcessor` already emits) |
| `function` | `tool {name}` | `gen_ai.tool.name`, input/output |
| `handoff` | `handoff {from}->{to}` | `gen_ai.agents.from_agent`, `gen_ai.agents.to_agent` |
| `guardrail` | `guardrail {name}` | `gen_ai.agents.guardrail.triggered` (bool) |
| `custom` | `{name}` | flattened `data` dict as `custom.*` |
| `response` | `response` | response id, usage |

Reuses existing `GenAIAttributes` constants where a semconv field already
exists (model, usage, agent.name). New `gen_ai.agents.*` namespace covers
fields with no existing semconv equivalent (from_agent/to_agent,
trace_id/span_id/parent_span_id, guardrail.triggered).

Full v1 scope: all span types mapped in the initial implementation (not a
generation-only MVP), matching what other vendors on the External Tracing
Processors List ship.

### Wiring — `install.py`

New flag, following the existing `instrument_openai` pattern
(`install.py:114`, `_maybe_instrument_openai`):

```python
def install(*, instrument_agents_sdk: bool = False, ...):
    ...
    if instrument_agents_sdk:
        _maybe_instrument_agents_sdk(tracer_provider)

def _maybe_instrument_agents_sdk(tracer_provider):
    try:
        from agents import add_trace_processor
    except ImportError:
        return
    add_trace_processor(Last9AgentsTracingProcessor(tracer_provider))
```

**Default is `False`**, unlike `instrument_openai`'s default of `True`. This
is a deliberate asymmetry: auto-enabling could double-trace if a caller has
already wired their own Agents SDK tracing processor via
`set_trace_processors()`, which fully replaces the default processor list.
Callers opt in explicitly.

### Packaging

New optional extra in `pyproject.toml`: `last9-genai[openai-agents]`, pulling
in the `openai-agents` package. Core install is unaffected — `agents_sdk.py`
guards its import inside `_maybe_instrument_agents_sdk`, same pattern as
`_maybe_instrument_openai`.

### Testing — `tests/test_agents_sdk.py`

Mirrors the existing `test_log_processor.py` pattern: real `TracerProvider` +
`InMemorySpanExporter` + the processor under test, feeding real input
objects and asserting on exported spans.

Uses the **real `agents` package** (added as a test/dev dependency) to
construct actual `agents.tracing.Trace` / `Span` / `*SpanData` objects, rather
than duck-typed fakes — chosen over fakes to catch upstream shape drift in
the actual SDK rather than a hand-maintained approximation of it.

Test cases:
- Single trace, single agent span → one root + one child span exported, correct parent/child linkage.
- Nested spans (agent → generation → function) → three-level parent chain preserved.
- Concurrent traces across threads → no cross-contamination between `_active`/`_active_traces` entries.
- Span with `error` set → `StatusCode.ERROR` on the exported span.
- Each span type (`agent`, `generation`, `function`, `handoff`, `guardrail`, `custom`, `response`) → correct attribute mapping per the table above.

### Example — `examples/openai_agents_sdk_integration.py`

Runnable demo following the style of `examples/openai_integration.py`:
`install(instrument_agents_sdk=True)` wired up, then a real `agents.Agent` +
`Runner.run()` call, printing/exporting spans via console or OTLP exporter.

## Follow-up (separate task, out of scope here)

Once implemented and verified end-to-end (real Agents SDK run → spans land
correctly in Last9), open a PR against `openai/openai-agents-python`
`docs/tracing.md`, adding one line to the External Tracing Processors List
linking to this integration's README section.
