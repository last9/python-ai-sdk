# OpenAI Agents SDK tracing support — design

**Date:** 2026-07-03
**Status:** Approved, pending plan/implementation

## Problem

`last9_genai` has no support for the OpenAI Agents SDK's native tracing
system. The SDK does not emit OTel spans — it produces its own Agents
trace/Agents span object graph (see `CONTEXT.md`), consumable only via its
`TracingProcessor` interface (`agents.tracing.add_trace_processor`). Every
existing `last9_genai` integration (OpenAI, Anthropic, LangChain,
`agent_context()`) assumes OTel spans already exist by the time they run;
none of them bridge a non-OTel tracing system into OTel spans.

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

- Constructor takes `max_content_length: int = 4096`, same name/default as
  `Last9LogToSpanProcessor`. Applied uniformly to generation input/output and
  function input/output — same truncate-with-`...[truncated]`-suffix
  approach, no special-casing per span type.

- `on_trace_start(agents_trace)` / `on_trace_end(agents_trace)` — creates/ends
  a **root OTel span** per Agents trace, keyed by `agents_trace.trace_id` in
  an `_active_traces` dict (Agents trace_id → OTel span). OTel span name is
  `agents_trace.name` (the workflow name, e.g. "Customer Service");
  `agents_trace.group_id` and `agents_trace.metadata` (when present) are set
  as `gen_ai.agents.group_id` and `custom.*` attributes on the root OTel
  span. This gives each Agents SDK run a single-root trace tree in the Last9
  UI instead of one root per top-level Agents span.
- `on_span_start(agents_span)` / `on_span_end(agents_span)` — creates/ends a
  real OTel span per Agents span via `tracer.start_span()`. Parent context
  resolved by looking up `agents_span.parent_id` in an `_active` dict
  (Agents span_id → OTel span), falling back to
  `_active_traces[agents_span.trace_id]` whenever the direct lookup misses —
  both for genuinely top-level Agents spans (`parent_id is None`) and for a
  `parent_id` that's set but not found in `_active` (parent already ended
  before this child started, or a dangling reference from upstream). The
  fallback keeps the span attached to its trace's root rather than becoming
  an orphaned OTel root trace; a debug-level log line is emitted on the
  dangling-reference case so it stays visible without being noisy.
- Both dicts guarded by a single `threading.Lock` — Agents SDK traces can run
  concurrently across threads (e.g. concurrent FastAPI request handlers).
- Original Agents SDK identifiers preserved as OTel span attributes:
  `gen_ai.agents.trace_id`, `gen_ai.agents.span_id`,
  `gen_ai.agents.parent_span_id`. The OTel span's own `trace_id`/`span_id`
  are never derived from these — OTel identity is entirely OTel-native.
- `agents_span.started_at` / `agents_span.ended_at` (ISO datetime strings)
  converted to epoch nanoseconds for the OTel span's `start_time` /
  `end_time` params.
- `agents_span.error` (when present) →
  `otel_span.set_status(StatusCode.ERROR, message)`.
- `force_flush()` delegates to `tracer_provider.force_flush()`. `shutdown()`
  does **not** call `tracer_provider.shutdown()` — the processor doesn't own
  the provider's lifecycle, `install()`'s `InstallHandle.shutdown()` does;
  calling both would double-shutdown. Instead, `shutdown()` forcibly ends any
  OTel spans still open in `_active`/`_active_traces` (leaked because their
  matching `on_span_end`/`on_trace_end` never fired — crash, uncaught
  exception, killed process) with `StatusCode.ERROR` and a
  `gen_ai.agents.leaked_on_shutdown` attribute, so they still export instead
  of vanishing silently, then clears both dicts.

Rejected alternative: hashing/padding original Agents IDs into valid OTel
trace/span IDs and hand-constructing `ReadableSpan` objects to bypass
`tracer.start_span()`, preserving exact ID fidelity. Rejected because it
requires touching unsupported SDK internals and bypasses
`Last9SpanProcessor.on_start`, which would need separate re-plumbing for no
clear benefit over attribute-based cross-referencing.

### Span type → attribute mapping

Source: `agents.tracing.span_data.*SpanData.export()` **only** — not raw
attributes. `AgentSpanData.export()` deliberately omits `metadata` (present
on the object, absent from export) even though every other span type exports
everything it holds; that's read as a signal about the SDK's own intended
data-boundary (possibly PII-shaped), not an oversight. The processor doesn't
reach past `.export()` to recover it.

| Agents span type | OTel span name | Key attributes |
|---|---|---|
| `agent` | `agent {name}` | `gen_ai.agent.name`, handoffs/tools (JSON) |
| `generation` | `chat {model}` | `gen_ai.request.model`, `gen_ai.usage.*`, input/output written as `gen_ai.prompt`/`gen_ai.completion` flat JSON + indexed attrs + `gen_ai.content.prompt`/`gen_ai.content.completion` events — same attribute/event **naming convention** `Last9LogToSpanProcessor` uses (dashboard compatibility), via a purpose-built writer, not shared code |
| `function` | `tool {name}` | `gen_ai.tool.name`, input/output, `gen_ai.agents.mcp_data` (JSON, when present — MCP tool metadata) |
| `handoff` | `handoff {from}->{to}` | `gen_ai.agents.from_agent`, `gen_ai.agents.to_agent` |
| `guardrail` | `guardrail {name}` | `gen_ai.agents.guardrail.triggered` (bool) |
| `custom` | `{name}` | `data` dict flattened as `custom.{key}` = `str(value)` per top-level key, truncated at `max_content_length` — same existing convention `processor.py`'s `_add_context_attributes_on_start` uses for custom context vars, not a new scheme. `str()` never throws regardless of value shape, so this is failure-safe for arbitrary/non-JSON-serializable values without extra handling. |
| `response` | `response` | response id, usage |

Reuses existing `GenAIAttributes` constants where a semconv field already
exists (model, usage, agent.name). New `gen_ai.agents.*` namespace covers
fields with no existing semconv equivalent (from_agent/to_agent,
trace_id/span_id/parent_span_id, guardrail.triggered).

Full v1 scope: all span types mapped in the initial implementation (not a
generation-only MVP), matching what other vendors on the External Tracing
Processors List ship.

**Generation span writer is new code, not reused code.** `GenerationSpanData.
input`/`.output` arrive as complete `Sequence[Mapping[str, Any]]` lists at
span end — one shot, not the incremental per-message OTel log events
`Last9LogToSpanProcessor` consumes via ambient `trace.get_current_span()`.
The two processors share only the target attribute/event naming convention
(for Last9 dashboard compatibility), not a code path — a small dedicated
writer in `agents_sdk.py` builds `gen_ai.prompt`/`gen_ai.completion` flat +
indexed attrs and `gen_ai.content.*` events directly from the whole list.

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

**Data residency note:** `add_trace_processor()` is additive — it runs
alongside the SDK's default processor, which already ships traces to
OpenAI's own backend (platform.openai.com trace viewer). This design
deliberately stays additive rather than switching to `set_trace_processors()`
(which would replace the list and silently stop the caller's existing
OpenAI-backend export as a side effect of opting into Last9). Dual-export
is the default; callers who want Last9-only must disable OpenAI's own
export themselves (`tracing_disabled` / `OPENAI_AGENTS_DISABLE_TRACING`).
Called out explicitly in the README, not just this spec.

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
