# last9_genai

Python OTel enrichment library for GenAI observability — adds Last9-specific
and OTel GenAI semantic-convention attributes to spans, and bridges non-OTel
tracing systems (like the OpenAI Agents SDK) into the OTel span pipeline.

## Language

**Agents span**:
An `agents.tracing.Span` object from the OpenAI Agents SDK's own tracing
system — not an OTel span. Identified by `span_id` (format `span_<hex>`),
carries `span_data` (agent/generation/function/handoff/guardrail/custom/
response), and is delivered via `TracingProcessor.on_span_start`/`on_span_end`
callbacks, not via OTel context managers.
_Avoid_: "span" alone when the OTel counterpart is also in scope.

**Agents trace**:
An `agents.tracing.Trace` object — the Agents SDK's top-level unit wrapping
one full run (one or more agent spans). Identified by `trace_id` (format
`trace_<hex>`), carries `name` (workflow name), `group_id`, `metadata`.
Delivered via `TracingProcessor.on_trace_start`/`on_trace_end`.
_Avoid_: "trace" alone when the OTel counterpart is also in scope; "workflow"
(that's `trace.name`'s value, not the object itself).

**OTel span**:
A real `opentelemetry.trace.Span` created by this library's tracer, exported
through the standard OTel pipeline (`Last9SpanProcessor` → OTLP exporter).
Has its own OTel-native `trace_id`/`span_id` — never reuses or forges IDs
from an Agents span/trace. Where traceability to the original Agents
identifiers matters, they're preserved as span attributes
(`gen_ai.agents.trace_id`, `gen_ai.agents.span_id`), not as the OTel IDs
themselves.

**Last9SpanProcessor**:
The library's core OTel `SpanProcessor` — enriches any OTel span passing
through it with context attributes (conversation/workflow/agent identity,
cost). Vendor-agnostic: doesn't know or care whether the span originated
from raw OpenAI/Anthropic calls, LangChain, or a bridged Agents span.

## Relationships

- An **Agents trace** contains one or more **Agents spans** (parent/child via `parent_id`).
- Each **Agents trace** maps to exactly one root **OTel span**; each **Agents span** maps to exactly one **OTel span**, parented to its Agents parent's corresponding OTel span (or the trace's root OTel span, if top-level, or if the referenced parent is missing — e.g. it already ended, or a dangling upstream reference).
- Every **OTel span** — regardless of origin (Agents SDK bridge, raw OpenAI/Anthropic call, LangChain) — passes through **Last9SpanProcessor** exactly once before export.

## Example dialogue

> **Dev:** "The span's `parent_id` is None — is that the root of the trace?"
> **Domain expert:** "Only if you mean the Agents span's parent_id. An Agents span with no parent still nests under the Agents trace's root OTel span — it's not an OTel root itself unless the whole Agents trace has no explicit tracing wrapper at all."

## Flagged ambiguities

- "trace" and "span" used interchangeably for both the Agents SDK's native objects and OTel's objects — resolved: always qualify as "Agents trace/span" vs "OTel span" in code, comments, and docs for this integration.
