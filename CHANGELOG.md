# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-04-20

### Added
- **`install()`** one-call setup helper that wires `TracerProvider`,
  `LoggerProvider`, `Last9SpanProcessor`, `Last9LogToSpanProcessor`, the
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var, and
  `OpenAIInstrumentor().instrument(logger_provider=...)` (when
  `opentelemetry-instrumentation-openai-v2` is installed). Collapses the
  typical six-line boilerplate to a single call.
- `install()` returns an `InstallHandle` dataclass so callers can reach the
  provider (to attach exporters) and call `shutdown()`.
- Accepts caller-provided `tracer_provider` / `logger_provider`, forwards
  cost-tracking kwargs (`custom_pricing`, `enable_cost_tracking`, …) through
  to `Last9SpanProcessor`, and can be opted out of instrumentation / global
  registration.

## [1.2.0] - 2026-04-20

### Added
- **`agent_context()`**: new context manager for OTel GenAI semantic-convention
  agent identity attributes (`gen_ai.agent.id`, `gen_ai.agent.name`,
  `gen_ai.agent.description`, `gen_ai.agent.version`). All spans created inside
  the context are auto-tagged by `Last9SpanProcessor`, matching the shape
  OpenAI Agents SDK and `autogen-core` emit natively on their own agent spans.
- `agent_context` composes with `conversation_context()` and
  `workflow_context()`; covered by tests for triple-nesting, multi-agent
  handoff (same conversation), inner-overrides-outer, and mixed-exit order.

### Notes
- `agent_name` is the only required argument (per OTel semconv).
  `agent_id` / `agent_description` / `agent_version` are optional.
- Native-instrumented agent spans (e.g. AutoGen's `invoke_agent`,
  OpenAI Agents SDK) set `gen_ai.agent.*` directly inside the span body and
  will override values from `agent_context`. Sibling and child spans still
  receive `agent_context`'s values.

## [1.1.0] - 2026-04-20

### Added
- **`Last9LogToSpanProcessor`**: new OTel `LogRecordProcessor` that promotes
  GenAI log events emitted by `opentelemetry-instrumentation-openai-v2` (new
  GenAI semconv) onto the currently active span as both flat span attributes
  and indexed attributes so the Last9 LLM dashboard renders prompts,
  completions, and tool calls.
  - Flat attrs: `gen_ai.prompt`, `gen_ai.completion` (JSON arrays)
  - Span events: `gen_ai.content.prompt`, `gen_ai.content.completion`
  - Indexed attrs: `gen_ai.prompt.{i}.*`, `gen_ai.completion.{i}.*`
    (AgentOps / Traceloop compatible)
- `Last9SpanProcessor` now accepts an optional `log_processor=` kwarg; per-span
  counter state in the bridge is released when its span ends.

### Fixed
- LLM dashboard now shows user/assistant/tool messages for apps using the new
  GenAI semconv (openai-v2) — previously these payloads were only emitted as
  log records and never reached the dashboard.

### Notes
- Python 3.14 users must pin `wrapt<2` because
  `opentelemetry-instrumentation-openai-v2` 2.3b0 calls
  `wrap_function_wrapper(module=..., name=..., wrapper=...)` and wrapt 2.0
  renamed the first kwarg to `target=`. Without the pin, instrumentation fails
  silently and no log events are emitted.

## [1.0.0] - 2026-02-14

### Added
- Initial open source release of Last9 Python AI SDK
- **Cost Tracking**: Automatic cost calculation for 20+ AI models
  - Anthropic: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
  - OpenAI: GPT-4o, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
  - Google: Gemini Pro, Gemini 1.5 Pro, Gemini 1.5 Flash
  - Cohere: Command R, Command R+
  - And more...
- **Conversation Tracking**: Multi-turn conversation tracking with `gen_ai.conversation.id`
- **Workflow Management**: Cost aggregation across multi-step workflows
- **Span Classification**: `gen_ai.l9.span.kind` for filtering (llm/tool/prompt)
- **Prompt Versioning**: Hash-based prompt template tracking and versioning
- **Tool/Function Tracking**: Enhanced attributes for tool and function calls
- **Performance Metrics**: Response times, request/response sizes, quality scores
- **Content Events**: Input/output prompts as span events
- **Standard Compliance**: Full compatibility with OpenTelemetry GenAI v1.28.0 conventions

### Documentation
- Comprehensive README with usage examples
- Installation guide (INSTALL.md)
- Complete API reference
- Working examples:
  - Basic usage with cost tracking
  - Anthropic Claude SDK integration
  - Conversation tracking with multi-turn support
  - Tool and function call tracking

### Technical Details
- Built on OpenTelemetry Python SDK
- Requires Python >=3.9 (aligned with OpenTelemetry API requirements)
- Zero dependencies beyond OpenTelemetry
- Works with any OTLP-compatible backend
- Feature parity with last9-node-agent

### Integration Support
- OpenAI Python SDK
- Anthropic Python SDK
- LangChain (via examples)
- FastAPI (via examples)
- Framework-agnostic design

---

## Versioning Guidelines

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

For more information, see the [SPEC.md](SPEC.md) document.
