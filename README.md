# last9-genai

OpenTelemetry SDK for Python that tracks LLM cost, conversations, and agent workflows — with one-call setup for OpenAI and AutoGen apps.

[![PyPI version](https://badge.fury.io/py/last9-genai.svg)](https://badge.fury.io/py/last9-genai)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **`install()`** — one call wires TracerProvider, LoggerProvider, processors, and OpenAI instrumentation
- **Conversation tracking** — `gen_ai.conversation.id` across multi-turn sessions
- **Workflow cost aggregation** — group LLM calls by workflow, track total spend
- **Agent identity** — `gen_ai.agent.*` attributes per OTel GenAI semantic conventions
- **Cost calculation** — automatic for 20+ models; bring-your-own pricing for the rest
- **Log-to-span bridge** — promotes `opentelemetry-instrumentation-openai-v2` log events onto spans so the Last9 LLM dashboard renders prompts, completions, and tool calls
- **`@observe` decorator** — manual span creation with tags, metadata, and category
- **Full OTel GenAI v1.28.0 compliance**

## Quick Start

```bash
pip install last9-genai opentelemetry-exporter-otlp-proto-grpc
```

```python
from last9_genai import install
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

handle = install()
handle.tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
```

That's it. All providers, processors, and OpenAI instrumentation are wired automatically.

Set these environment variables before running:

```bash
OTEL_SERVICE_NAME=my-llm-app
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.last9.io
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <your-token>"
```

> **Python 3.14 + openai-v2**: pin `wrapt<2`. A kwarg rename in wrapt 2.0 breaks `opentelemetry-instrumentation-openai-v2` instrumentation silently.

## Conversation & Workflow Tracking

```python
from last9_genai import install, conversation_context, workflow_context

handle = install()
# ... wire OTLP exporter ...

with conversation_context(conversation_id="thread-123", user_id="user-456"):
    with workflow_context(workflow_id="support-flow", workflow_type="chat"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Span has gen_ai.conversation.id, workflow.id, user.id automatically
```

Contexts nest — `conversation_context` wraps multiple `workflow_context` calls, all spans in scope get tagged.

## Agent Identity

```python
from last9_genai import agent_context

with conversation_context(conversation_id="thread-1"):
    with agent_context(agent_name="support-bot", agent_id="bot-001"):
        # All spans: gen_ai.agent.name, gen_ai.agent.id
        response = client.chat.completions.create(...)
```

`agent_context` composes with `conversation_context` and `workflow_context`. Use it for multi-agent handoffs — each agent sets its own identity on its spans.

## Manual Instrumentation

When `install()` isn't enough — bring your own providers:

```python
from opentelemetry import trace, _logs
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from last9_genai import Last9SpanProcessor, Last9LogToSpanProcessor
import os

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

log_bridge = Last9LogToSpanProcessor()

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(Last9SpanProcessor(log_processor=log_bridge))
trace.set_tracer_provider(tracer_provider)

logger_provider = LoggerProvider()
logger_provider.add_log_record_processor(log_bridge)
_logs.set_logger_provider(logger_provider)

OpenAIInstrumentor().instrument(logger_provider=logger_provider)
```

## @observe Decorator

```python
from last9_genai import observe
from openai import OpenAI

client = OpenAI()

@observe(
    tags=["production"],
    metadata={"category": "customer_support"}
)
def handle_query(query: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
```

`category` appears in the Last9 LLM dashboard Category column and filter dropdown. Use underscores for multi-word categories (`data_analysis` → "data analysis").

## Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name in traces | `unknown-service` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL | required |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers (`key=value`) | required |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture prompt/completion bodies | `false` |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes | — |

### `install()` kwargs

| Kwarg | Description | Default |
|-------|-------------|---------|
| `capture_content` | Sets `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | `True` |
| `instrument_openai` | Call `OpenAIInstrumentor().instrument(logger_provider=...)` | `True` |
| `set_global` | Register providers as OTel globals | `True` |
| `tracer_provider` | Provide an existing `TracerProvider` | `None` |
| `logger_provider` | Provide an existing `LoggerProvider` | `None` |
| `**span_processor_kwargs` | Forwarded to `Last9SpanProcessor` (e.g. `custom_pricing`) | — |

`install()` returns an `InstallHandle` with `.tracer_provider`, `.logger_provider`, and `.shutdown()`.

## Cost Tracking

```python
from last9_genai import install, ModelPricing

handle = install(
    capture_content=True,
    custom_pricing={
        "gpt-4o":            ModelPricing(input=2.50, output=10.0),
        "gpt-4o-mini":       ModelPricing(input=0.15, output=0.60),
        "claude-sonnet-4-5": ModelPricing(input=3.0,  output=15.0),
    }
)
```

Prices are **USD per million tokens**. Conversion: per-token `$0.000003` → `3.0`; per-1K `$0.003` → `3.0`.

### Common models

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|------------|
| Anthropic | claude-opus-4-6 | 15.0 | 75.0 |
| Anthropic | claude-sonnet-4-5 | 3.0 | 15.0 |
| Anthropic | claude-haiku-4-5 | 0.8 | 4.0 |
| OpenAI | gpt-4o | 2.50 | 10.0 |
| OpenAI | gpt-4o-mini | 0.15 | 0.60 |
| OpenAI | o1 | 15.0 | 60.0 |
| Google | gemini-1.5-pro | 1.25 | 10.0 |
| Google | gemini-2.0-flash | 0.075 | 0.30 |

For current pricing: [Anthropic](https://www.anthropic.com/pricing) · [OpenAI](https://openai.com/api/pricing/) · [Google](https://ai.google.dev/pricing) · [llm-prices.com](https://www.llm-prices.com/)

Use `"azure/gpt-4o"` for Azure, `"ollama/llama3.1"` with `input=0.0, output=0.0` for self-hosted.

## Architecture

```
your app
  └── install()
        ├── TracerProvider
        │     └── Last9SpanProcessor         ← enriches spans with cost, conversation,
        │           │                             workflow, agent attrs
        │           └── (your OTLP exporter)
        └── LoggerProvider
              └── Last9LogToSpanProcessor    ← bridges openai-v2 log events
                                                 onto the active span
```

**How `Last9LogToSpanProcessor` works:** `opentelemetry-instrumentation-openai-v2` emits prompt/completion content as OTel log records (new GenAI semconv), not span attributes. The bridge listens to those log records and writes `gen_ai.prompt`, `gen_ai.completion`, span events, and indexed `gen_ai.prompt.{i}.*` / `gen_ai.completion.{i}.*` onto the active span — so the Last9 LLM dashboard can render them.

## Troubleshooting

### `gen_ai.prompt` / `gen_ai.completion` missing on spans

Two likely causes:

1. `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is not `true`. `install(capture_content=True)` sets this automatically.
2. `OpenAIInstrumentor().instrument()` was called without `logger_provider=`. The bridge only receives events if openai-v2 routes logs to the same `LoggerProvider`. `install()` handles this automatically.

### No traces appearing in Last9

`install()` does **not** add an exporter — you must wire one:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

handle.tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
```

`OTLPSpanExporter` reads `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` at instantiation time.

### Python 3.14 + wrapt error

```
TypeError: wrap_function_wrapper() got an unexpected keyword argument 'module'
```

Pin `wrapt<2` — wrapt 2.0 renamed the kwarg and `opentelemetry-instrumentation-openai-v2` 2.3b0 hasn't caught up yet.

### Tool call spans missing message content

`execute_tool` span content capture (tool arguments and results) is Phase 2 work — not yet implemented. Tracked in the project issues.

## Span Attributes Reference

| Attribute | Description |
|-----------|-------------|
| `gen_ai.conversation.id` | Thread / session identifier |
| `gen_ai.prompt` | JSON array of prompt messages |
| `gen_ai.completion` | JSON array of completion choices |
| `gen_ai.prompt.{i}.role` / `.content` | Indexed prompt messages |
| `gen_ai.completion.{i}.role` / `.content` | Indexed completion choices |
| `workflow.id` | Workflow identifier |
| `workflow.type` | Workflow type |
| `user.id` | User identifier |
| `gen_ai.agent.id` | Agent identifier |
| `gen_ai.agent.name` | Agent name |
| `gen_ai.usage.cost` | Computed cost in USD |
| `gen_ai.l9.span.kind` | `llm` / `tool` / `prompt` |

## Examples

See [`examples/`](./examples/) directory:

- [`basic_usage.py`](./examples/basic_usage.py) — Simple LLM tracking
- [`openai_integration.py`](./examples/openai_integration.py) — OpenAI SDK
- [`anthropic_integration.py`](./examples/anthropic_integration.py) — Anthropic SDK
- [`langchain_integration.py`](./examples/langchain_integration.py) — LangChain
- [`fastapi_app.py`](./examples/fastapi_app.py) — FastAPI web app

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `uv run pytest`
4. Submit a pull request

## License

MIT

## Support

- Issues: [GitHub Issues](https://github.com/last9/python-ai-sdk/issues)
- Email: hello@last9.io
