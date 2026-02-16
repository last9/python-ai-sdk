# Last9 GenAI - Python SDK

> OpenTelemetry extension for LLM observability: track conversations, workflows, and costs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**Track conversations and workflows in your LLM applications** with automatic context propagation. Built on OpenTelemetry for seamless integration with your existing observability stack.

**Not a replacement** for OTel auto-instrumentation — works alongside it or standalone.

**Key Features:**
- 🎯 **Conversation Tracking**: Automatic multi-turn conversation tracking with `conversation_context`
- 🔄 **Workflow Management**: Track complex multi-step AI workflows with `workflow_context`
- 🎨 **Zero-Touch Instrumentation**: `@observe()` decorator for automatic tracking
- 📊 **Context Propagation**: Thread-safe attribute tracking across nested operations
- 💰 **Optional Cost Tracking**: Bring your own pricing for cost monitoring
- 🏷️ **Span Classification**: Filter by type (llm/tool/chain/agent/prompt)

## Features

### Core Tracking
- 🎯 **Conversation Tracking**: Multi-turn conversations with `gen_ai.conversation.id` and turn numbers
- 🔄 **Workflow Management**: Track multi-step AI operations across LLM calls, tools, and retrievals
- 📊 **Auto-Context Propagation**: Thread-safe context managers that automatically tag all nested operations
- 🎨 **Decorator Pattern**: `@observe()` for zero-touch instrumentation with full input/output/latency tracking
- 🔧 **SpanProcessor**: Automatic context enrichment for all spans in your application

### Enhanced Observability
- 🏷️ **Span Classification**: `gen_ai.l9.span.kind` for filtering (llm/tool/chain/agent/prompt)
- 🛠️ **Tool/Function Tracking**: Enhanced attributes for function calls and tool usage
- ⚡ **Performance Metrics**: Response times, token counts, and quality scores
- 🌐 **Provider Agnostic**: Works with OpenAI, Anthropic, Google, Cohere, etc.
- 📏 **Standard Attributes**: Full OpenTelemetry `gen_ai.*` semantic conventions

### Optional Features
- 💰 **Cost Tracking**: Bring your own model pricing for cost monitoring
- 💸 **Workflow Costing**: Aggregate costs across multi-step operations

## Relationship to OpenTelemetry GenAI

**This is an EXTENSION, not a replacement:**

| Package | Purpose | Approach |
|---------|---------|----------|
| **OTel GenAI**<br/>`opentelemetry-instrumentation-openai-v2` | Auto-instrument LLM SDKs | Automatic (monkey-patching) |
| **Last9 GenAI**<br/>`last9-genai` | Add conversation/workflow tracking | Context-based enrichment |

**You can use:**
1. **Last9 GenAI alone** - Full conversation and workflow tracking
2. **Both together** - OTel auto-traces + Last9 adds conversation/workflow context (recommended!)

See [Working with OTel Auto-Instrumentation](#working-with-otel-auto-instrumentation) for combined usage.

## Installation

**Basic:**
```bash
pip install last9-genai
```

**With OTLP export (recommended):**
```bash
pip install last9-genai[otlp]
```

**Requirements:**
- Python 3.10+
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`

## Quick Start

**Note:** The examples below use `client` to represent your LLM client. Initialize your preferred provider:

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

# Or Anthropic
from anthropic import Anthropic
anthropic_client = Anthropic()

# Or any other provider (Google, Cohere, etc.)
```

The SDK works with **any LLM provider** - just use your client normally!

### Track Conversations (Recommended)

Automatically track multi-turn conversations with zero manual instrumentation:

```python
from last9_genai import conversation_context, Last9SpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# Setup tracing with Last9 processor
provider = TracerProvider()
trace.set_tracer_provider(provider)
provider.add_span_processor(Last9SpanProcessor())

# Track conversations automatically - works with any LLM provider
with conversation_context(conversation_id="session_123", user_id="user_456"):
    # OpenAI
    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Anthropic (same context!)
    response2 = anthropic_client.messages.create(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "How are you?"}]
    )
    # Both calls automatically have conversation_id = "session_123"!
```

### Track Workflows

Track complex multi-step AI operations:

```python
from last9_genai import workflow_context

# Track entire workflow with automatic tagging
with workflow_context(workflow_id="rag_search", workflow_type="retrieval"):
    # All operations automatically tagged with workflow_id
    docs = retrieve_documents(query)  # Tagged
    context = rerank_documents(docs)   # Tagged
    response = generate_answer(context) # Tagged
    # Full workflow visibility with zero manual instrumentation!

# Nest workflows and conversations
with conversation_context(conversation_id="support_123"):
    with workflow_context(workflow_id="order_lookup"):
        # Both conversation AND workflow tracked automatically
        result = lookup_and_respond()
```

### Decorator Pattern (Zero-Touch)

Use `@observe()` for automatic tracking of everything:

```python
from last9_genai import observe

@observe()  # That's it!
def call_llm(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response

# Automatically tracks:
# - Input (prompt)
# - Output (response)
# - Latency (span duration)
# - Context (conversation_id, workflow_id if set)

# Works seamlessly with context managers
with conversation_context(conversation_id="session_456"):
    response = call_llm("Explain quantum computing")
    # Span automatically has conversation_id!
```

### Optional: Cost Tracking

Add cost monitoring by providing model pricing:

```python
from last9_genai import ModelPricing

# Add pricing when creating processor
processor = Last9SpanProcessor(custom_pricing={
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
    "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
})

# Or with decorator
pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

@observe(pricing=pricing)
def call_llm(prompt: str):
    # Now also tracks cost automatically
    return client.chat.completions.create(...)
```

### Decorator Pattern (Zero-Touch)

Use `@observe()` decorator for automatic tracking of input/output, latency, and cost:

```python
from last9_genai import observe, ModelPricing

pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

@observe(pricing=pricing)
def call_openai(prompt: str):
    """Automatically tracks everything!"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response

# That's it! Automatically tracks:
# - Input (prompt)
# - Output (response)
# - Latency (span duration)
# - Cost (calculated from usage)
# - Metadata (from context)

# Works with context too:
with conversation_context(conversation_id="session_123"):
    response = call_openai("Hello!")
    # Span automatically has conversation_id!
```

### Tags and Categories

Add tags and categories for better filtering and organization in your observability platform:

```python
from last9_genai import observe

@observe(
    tags=["production", "customer_support"],
    metadata={
        "category": "customer_support",  # Appears in Last9 dashboard Category column
        "version": "1.0.0",
        "priority": "high"
    }
)
def handle_support_query(query: str):
    """Categorized LLM call with metadata"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
    return response

# Categories automatically appear in Last9 dashboard:
# - Category column in traces table
# - Category filter dropdown
# - Enhanced trace details

# Use underscores for multi-word categories:
@observe(metadata={"category": "data_analysis"})  # Shows as "data analysis"
def analyze_data(data: str):
    return client.chat.completions.create(...)
```

**Common categories:**
- `customer_support`, `conversational_ai`, `code_assistant`
- `data_analysis`, `content_generation`, `summarization`
- `translation`, `research`, `qa_automation`

## Working with OTel Auto-Instrumentation

**Recommended**: Combine OTel auto-instrumentation with Last9 extensions:

```python
# Step 1: Auto-instrument with OpenTelemetry (standard attributes)
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
OpenAIInstrumentor().instrument()

# Step 2: Add Last9 extensions (cost, workflows)
from last9_genai import Last9GenAI, ModelPricing

l9 = Last9GenAI(custom_pricing={
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
})

# Now make LLM calls
from openai import OpenAI
client = OpenAI()

# OTel automatically traces this call (standard attributes)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Last9 adds cost on top of auto-traced span
from opentelemetry import trace
span = trace.get_current_span()
usage = {
    "input_tokens": response.usage.prompt_tokens,
    "output_tokens": response.usage.completion_tokens,
}
cost = l9.add_llm_cost_attributes(span, "gpt-4o", usage)
print(f"Cost: ${cost.total:.6f}")
```

**Result**: You get standard OTel attributes (automatic) + Last9 cost/workflow (manual).

## Usage Examples

### Multi-Turn Conversations

Track conversations across multiple turns automatically:

```python
from last9_genai import conversation_context

# Track a complete conversation session
with conversation_context(conversation_id="support_session_456", user_id="user_456"):
    # Turn 1
    response1 = client.chat.completions.create(
        messages=[{"role": "user", "content": "I need help with my order"}]
    )

    # Turn 2
    response2 = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "I need help with my order"},
            {"role": "assistant", "content": response1.choices[0].message.content},
            {"role": "user", "content": "Order #12345"}
        ]
    )

    # Both calls automatically tagged with:
    # - conversation_id = "support_session_456"
    # - user_id = "user_456"
    # All turns linked together for analysis!
```

### Complex Workflows

Track multi-step AI workflows with automatic tagging:

```python
from last9_genai import workflow_context

# RAG workflow example
with workflow_context(workflow_id="rag_pipeline", workflow_type="retrieval"):
    # Step 1: Query expansion (automatically tagged)
    expanded_query = expand_query(user_question)

    # Step 2: Retrieval (automatically tagged)
    documents = vector_search(expanded_query)

    # Step 3: Reranking (automatically tagged)
    relevant_docs = rerank(documents, user_question)

    # Step 4: Generation (automatically tagged)
    response = generate_answer(relevant_docs, user_question)

# All 4 steps automatically have:
# - workflow_id = "rag_pipeline"
# - workflow_type = "retrieval"
# Perfect for analyzing bottlenecks and performance!

### Nested Workflows and Conversations

Combine conversation and workflow tracking:

```python
# Track conversation
with conversation_context(conversation_id="user_session_789", user_id="user_789"):

    # Inside conversation, track a specific workflow
    with workflow_context(workflow_id="product_search", workflow_type="search"):
        # Search workflow steps
        results = search_products(query)
        recommendations = rank_results(results)

    # Outside workflow, still in conversation
    followup = handle_followup_question()

# Result:
# - search_products and rank_results: both conversation_id AND workflow_id
# - handle_followup_question: only conversation_id
# Perfect granularity for analysis!
```

### Tool/Function Tracking

Track tool calls:

```python
with tracer.start_span("gen_ai.tool.search") as span:
    l9.add_tool_attributes(
        span,
        tool_name="web_search",
        tool_type="search",
        arguments={"query": "weather"},
        result={"temp": 72},
        duration_ms=150
    )
```

## OpenTelemetry Integration

### Export to Last9

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.last9.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic YOUR_KEY"
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter()
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)
```

### Export to Console (Development)

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

console_exporter = ConsoleSpanExporter()
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(console_exporter)
)
```

## Configuration

### Disable Cost Tracking

```python
# Track tokens only, skip cost calculation
l9 = Last9GenAI(enable_cost_tracking=False)
```

### Custom Workflow Tracker

```python
from last9_genai import WorkflowCostTracker

tracker = WorkflowCostTracker()
l9 = Last9GenAI(workflow_tracker=tracker)
```

## Attributes Reference

### Standard OpenTelemetry (Always Set)

```python
gen_ai.system = "openai"
gen_ai.request.model = "gpt-4o"
gen_ai.usage.input_tokens = 150
gen_ai.usage.output_tokens = 250
```

### Last9 Extensions (Optional)

```python
# Cost (when pricing provided)
gen_ai.usage.cost_usd = 0.00225
gen_ai.usage.cost_input_usd = 0.000375
gen_ai.usage.cost_output_usd = 0.0025

# Classification
gen_ai.l9.span.kind = "llm"  # or "tool", "prompt"

# Workflow
workflow.id = "customer_support"
workflow.total_cost_usd = 0.015
workflow.llm_calls = 3

# Conversation
gen_ai.conversation.id = "session_123"
gen_ai.conversation.turn_number = 2
```

## Model Pricing

**No default pricing included.** You provide pricing for models you use.

### Finding Pricing

- **Anthropic**: https://www.anthropic.com/pricing
- **OpenAI**: https://openai.com/api/pricing/
- **Google**: https://ai.google.dev/pricing
- **Community**: https://www.llm-prices.com/

### Pricing Format

All prices in **USD per million tokens**:

```python
ModelPricing(
    input=3.0,   # $3 per 1M input tokens
    output=15.0  # $15 per 1M output tokens
)
```

**Conversion:**
- Per-token: `$0.000003` → `3.0`
- Per-1K: `$0.003` → `3.0`

### Common Models (February 2026)

```python
custom_pricing = {
    # Anthropic
    "claude-opus-4-6": ModelPricing(input=15.0, output=75.0),
    "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
    "claude-haiku-4-5": ModelPricing(input=0.8, output=4.0),

    # OpenAI
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
    "gpt-4o-mini": ModelPricing(input=0.15, output=0.60),
    "o1": ModelPricing(input=15.0, output=60.0),

    # Google
    "gemini-1.5-pro": ModelPricing(input=1.25, output=10.0),
    "gemini-2.0-flash": ModelPricing(input=0.075, output=0.30),
}
```

### Special Cases

**Azure OpenAI:**
```python
custom_pricing = {
    "azure/gpt-4o": ModelPricing(input=2.50, output=10.0),
}
```

**Self-hosted (free):**
```python
custom_pricing = {
    "ollama/llama3.1": ModelPricing(input=0.0, output=0.0),
}
```

**Fine-tuned:**
```python
custom_pricing = {
    "ft:gpt-3.5-turbo:org:model:id": ModelPricing(input=12.0, output=16.0),
}
```


## Examples

See [`examples/`](./examples/) directory:

**Basic Usage:**
- [`basic_usage.py`](./examples/basic_usage.py) - Simple LLM tracking
- [`openai_integration.py`](./examples/openai_integration.py) - OpenAI SDK
- [`anthropic_integration.py`](./examples/anthropic_integration.py) - Anthropic SDK
- [`langchain_integration.py`](./examples/langchain_integration.py) - LangChain
- [`fastapi_app.py`](./examples/fastapi_app.py) - FastAPI web app
- [`tool_integration.py`](./examples/tool_integration.py) - Function calls

**Auto-Tracking (Recommended):**
- [`context_tracking.py`](./examples/context_tracking.py) - Context managers for automatic tracking
- [`decorator_tracking.py`](./examples/decorator_tracking.py) - @observe() decorator pattern

**Advanced:**
- [`conversation_tracking.py`](./examples/conversation_tracking.py) - Multi-turn conversations

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests
4. Submit a PR

## License

MIT License - see [LICENSE](./LICENSE)

## Support

- **Issues**: https://github.com/last9/python-ai-sdk/issues
- **Documentation**: https://github.com/last9/python-ai-sdk
- **Last9**: https://last9.io

---

**Built with ❤️ by [Last9](https://last9.io)**
