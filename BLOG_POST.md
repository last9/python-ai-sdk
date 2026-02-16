# Beyond Trace IDs: Why LLM Observability Needs Conversation Tracking

**TL;DR:** OpenTelemetry's GenAI instrumentation gives you traces, but conversations span multiple traces. We built an SDK that bridges this gap with conversation IDs, automatic cost tracking, and context propagation - without replacing your existing OTel setup.

---

## The Problem: Conversations Don't Fit Into Traces

If you're building LLM applications, you've probably tried instrumenting them with OpenTelemetry. The GenAI semantic conventions are great - they give you `gen_ai.request.model`, `gen_ai.usage.input_tokens`, and all the standard attributes you need.

But here's what they **don't** give you:

### 1. **Conversation Tracking Across Traces**

A user opens your chatbot and has a 10-turn conversation. With standard OTel instrumentation, you get:

```
Trace 1: User asks "What's the weather?"
  └─ Span: OpenAI API call

Trace 2: User asks "Will I need an umbrella?"
  └─ Span: OpenAI API call

Trace 3: User asks "What about tomorrow?"
  └─ Span: OpenAI API call
```

**The problem:** Each request is a separate trace with a different `trace_id`. You can't answer:
- "What was the full conversation?"
- "How much did this entire conversation cost?"
- "What's the average cost per conversation for user X?"

You're stuck with individual traces that don't connect to the **business concept** of a conversation.

### 2. **No Built-in Cost Tracking**

OTel GenAI gives you token counts (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`), but:
- ❌ No cost calculation (even though you know your pricing)
- ❌ No cost aggregation across multiple LLM calls
- ❌ No way to track: "This RAG workflow cost $0.47"

You end up writing custom logic to:
1. Extract token counts from traces
2. Multiply by your model pricing
3. Aggregate costs across related spans
4. Store and query this data separately

### 3. **No Workflow-Level Context**

Your RAG pipeline has multiple steps:
```python
def handle_query(user_query):
    # Step 1: Retrieve documents (embedding call)
    docs = retrieve_documents(user_query)

    # Step 2: Rerank (another LLM call)
    relevant_docs = rerank(docs, user_query)

    # Step 3: Generate response (GPT-4 call)
    response = generate_response(relevant_docs, user_query)

    return response
```

With standard OTel, you get three separate spans, but:
- ❌ No way to tag them all as part of "RAG workflow ID: xyz"
- ❌ No automatic cost aggregation across the workflow
- ❌ Manual span correlation required in your observability backend

---

## Why This Matters for AI Engineers

When you're building production LLM applications, you need to answer questions like:

**Business Questions:**
- "What's our average cost per conversation?"
- "Which users are having the most expensive conversations?"
- "How much did we spend on GPT-4 vs Claude this week?"

**Engineering Questions:**
- "This conversation went off the rails - what was the full context?"
- "Our costs spiked - which workflows are responsible?"
- "How do I debug this multi-step RAG pipeline?"

**OpenTelemetry gives you traces and spans, but you need *conversations* and *workflows*.**

---

## The Solution: Conversation IDs + Cost Tracking

We built `last9-genai` as an **extension** to OpenTelemetry that adds:

1. **Conversation tracking** - Group all spans from a multi-turn conversation
2. **Workflow tracking** - Track complex multi-step AI operations
3. **Automatic cost tracking** - Calculate costs from token usage + your pricing
4. **Context propagation** - Automatically tag all nested operations

**Important:** This is not a replacement for OTel GenAI instrumentation. It works **alongside** it (or standalone).

---

## How It Works: Code Examples

### Example 1: Track a Multi-Turn Conversation

```python
from last9_genai import conversation_context, observe, ModelPricing
from openai import OpenAI

client = OpenAI()

# Define your pricing (USD per million tokens)
pricing = {
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
}

@observe(pricing=pricing)
def chat(message: str, conversation_id: str):
    """Each call is automatically tracked with conversation context"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# User has a 3-turn conversation
with conversation_context(conversation_id="user_123_session_456", user_id="user_123"):
    # Turn 1
    response1 = chat("What's the weather in SF?", "user_123_session_456")

    # Turn 2
    response2 = chat("Will I need an umbrella?", "user_123_session_456")

    # Turn 3
    response3 = chat("What about tomorrow?", "user_123_session_456")
```

**What you get:**

All three LLM calls are automatically tagged with:
```json
{
  "gen_ai.conversation.id": "user_123_session_456",
  "gen_ai.conversation.turn": 1,  // auto-incremented
  "gen_ai.usage.cost_usd": 0.000234,  // calculated automatically
  "user.id": "user_123",
  "gen_ai.l9.span.kind": "llm"  // span classification
}
```

**Now you can query:** "Show me all spans where `conversation_id = user_123_session_456`" - and you get the complete conversation, **even though they're in separate traces**.

---

### Example 2: Track Costs Across a Multi-Step Workflow

```python
from last9_genai import workflow_context, observe, ModelPricing

pricing = {
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
    "text-embedding-3-small": ModelPricing(input=0.02, output=0.0),
}

@observe(span_kind="tool", pricing=pricing)
def retrieve_documents(query: str):
    # Embedding API call
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    # Search vector DB...
    return ["doc1", "doc2", "doc3"]

@observe(pricing=pricing)
def rerank_documents(docs, query):
    # LLM call to rerank
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "Rerank these documents by relevance"
        }]
    )
    return response

@observe(pricing=pricing)
def generate_response(docs, query):
    # Final LLM call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"Use these docs: {docs}"
        }]
    )
    return response.choices[0].message.content

# Track the entire RAG workflow
with workflow_context(workflow_id="rag_12345", workflow_type="rag"):
    docs = retrieve_documents("What is RAG?")           # Cost: $0.000002
    ranked = rerank_documents(docs, "What is RAG?")     # Cost: $0.000123
    answer = generate_response(ranked, "What is RAG?")  # Cost: $0.000456

# Total workflow cost: $0.000581 (automatically calculated)
```

**What you get:**

All spans tagged with:
```json
{
  "workflow.id": "rag_12345",
  "workflow.type": "rag",
  "gen_ai.usage.cost_usd": 0.000581,  // aggregated across all steps
  "gen_ai.l9.span.kind": "tool"  // or "llm" depending on operation
}
```

**Now you can answer:**
- "Show me all RAG workflows that cost more than $1"
- "What's the average cost of our RAG pipeline?"
- "Which workflow ID had the error?"

---

### Example 3: Nested Conversations + Workflows

Real applications have both - a multi-turn conversation where each turn triggers a workflow:

```python
with conversation_context(conversation_id="support_session_789", user_id="user_456"):
    # Turn 1: User asks question
    with workflow_context(workflow_id="rag_001", workflow_type="rag"):
        docs = retrieve_documents(user_query)
        response1 = generate_response(docs, user_query)

    # Turn 2: Follow-up question
    with workflow_context(workflow_id="rag_002", workflow_type="rag"):
        docs = retrieve_documents(followup_query)
        response2 = generate_response(docs, followup_query)
```

**All spans automatically inherit:**
```json
{
  "gen_ai.conversation.id": "support_session_789",
  "user.id": "user_456",
  "workflow.id": "rag_001",  // or rag_002
  "workflow.type": "rag",
  "gen_ai.conversation.turn": 1  // or 2
}
```

**Context propagates automatically** - no need to manually pass IDs through your call stack.

---

## Key Benefits: Why This Approach Works

### 1. **Bridges the Trace ID Gap**

Traditional OTel observability is **trace-centric**:
```
trace_id: abc123  →  Single request/response
```

But LLM applications are **conversation-centric**:
```
conversation_id: user_session_456  →  Multiple traces over time
```

**`conversation_id` becomes your primary grouping dimension**, letting you:
- Debug full conversations, not just individual API calls
- Calculate conversation-level metrics (cost, latency, token usage)
- Track user behavior across sessions

### 2. **Automatic Cost Tracking**

You bring your model pricing, we handle the calculation:

```python
pricing = {
    "gpt-4o": ModelPricing(input=2.50, output=10.0),  # USD per million tokens
}

@observe(pricing=pricing)
def chat(message):
    # SDK automatically:
    # 1. Extracts token counts from response
    # 2. Calculates cost: (input_tokens/1M * 2.50) + (output_tokens/1M * 10.0)
    # 3. Adds gen_ai.usage.cost_usd attribute to span
    return client.chat.completions.create(...)
```

No custom backends, no post-processing - **cost is a first-class metric in your spans**.

### 3. **Works With Your Existing Stack**

This is an **OpenTelemetry extension**, not a proprietary solution:

- ✅ Uses standard OTel SDK (TracerProvider, SpanProcessor)
- ✅ Exports to any OTLP backend (Last9, Datadog, Honeycomb, etc.)
- ✅ Works alongside OTel auto-instrumentation
- ✅ No vendor lock-in

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from last9_genai import Last9SpanProcessor

# Use any OTLP backend
otlp_exporter = OTLPSpanExporter(endpoint="https://your-backend.com")
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Add conversation/workflow tracking
provider.add_span_processor(Last9SpanProcessor())
```

### 4. **Zero-Touch Instrumentation**

Use the `@observe()` decorator for automatic tracking:

```python
@observe(pricing=pricing)
def my_llm_function(prompt: str):
    # Automatically tracked:
    # - Input/output capture
    # - Latency
    # - Token usage
    # - Cost calculation
    # - Error handling
    # - Conversation context (if inside conversation_context)
    return client.chat.completions.create(...)
```

No manual span creation, no boilerplate.

---

## Real-World Use Cases

### **Use Case 1: Debug Production Conversations**

**Problem:** User reports "the chatbot gave a weird response"

**Without conversation tracking:**
```
Search for: user_id = "user_123" AND timestamp ~= "2024-01-15"
→ Get 47 traces
→ Manually correlate which traces belong to the problematic conversation
→ Miss context from earlier turns
```

**With conversation tracking:**
```
Search for: conversation_id = "user_123_session_456"
→ Get all 8 turns of the conversation in order
→ See full context: what the user asked, what the model responded
→ Identify that turn 6 had incorrect context from turn 2
```

### **Use Case 2: Cost Attribution**

**Problem:** You need to charge customers based on LLM usage

**Without cost tracking:**
```
1. Export all traces to data warehouse
2. Parse token counts from span attributes
3. Join with pricing table
4. Calculate costs per trace
5. Group by customer_id
6. Generate invoice
```

**With cost tracking:**
```
Query: SELECT SUM(gen_ai.usage.cost_usd)
       WHERE customer_id = "acme_corp"
       AND date = "2024-01"
→ $247.32
```

### **Use Case 3: Optimize RAG Pipelines**

**Problem:** RAG pipeline costs are too high

**With workflow tracking:**
```
Query: AVG(gen_ai.usage.cost_usd) GROUP BY workflow.type
→ rag_pipeline_v1: $0.42 per query
→ rag_pipeline_v2: $0.18 per query

Query: Show me spans WHERE workflow.type = "rag_pipeline_v1"
       AND gen_ai.usage.cost_usd > 0.50
→ See which queries are expensive
→ Identify that reranking step is the culprit
→ Optimize or remove reranking
```

---

## Getting Started

### Installation

```bash
# From GitHub (available now)
pip install git+https://github.com/last9/python-ai-sdk.git@v1.0.0

# Or from PyPI (coming soon)
pip install last9-genai[otlp]
```

### Basic Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from last9_genai import Last9SpanProcessor, conversation_context, observe, ModelPricing

# Set up OpenTelemetry
provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(endpoint="https://otlp.your-backend.io")
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Add Last9 processor for conversation/cost tracking
provider.add_span_processor(Last9SpanProcessor())

trace.set_tracer_provider(provider)

# Define your pricing
pricing = {
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
}

# Start tracking!
with conversation_context(conversation_id="user_session", user_id="user_123"):
    @observe(pricing=pricing)
    def chat(message):
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}]
        )
```

---

## Comparison: OTel GenAI vs Last9 GenAI

| Feature | OTel GenAI Instrumentation | Last9 GenAI SDK |
|---------|---------------------------|-----------------|
| **Auto-instrument LLM SDKs** | ✅ Yes (monkey-patching) | ❌ No (manual decorator) |
| **Standard gen_ai.* attributes** | ✅ Yes | ✅ Yes |
| **Conversation tracking** | ❌ No | ✅ Yes (`conversation_id`) |
| **Multi-turn tracking** | ❌ No | ✅ Yes (auto turn counter) |
| **Workflow tracking** | ❌ No | ✅ Yes (`workflow_id`) |
| **Cost calculation** | ❌ No | ✅ Yes (with your pricing) |
| **Cost aggregation** | ❌ No | ✅ Yes (workflow-level) |
| **Context propagation** | ❌ Manual | ✅ Automatic (thread-safe) |
| **Span classification** | ❌ No | ✅ Yes (llm/tool/chain/agent) |
| **Custom attributes** | ✅ Manual | ✅ Automatic propagation |

**Recommendation:** Use both together!
- OTel GenAI auto-instruments your LLM SDKs (traces every API call)
- Last9 GenAI adds conversation/workflow context and cost tracking

---

## What's Next?

We built this SDK because we needed it at Last9 for our own LLM applications. The gap between "traces" and "conversations" was too painful, and cost tracking was essential for production.

**Try it out:**
- GitHub: https://github.com/last9/python-ai-sdk
- Examples: https://github.com/last9/python-ai-sdk/tree/main/examples
- Installation: https://github.com/last9/python-ai-sdk#installation

**Questions? Feedback?**
- Open an issue: https://github.com/last9/python-ai-sdk/issues
- Tweet at us: [@last9io](https://twitter.com/last9io)

---

## Key Takeaways

1. **LLM observability is different** - conversations span multiple traces, traditional trace_id isn't enough
2. **Conversation ID is your primary dimension** - group all related spans across traces
3. **Cost is a first-class metric** - calculate it automatically from token usage + your pricing
4. **Context propagates automatically** - no need to pass IDs through your call stack
5. **Works with existing OTel** - it's an extension, not a replacement

Build better LLM applications with proper observability. 🚀

---

*Written by the Last9 team - making observability suck less, one SDK at a time.*
