#!/usr/bin/env python3
"""
@observe() decorator for automatic tracking

This example demonstrates automatic tracking of:
- Input/output (as span events)
- Latency (as span duration)
- Cost (calculated from usage)
- Metadata (from context)

Provides industry-standard decorator pattern for LLM observability.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import (
    observe,
    ModelPricing,
    conversation_context,
    workflow_context,
)


def setup_tracing():
    """Set up OpenTelemetry tracing"""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Add console exporter for demo
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    return trace.get_tracer(__name__)


# Define pricing
PRICING = {
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
    "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
}


# Mock LLM response (simulates OpenAI response structure)
class MockUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("Message", (), {"content": content})()
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, model, content, input_tokens, output_tokens):
        self.model = model
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(input_tokens, output_tokens)


def simulate_llm_call(model: str, prompt: str) -> MockResponse:
    """Simulate an LLM API call"""
    time.sleep(0.1)  # Simulate latency
    return MockResponse(
        model=model,
        content=f"Simulated response to: {prompt[:50]}...",
        input_tokens=len(prompt.split()) * 2,
        output_tokens=50,
    )


# ============================================================================
# Example 1: Basic @observe() usage
# ============================================================================


@observe(pricing=PRICING)
def call_gpt4(prompt: str):
    """Basic LLM call with automatic tracking"""
    return simulate_llm_call("gpt-4o", prompt)


def basic_example():
    """Basic decorator usage"""
    setup_tracing()

    print("\n🔄 Example 1: Basic @observe() decorator\n")

    response = call_gpt4("What is the capital of France?")
    print(f"   ✅ Response: {response.choices[0].message.content[:50]}...")

    print("\n   Automatically tracked:")
    print("      - Input: 'What is the capital of France?'")
    print("      - Output: Response content")
    print("      - Latency: ~100ms")
    print(f"      - Cost: ${(12 * 2.50 + 50 * 10.0) / 1_000_000:.6f}")
    print("      - Model: gpt-4o")


# ============================================================================
# Example 2: With conversation context
# ============================================================================


@observe(pricing=PRICING)
def chat_with_context(prompt: str):
    """LLM call that automatically picks up context"""
    return simulate_llm_call("gpt-4o", prompt)


def context_example():
    """Decorator with conversation context"""
    setup_tracing()

    print("\n🔄 Example 2: @observe() with conversation context\n")

    with conversation_context(conversation_id="user_session_123", user_id="user_123"):
        print("   Inside conversation context...")

        response1 = chat_with_context("Hello!")
        print(f"   ✅ Turn 1: {response1.choices[0].message.content[:40]}...")

        response2 = chat_with_context("Tell me more")
        print(f"   ✅ Turn 2: {response2.choices[0].message.content[:40]}...")

    print("\n   Both calls automatically have:")
    print("      - conversation_id = 'user_session_123'")
    print("      - user_id = 'user_123'")
    print("      - Input/output captured")
    print("      - Cost calculated")


# ============================================================================
# Example 3: Nested functions with workflow
# ============================================================================


@observe(pricing=PRICING, as_type="llm")
def search_query_generation(topic: str):
    """Generate a search query"""
    return simulate_llm_call("gpt-3.5-turbo", f"Generate search query for: {topic}")


@observe(pricing=PRICING, as_type="llm")
def summarize_results(results: str):
    """Summarize search results"""
    return simulate_llm_call("gpt-4o", f"Summarize: {results}")


@observe(as_type="chain", pricing=PRICING)
def search_workflow(topic: str):
    """Complete search workflow with multiple LLM calls"""
    # Generate query
    query_response = search_query_generation(topic)
    query = query_response.choices[0].message.content

    # Simulate search (not tracked)
    search_results = f"Mock search results for: {query}"

    # Summarize
    summary_response = summarize_results(search_results)

    return summary_response


def workflow_example():
    """Decorator with nested workflow"""
    setup_tracing()

    print("\n🔄 Example 3: Nested @observe() with workflow\n")

    with workflow_context(workflow_id="rag_search", workflow_type="search"):
        print("   Executing search workflow...")

        result = search_workflow("AI observability")
        print(f"   ✅ Result: {result.choices[0].message.content[:40]}...")

    print("\n   Automatically tracked:")
    print("      - 3 spans: search_workflow → search_query_generation → summarize_results")
    print("      - All have workflow_id = 'rag_search'")
    print("      - Each span has input/output/cost")
    print("      - Parent-child relationships preserved")


# ============================================================================
# Example 4: Async functions
# ============================================================================


@observe(pricing=PRICING)
async def async_llm_call(prompt: str):
    """Async LLM call with automatic tracking"""
    # Simulate async API call
    await asyncio.sleep(0.1)
    return simulate_llm_call("claude-3-5-sonnet", prompt)


async def async_example():
    """Decorator with async functions"""
    setup_tracing()

    print("\n🔄 Example 4: @observe() with async functions\n")

    print("   Making async LLM call...")
    response = await async_llm_call("Explain async/await")
    print(f"   ✅ Response: {response.choices[0].message.content[:40]}...")

    print("\n   Works seamlessly with async!")
    print("      - Async/await fully supported")
    print("      - Context propagates across await boundaries")


# ============================================================================
# Example 5: Custom span names and types
# ============================================================================


@observe(name="customer_support_llm", as_type="llm", pricing=PRICING)
def support_bot(customer_message: str):
    """Customer support bot with custom span name"""
    return simulate_llm_call("gpt-4o", f"Customer: {customer_message}")


@observe(name="database_query", as_type="tool", capture_input=True, capture_output=False)
def lookup_order(order_id: str):
    """Tool call (non-LLM) with custom tracking"""
    time.sleep(0.05)
    return {"order_id": order_id, "status": "shipped"}


@observe(as_type="chain", pricing=PRICING)
def customer_support_workflow(customer_message: str, order_id: str):
    """Complete support workflow"""
    # Look up order (tool call)
    order = lookup_order(order_id)

    # Generate response (LLM call)
    response = support_bot(f"{customer_message}. Order status: {order['status']}")

    return response


def custom_example():
    """Decorator with custom names and types"""
    setup_tracing()

    print("\n🔄 Example 5: Custom span names and types\n")

    print("   Executing customer support workflow...")
    response = customer_support_workflow("Where is my order?", "ORD-12345")
    print(f"   ✅ Response: {response.choices[0].message.content[:40]}...")

    print("\n   Span hierarchy:")
    print("      - customer_support_workflow (chain)")
    print("         ├─ database_query (tool)")
    print("         └─ customer_support_llm (llm)")


# ============================================================================
# Example 6: Error handling
# ============================================================================


@observe(pricing=PRICING)
def failing_llm_call(prompt: str):
    """LLM call that fails"""
    raise Exception("Simulated API error")


def error_example():
    """Decorator with error handling"""
    setup_tracing()

    print("\n🔄 Example 6: Error handling\n")

    try:
        print("   Making failing LLM call...")
        failing_llm_call("This will fail")
    except Exception as e:
        print(f"   ✅ Error caught: {e}")

    print("\n   Error automatically recorded:")
    print("      - Exception details in span")
    print("      - Span status set to ERROR")
    print("      - Stack trace captured")


# ============================================================================
# Example 7: Selective capture
# ============================================================================


@observe(pricing=PRICING, capture_input=True, capture_output=False)
def sensitive_output_call(prompt: str):
    """Only capture input, not output (for sensitive data)"""
    return simulate_llm_call("gpt-4o", prompt)


@observe(pricing=PRICING, capture_input=False, capture_output=True)
def sensitive_input_call(secret_prompt: str):
    """Only capture output, not input (for secrets)"""
    return simulate_llm_call("gpt-4o", secret_prompt)


@observe(pricing=PRICING, capture_args=False)
def no_args_capture(api_key: str, prompt: str):
    """Don't capture function arguments (for secrets in args)"""
    return simulate_llm_call("gpt-4o", prompt)


def selective_example():
    """Selective capture for sensitive data"""
    setup_tracing()

    print("\n🔄 Example 7: Selective capture (privacy)\n")

    print("   Call 1: Capture input only...")
    sensitive_output_call("Public question")
    print("   ✅ Input captured, output hidden")

    print("\n   Call 2: Capture output only...")
    sensitive_input_call("secret_password_123")
    print("   ✅ Input hidden, output captured")

    print("\n   Call 3: No argument capture...")
    no_args_capture("sk-secret-key", "Public question")
    print("   ✅ Arguments hidden (api_key not logged)")


if __name__ == "__main__":
    print("Last9 GenAI - @observe() Decorator Examples")
    print("=" * 70)
    print("\nAutomatic tracking of input/output, latency, cost, and metadata.\n")

    try:
        # Run all examples
        basic_example()
        context_example()
        workflow_example()
        asyncio.run(async_example())
        custom_example()
        error_example()
        selective_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n" + "=" * 70)
        print("✅ All @observe() decorator examples completed!")
        print("\n📊 What was automatically tracked:")
        print("   ✅ Input/output (as span events)")
        print("   ✅ Latency (as span duration)")
        print("   ✅ Cost per generation (calculated from usage)")
        print("   ✅ Metadata (conversation_id, workflow_id, user_id)")
        print("   ✅ Model, system, finish_reason")
        print("   ✅ Function arguments")
        print("   ✅ Error details")
        print("\n🎯 Zero manual instrumentation required!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
