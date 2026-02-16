#!/usr/bin/env python3
"""
Context-based automatic tracking example

This example demonstrates automatic conversation and workflow tracking
using context managers for zero-touch instrumentation.

This is the RECOMMENDED approach for production applications as it requires
minimal code changes and automatically propagates context to all nested operations.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import (
    Last9SpanProcessor,
    ModelPricing,
    conversation_context,
    workflow_context,
    propagate_attributes,
)


def setup_tracing():
    """Set up OpenTelemetry tracing with Last9 auto-enrichment"""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Add console exporter for demo
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Add Last9 processor for automatic enrichment
    custom_pricing = {
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
    }

    l9_processor = Last9SpanProcessor(custom_pricing=custom_pricing, enable_cost_tracking=True)
    provider.add_span_processor(l9_processor)

    return trace.get_tracer(__name__)


def simulate_llm_call(tracer, model: str, prompt: str) -> dict:
    """Simulate an LLM API call"""
    with tracer.start_span("gen_ai.chat.completions") as span:
        # Simulate API call
        time.sleep(0.1)

        # Add standard OTel attributes (normally done by auto-instrumentation)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.system", "openai" if "gpt" in model else "anthropic")
        span.set_attribute("gen_ai.operation.name", "chat")

        # Simulate token usage
        input_tokens = len(prompt.split()) * 2
        output_tokens = 50
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        # Last9SpanProcessor automatically adds:
        # - gen_ai.conversation.id (from context)
        # - workflow.id (from context)
        # - gen_ai.usage.cost_usd (calculated)
        # - user.id (from context)

        return {
            "response": f"Simulated response to: {prompt[:50]}...",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def basic_context_example():
    """Basic conversation context example"""
    tracer = setup_tracing()

    print("\n🔄 Example 1: Basic conversation context\n")

    # Set conversation context once
    with conversation_context(conversation_id="user_123_session", user_id="user_123"):
        print("   Inside conversation context...")

        # All LLM calls automatically tagged
        result1 = simulate_llm_call(tracer, "gpt-4o", "What is the capital of France?")
        print(f"   ✅ Call 1: {result1['response'][:50]}...")

        result2 = simulate_llm_call(tracer, "gpt-4o", "Tell me more about Paris")
        print(f"   ✅ Call 2: {result2['response'][:50]}...")

        # Both calls have conversation_id and user_id automatically!

    print("   Done! Both calls automatically have:")
    print("      - gen_ai.conversation.id = 'user_123_session'")
    print("      - user.id = 'user_123'")
    print("      - gen_ai.usage.cost_usd (calculated)")


def nested_workflow_example():
    """Nested workflow within conversation"""
    tracer = setup_tracing()

    print("\n🔄 Example 2: Nested workflow context\n")

    with conversation_context(conversation_id="support_session_456", user_id="user_456"):
        print("   Inside conversation context...")

        # Initial query
        simulate_llm_call(tracer, "gpt-4o", "I need help with my order")
        print("   ✅ Initial query processed")

        # Nested workflow for order lookup
        with workflow_context(workflow_id="order_lookup", workflow_type="customer_support"):
            print("   Inside workflow context...")

            # These calls have BOTH conversation AND workflow context
            simulate_llm_call(tracer, "gpt-3.5-turbo", "Look up order #12345")
            print("   ✅ Order lookup completed")

            simulate_llm_call(tracer, "gpt-3.5-turbo", "Generate order summary")
            print("   ✅ Summary generated")

        # Back to conversation only
        simulate_llm_call(tracer, "gpt-4o", "Provide final response to customer")
        print("   ✅ Final response sent")

    print("\n   Workflow calls automatically have:")
    print("      - gen_ai.conversation.id = 'support_session_456'")
    print("      - workflow.id = 'order_lookup'")
    print("      - workflow.type = 'customer_support'")
    print("      - user.id = 'user_456'")


def propagate_attributes_example():
    """Using propagate_attributes directly"""
    tracer = setup_tracing()

    print("\n🔄 Example 3: propagate_attributes pattern\n")

    def chat_handler(session_id: str, user_id: str, message: str):
        """Simulated chat handler function"""
        # Set attributes once - they propagate to all nested operations
        propagate_attributes(conversation_id=session_id, user_id=user_id, custom_channel="web_chat")

        # All these automatically get the attributes
        simulate_llm_call(tracer, "claude-3-5-sonnet", message)
        simulate_llm_call(tracer, "claude-3-5-sonnet", "Follow-up question")

    # Call the handler
    print("   Calling chat_handler...")
    chat_handler("web_session_789", "user_789", "Hello, I need assistance")
    print("   ✅ Handler completed")

    print("\n   All calls automatically have:")
    print("      - gen_ai.conversation.id = 'web_session_789'")
    print("      - user.id = 'user_789'")
    print("      - custom.custom_channel = 'web_chat'")


def fastapi_pattern_example():
    """Example showing FastAPI integration pattern"""
    tracer = setup_tracing()

    print("\n🔄 Example 4: FastAPI integration pattern\n")

    # Simulated FastAPI endpoint
    def simulated_chat_endpoint(session_id: str, user_id: str, message: str):
        """Simulated FastAPI endpoint"""

        # Set context once per request
        with conversation_context(conversation_id=session_id, user_id=user_id):

            # Check if this needs a search workflow
            if "search" in message.lower():
                with workflow_context(workflow_id="rag_search", workflow_type="search"):
                    print("      🔍 Executing RAG search workflow...")
                    simulate_llm_call(tracer, "gpt-3.5-turbo", "Generate search query")
                    simulate_llm_call(tracer, "gpt-3.5-turbo", "Rerank results")
                    print("      ✅ Search complete")

            # Generate final response
            result = simulate_llm_call(tracer, "gpt-4o", message)
            return result

    print("   Simulating /chat endpoint request...")
    result = simulated_chat_endpoint(
        "api_session_999", "user_999", "Search for best restaurants in Paris"
    )
    print(f"   ✅ Response: {result['response'][:50]}...")

    print("\n   With just ONE context manager:")
    print("      - All calls tagged with conversation_id")
    print("      - Search workflow calls also have workflow.id")
    print("      - Zero manual attribute setting!")


def multi_turn_conversation_example():
    """Example with turn numbers"""
    tracer = setup_tracing()

    print("\n🔄 Example 5: Multi-turn conversation with turn tracking\n")

    conversation_id = "multi_turn_session"
    messages = ["Hello!", "What's the weather?", "Thank you!"]

    for turn_num, message in enumerate(messages, start=1):
        with conversation_context(
            conversation_id=conversation_id, user_id="user_multi", turn_number=turn_num
        ):
            print(f"   Turn {turn_num}: {message}")
            simulate_llm_call(tracer, "gpt-4o", message)
            print(f"   ✅ Turn {turn_num} complete")

    print("\n   Each turn automatically has:")
    print("      - gen_ai.conversation.id (same across all turns)")
    print("      - gen_ai.conversation.turn_number (1, 2, 3...)")


if __name__ == "__main__":
    print("Last9 GenAI - Context-Based Automatic Tracking")
    print("=" * 70)
    print("\nThis demonstrates automatic attribute propagation for LLM observability.")
    print("All examples use context managers for zero-touch instrumentation.\n")

    try:
        # Run all examples
        basic_context_example()
        nested_workflow_example()
        propagate_attributes_example()
        fastapi_pattern_example()
        multi_turn_conversation_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n" + "=" * 70)
        print("✅ All context tracking examples completed!")
        print("\n📊 Key Benefits:")
        print("   ✅ Set context once, applies everywhere")
        print("   ✅ No need to pass IDs through call stack")
        print("   ✅ Works across async boundaries")
        print("   ✅ Nested workflows supported")
        print("   ✅ Automatic cost calculation")
        print("\n🎯 Industry-standard context propagation with OpenTelemetry!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
