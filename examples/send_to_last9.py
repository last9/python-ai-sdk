#!/usr/bin/env python3
"""
Example: Send telemetry data to Last9

This script demonstrates how to configure OpenTelemetry to send
GenAI telemetry data to Last9's OTLP endpoint.

Prerequisites:
1. Install OTLP exporter: pip install last9-genai[otlp]
2. Set Last9 credentials:
   - LAST9_OTLP_ENDPOINT (e.g., https://otlp.last9.io)
   - LAST9_API_KEY or OTEL_EXPORTER_OTLP_HEADERS

Usage:
    export LAST9_OTLP_ENDPOINT="https://otlp.last9.io"
    export LAST9_API_KEY="your-api-key"
    python examples/send_to_last9.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

# Import OTLP exporter
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    OTLP_AVAILABLE = True
except ImportError:
    print("❌ OTLP exporter not installed!")
    print("   Install with: pip install last9-genai[otlp]")
    OTLP_AVAILABLE = False
    sys.exit(1)

from last9_genai import (
    Last9SpanProcessor,
    conversation_context,
    workflow_context,
    observe,
    ModelPricing,
)


def setup_last9_export():
    """
    Configure OpenTelemetry to export to Last9 OTLP endpoint

    This sets up:
    1. Service resource identification
    2. OTLP exporter with Last9 endpoint and authentication
    3. Last9SpanProcessor for automatic attribute enrichment
    4. BatchSpanProcessor for efficient export
    """
    # Get Last9 configuration from environment
    otlp_endpoint = os.getenv("LAST9_OTLP_ENDPOINT")
    api_key = os.getenv("LAST9_API_KEY")

    if not otlp_endpoint:
        print("❌ LAST9_OTLP_ENDPOINT not set!")
        print("   Set it with: export LAST9_OTLP_ENDPOINT='https://otlp.last9.io'")
        sys.exit(1)

    if not api_key:
        print("⚠️  LAST9_API_KEY not set - using unauthenticated (may not work)")
        print("   Set it with: export LAST9_API_KEY='your-api-key'")

    print(f"🔧 Configuring Last9 export...")
    print(f"   Endpoint: {otlp_endpoint}")
    print(f"   Authentication: {'✅ Configured' if api_key else '❌ Missing'}")

    # Create resource with service identification
    resource = Resource.create(
        {
            SERVICE_NAME: "last9-genai-python-example",
            SERVICE_VERSION: "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        }
    )

    # Create tracer provider with resource
    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter with Last9 endpoint
    # gRPC metadata format: tuple of (key, value) pairs
    headers = ()
    if api_key:
        # Last9 typically uses 'authorization' header with API key
        # Adjust the header name/format based on Last9's requirements
        headers = (("authorization", f"Basic {api_key}"),)

    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers=headers,
    )

    # Add Last9SpanProcessor for automatic attribute enrichment
    provider.add_span_processor(Last9SpanProcessor())

    # Add BatchSpanProcessor for efficient export to Last9
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    print("✅ Last9 export configured successfully!\n")

    return trace.get_tracer(__name__)


def simulate_llm_response(model: str, prompt: str):
    """Simulate an LLM response (replace with real API call)"""
    input_tokens = len(prompt.split()) * 2
    output_tokens = 100

    return {
        "id": "msg_demo_123",
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "content": f"Simulated response to: {prompt[:50]}...",
        "finish_reason": "stop",
    }


@observe(
    pricing={
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "claude-3-5-sonnet-20241022": ModelPricing(input=3.0, output=15.0),
    },
    capture_input=True,
    capture_output=True,
)
def chat_with_llm(model: str, prompt: str):
    """Decorated function that automatically creates spans"""
    return simulate_llm_response(model, prompt)


def example_conversation():
    """Example: Multi-turn conversation tracked in Last9"""
    print("📊 Example 1: Multi-turn conversation tracking")
    print("-" * 50)

    conversation_id = "demo_conversation_001"
    user_id = "demo_user_alice"

    with conversation_context(conversation_id=conversation_id, user_id=user_id):
        # Turn 1
        print("👤 User: Tell me about quantum computing")
        response1 = chat_with_llm("gpt-4o", "Tell me about quantum computing")
        print(f"🤖 Assistant: {response1['content'][:80]}...\n")

        # Turn 2
        print("👤 User: How is it different from classical computing?")
        response2 = chat_with_llm("gpt-4o", "How is it different from classical computing?")
        print(f"🤖 Assistant: {response2['content'][:80]}...\n")

        # Turn 3
        print("👤 User: What are the practical applications?")
        response3 = chat_with_llm("gpt-4o", "What are the practical applications?")
        print(f"🤖 Assistant: {response3['content'][:80]}...\n")

    print(f"✅ Conversation {conversation_id} completed - 3 turns tracked")
    print(f"   View in Last9 dashboard filtered by conversation_id={conversation_id}\n")


def example_rag_workflow():
    """Example: RAG workflow with retrieval, context, and generation"""
    print("📊 Example 2: RAG workflow tracking")
    print("-" * 50)

    workflow_id = "demo_rag_workflow_001"

    @observe(as_type="tool", name="retrieve_documents")
    def retrieve_docs(query: str):
        print(f"   🔍 Retrieving documents for: {query}")
        return [
            {"doc_id": 1, "text": "Quantum computers use qubits..."},
            {"doc_id": 2, "text": "Superposition allows qubits to..."},
        ]

    @observe(as_type="tool", name="build_context")
    def build_context(documents: list):
        print(f"   📝 Building context from {len(documents)} documents")
        return "\n".join([doc["text"] for doc in documents])

    @observe(
        pricing={"claude-3-5-sonnet-20241022": ModelPricing(input=3.0, output=15.0)},
        name="generate_answer",
    )
    def generate_answer(context: str, query: str):
        print(f"   🤖 Generating answer...")
        prompt = f"Context: {context}\n\nQuestion: {query}"
        return simulate_llm_response("claude-3-5-sonnet-20241022", prompt)

    with workflow_context(workflow_id=workflow_id, workflow_type="rag_search"):
        user_query = "How do quantum computers work?"
        print(f"👤 User query: {user_query}\n")

        # RAG pipeline
        docs = retrieve_docs(user_query)
        context = build_context(docs)
        answer = generate_answer(context, user_query)

        print(f"\n🤖 Final answer: {answer['content'][:80]}...\n")

    print(f"✅ RAG workflow {workflow_id} completed")
    print(f"   View in Last9 dashboard filtered by workflow_id={workflow_id}\n")


def example_nested_contexts():
    """Example: Conversation nested in workflow"""
    print("📊 Example 3: Nested conversation + workflow")
    print("-" * 50)

    with conversation_context(conversation_id="support_conv_123", user_id="customer_bob"):
        with workflow_context(workflow_id="support_ticket_456", workflow_type="customer_support"):
            print("   🎫 Processing support ticket...")

            # Tool call
            @observe(as_type="tool", name="check_account_status")
            def check_account():
                print("   🔍 Checking account status...")
                return {"status": "active", "plan": "premium"}

            account_info = check_account()

            # LLM call
            print("   🤖 Generating support response...")
            response = chat_with_llm("gpt-4o", f"Generate support response for {account_info}")

            print(f"\n🤖 Support response: {response['content'][:80]}...\n")

    print("✅ Support ticket processed")
    print("   View in Last9 with both conversation_id AND workflow_id filters\n")


def main():
    """Main function to run examples"""
    print("\n" + "=" * 60)
    print("🚀 Last9 GenAI Python SDK - Send Telemetry to Last9")
    print("=" * 60 + "\n")

    if not OTLP_AVAILABLE:
        return

    # Setup Last9 export
    tracer = setup_last9_export()

    print("📡 Sending telemetry data to Last9...\n")
    print("=" * 60 + "\n")

    try:
        # Run examples
        example_conversation()
        example_rag_workflow()
        example_nested_contexts()

        # Force flush to ensure all spans are sent
        print("⏳ Flushing spans to Last9...")
        trace.get_tracer_provider().force_flush(timeout_millis=10000)

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\n📊 View your telemetry in Last9 dashboard:")
        print("   • Filter by service: last9-genai-python-example")
        print("   • Filter by conversation_id: demo_conversation_001")
        print("   • Filter by workflow_id: demo_rag_workflow_001")
        print("   • Filter by gen_ai.l9.span.kind: llm, tool")
        print("\n💰 Cost tracking:")
        print("   • View gen_ai.usage.cost_usd attribute")
        print("   • Aggregate by workflow_id for workflow costs")
        print("   • Track per-conversation costs with conversation_id")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
