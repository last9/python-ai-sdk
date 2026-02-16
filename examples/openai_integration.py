#!/usr/bin/env python3
"""
Integration example with OpenAI Python SDK

This example shows how to integrate Last9 GenAI attributes with the official
OpenAI Python SDK for comprehensive observability and cost tracking.

Install dependencies:
    pip install openai opentelemetry-api opentelemetry-sdk last9-genai
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import ModelPricing, Last9GenAI, create_llm_span, SpanKinds

# Comment out if you don't have openai installed
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI SDK not installed. Install with: pip install openai")


def setup_tracing():
    """Set up OpenTelemetry tracing"""
    trace.set_tracer_provider(TracerProvider())

    # Use console exporter for demo - replace with OTLP exporter for production
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return trace.get_tracer(__name__)


def chat_with_observability(client, model: str, messages: list, workflow_id: str = None):
    """
    OpenAI chat call with full Last9 observability

    Args:
        client: OpenAI client
        model: Model name (e.g., 'gpt-4o', 'gpt-3.5-turbo')
        messages: List of message dictionaries
        workflow_id: Optional workflow ID for cost tracking

    Returns:
        Tuple of (response, cost_breakdown)
    """
    tracer = trace.get_tracer(__name__)
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    with tracer.start_span("gen_ai.chat.completions") as span:
        start_time = time.time()

        # Extract prompt for analysis
        prompt_content = ""
        for msg in messages:
            prompt_content += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"

        # Add standard and Last9 attributes before the call
        l9_genai.add_standard_llm_attributes(
            span,
            model,
            operation="chat.completions",
            conversation_id=workflow_id or "default_session",
        )
        l9_genai.set_span_kind(span, SpanKinds.LLM)

        try:
            # Make the actual OpenAI API call
            if OPENAI_AVAILABLE:
                response = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=1000, temperature=0.7
                )

                # Calculate response time
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000

                # Extract usage information from OpenAI response
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                # Extract response content
                response_content = response.choices[0].message.content

                # Add response attributes
                l9_genai.add_standard_llm_attributes(
                    span,
                    model,
                    request_params={"max_tokens": 1000, "temperature": 0.7},
                    response_data={
                        "id": response.id,
                        "model": response.model,
                        "finish_reason": response.choices[0].finish_reason,
                    },
                    usage=usage,
                )

                # Add Last9 cost tracking
                cost = l9_genai.add_llm_cost_attributes(span, model, usage, workflow_id)

                # Add performance metrics
                l9_genai.add_performance_attributes(
                    span,
                    response_time_ms=response_time_ms,
                    request_size_bytes=len(prompt_content.encode()),
                    response_size_bytes=len(response_content.encode()),
                )

                # Add workflow tracking if specified
                if workflow_id:
                    l9_genai.add_workflow_attributes(span, workflow_id, "openai_chat")

                span.set_status(trace.Status(trace.StatusCode.OK))

                return response, cost

            else:
                # Mock response for demo when OpenAI SDK is not available
                print("📝 Simulating OpenAI response (SDK not installed)")

                mock_usage = {
                    "input_tokens": len(prompt_content.split()) * 2,
                    "output_tokens": 150,
                    "total_tokens": len(prompt_content.split()) * 2 + 150,
                }

                cost = l9_genai.add_llm_cost_attributes(span, model, mock_usage, workflow_id)

                mock_response = {
                    "content": "This is a simulated OpenAI response.",
                    "usage": mock_usage,
                    "id": "chatcmpl-mock-123",
                    "model": model,
                }

                return mock_response, cost

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def openai_workflow_example():
    """Complete workflow example with OpenAI"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    # Initialize OpenAI client (you'll need your API key)
    if OPENAI_AVAILABLE:
        # Replace with your actual API key or use environment variable
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))
    else:
        client = None

    workflow_id = "customer_support_workflow"

    print("🔄 Starting customer support workflow with OpenAI...")

    # Step 1: Initial customer query processing
    with tracer.start_span("support_query_analysis") as span:
        l9_genai.set_span_kind(span, SpanKinds.LLM)
        l9_genai.add_workflow_attributes(
            span, workflow_id, workflow_type="customer_support", user_id="customer_456"
        )

        messages = [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {
                "role": "user",
                "content": "I can't access the premium features I paid for. Can you help?",
            },
        ]

        try:
            response1, cost1 = chat_with_observability(client, "gpt-4o", messages, workflow_id)
            print(f"✅ Query analysis completed - Cost: ${cost1.total:.6f}")

        except Exception as e:
            print(f"❌ Error in query analysis: {e}")

    # Step 2: Generate detailed response with cheaper model
    with tracer.start_span("response_generation") as span:
        l9_genai.set_span_kind(span, SpanKinds.LLM)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide detailed troubleshooting steps.",
            },
            {
                "role": "user",
                "content": "Create a step-by-step troubleshooting guide for premium feature access issues.",
            },
        ]

        try:
            response2, cost2 = chat_with_observability(
                client, "gpt-3.5-turbo", messages, workflow_id  # Using cheaper model
            )
            print(f"✅ Response generation completed - Cost: ${cost2.total:.6f}")

        except Exception as e:
            print(f"❌ Error in response generation: {e}")

    # Step 3: Follow-up suggestions
    with tracer.start_span("followup_suggestions") as span:
        l9_genai.set_span_kind(span, SpanKinds.LLM)

        messages = [
            {
                "role": "user",
                "content": "What can we do proactively to prevent premium access issues?",
            }
        ]

        try:
            response3, cost3 = chat_with_observability(
                client, "gpt-3.5-turbo", messages, workflow_id
            )
            print(f"✅ Follow-up suggestions completed - Cost: ${cost3.total:.6f}")

        except Exception as e:
            print(f"❌ Error in follow-up suggestions: {e}")

    # Check total workflow cost
    workflow = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
    if workflow:
        print(f"\n💰 Workflow Summary:")
        print(f"   Total cost: ${workflow.total_cost:.6f} USD")
        print(f"   LLM calls: {workflow.llm_calls}")
        print(f"   Tool calls: {workflow.tool_calls}")


def streaming_example():
    """Example with OpenAI streaming responses"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    print("\n🔄 Streaming response example...")

    if OPENAI_AVAILABLE:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

        with tracer.start_span("gen_ai.chat.completions.stream") as span:
            l9_genai.set_span_kind(span, SpanKinds.LLM)

            messages = [{"role": "user", "content": "Count from 1 to 5"}]

            try:
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=messages, stream=True
                )

                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)

                print("\n✅ Streaming completed")

                # Note: With streaming, usage data isn't available until the end
                # You may need to estimate or wait for the final chunk
                estimated_usage = {
                    "input_tokens": len(" ".join([m["content"] for m in messages]).split()) * 2,
                    "output_tokens": len(full_response.split()) * 2,
                }

                cost = l9_genai.add_llm_cost_attributes(span, "gpt-3.5-turbo", estimated_usage)
                print(f"   Estimated cost: ${cost.total:.6f}")

            except Exception as e:
                print(f"❌ Error in streaming: {e}")
    else:
        print("   (OpenAI SDK not available - skipping)")


if __name__ == "__main__":
    print("Last9 GenAI Attributes - OpenAI Integration Example")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("⚠️  Running in simulation mode (OpenAI SDK not installed)")
        print("   Install with: pip install openai")
        print("   Set OPENAI_API_KEY environment variable")

    try:
        openai_workflow_example()
        streaming_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n✅ OpenAI integration examples completed!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
