#!/usr/bin/env python3
"""
Basic usage example of Last9 GenAI attributes with OpenTelemetry

This example shows how to add Last9-specific attributes to your OpenTelemetry
spans when making LLM calls, providing cost tracking and enhanced observability.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import Last9GenAI, ModelPricing, create_llm_span


def setup_tracing():
    """Set up OpenTelemetry tracing with console export for demo"""
    trace.set_tracer_provider(TracerProvider())

    # Use console exporter for demo - replace with OTLP exporter for production
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return trace.get_tracer(__name__)


def simulate_llm_call(model: str, prompt: str) -> dict:
    """Simulate an LLM API call"""
    # This would be your actual LLM API call
    # For demo, we return mock usage data
    input_tokens = len(prompt.split()) * 2  # Rough estimation
    output_tokens = 150  # Mock response length

    return {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "response": {"id": "msg_123456", "model": model, "finish_reason": "stop"},
        "content": "This is a simulated AI response to your prompt.",
    }


def basic_llm_example():
    """Basic example of LLM call with Last9 cost tracking"""
    tracer = setup_tracing()

    # Initialize with pricing for cost tracking
    l9_genai = Last9GenAI(
        custom_pricing={
            "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }
    )

    print("🔄 Making LLM call with Last9 cost tracking...")

    # Method 1: Manual span creation and attribute addition
    with tracer.start_span("gen_ai.chat.completions") as span:
        model = "claude-3-5-sonnet"
        prompt = "Explain the benefits of observability in AI applications"

        # Simulate LLM call
        result = simulate_llm_call(model, prompt)

        # Add standard OpenTelemetry GenAI attributes
        l9_genai.add_standard_llm_attributes(
            span,
            model,
            conversation_id="demo_session_123",
            request_params={"max_tokens": 1000, "temperature": 0.7, "top_p": 0.9},
            response_data=result["response"],
            usage=result["usage"],
        )

        # Add Last9-specific attributes
        l9_genai.set_span_kind(span, "llm")
        cost = l9_genai.add_llm_cost_attributes(
            span, model, result["usage"], workflow_id="demo_workflow"
        )

        print(f"✅ LLM call completed!")
        print(f"   Model: {model}")
        print(f"   Input tokens: {result['usage']['input_tokens']}")
        print(f"   Output tokens: {result['usage']['output_tokens']}")
        print(f"   Total cost: ${cost.total:.6f} USD")
        print(f"   Input cost: ${cost.input:.6f} USD")
        print(f"   Output cost: ${cost.output:.6f} USD")


def convenience_function_example():
    """Example using convenience function for span creation"""
    tracer = setup_tracing()
    l9_genai = Last9GenAI()

    print("\n🔄 Using convenience function for span creation...")

    # Method 2: Using convenience function
    with create_llm_span(
        tracer,
        span_name="customer_support_chat",
        model="gpt-4o",
        workflow_id="support_workflow_456",
        conversation_id="customer_session_789",
        l9_genai=l9_genai,
    ) as span:
        # Simulate customer support LLM call
        prompt = "How can I cancel my subscription?"
        result = simulate_llm_call("gpt-4o", prompt)

        # Add usage and cost data
        cost = l9_genai.add_llm_cost_attributes(
            span, "gpt-4o", result["usage"], "support_workflow_456"
        )

        print(f"✅ Customer support call completed!")
        print(f"   Model: gpt-4o")
        print(f"   Total cost: ${cost.total:.6f} USD")


def workflow_example():
    """Example of multi-step workflow with cost aggregation"""
    tracer = setup_tracing()
    l9_genai = Last9GenAI()

    print("\n🔄 Multi-step workflow example...")

    workflow_id = "multi_step_analysis"

    # Step 1: Initial analysis
    with create_llm_span(
        tracer, "analysis_step", "claude-3-haiku", workflow_id=workflow_id
    ) as span:
        l9_genai.add_workflow_attributes(
            span, workflow_id, workflow_type="document_analysis", user_id="user_123"
        )

        result = simulate_llm_call("claude-3-haiku", "Analyze this document for key themes")
        cost1 = l9_genai.add_llm_cost_attributes(
            span, "claude-3-haiku", result["usage"], workflow_id
        )
        print(f"   Step 1 cost: ${cost1.total:.6f}")

    # Step 2: Detailed processing
    with create_llm_span(
        tracer, "processing_step", "claude-3-5-sonnet", workflow_id=workflow_id
    ) as span:
        result = simulate_llm_call(
            "claude-3-5-sonnet", "Provide detailed analysis of the themes found"
        )
        cost2 = l9_genai.add_llm_cost_attributes(
            span, "claude-3-5-sonnet", result["usage"], workflow_id
        )
        print(f"   Step 2 cost: ${cost2.total:.6f}")

    # Step 3: Summary generation
    with create_llm_span(tracer, "summary_step", "claude-3-haiku", workflow_id=workflow_id) as span:
        result = simulate_llm_call("claude-3-haiku", "Create an executive summary")
        cost3 = l9_genai.add_llm_cost_attributes(
            span, "claude-3-haiku", result["usage"], workflow_id
        )
        print(f"   Step 3 cost: ${cost3.total:.6f}")

    # Check total workflow cost
    workflow_cost = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
    print(f"✅ Workflow completed!")
    print(f"   Total workflow cost: ${workflow_cost.total_cost:.6f} USD")
    print(f"   Total LLM calls: {workflow_cost.llm_calls}")


if __name__ == "__main__":
    print("Last9 GenAI Attributes - Basic Usage Example")
    print("=" * 50)

    try:
        # Run examples
        basic_llm_example()
        convenience_function_example()
        workflow_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n✅ All examples completed!")
        print("\nIn production, replace ConsoleSpanExporter with OTLPSpanExporter")
        print("pointing to your Last9/OpenTelemetry endpoint.")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
