#!/usr/bin/env python3
"""
Integration example with Anthropic Claude SDK

This example shows how to integrate Last9 GenAI attributes with the official
Anthropic Python SDK for comprehensive observability and cost tracking.

Install dependencies:
    pip install anthropic opentelemetry-api opentelemetry-sdk
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import ModelPricing, Last9GenAI, create_llm_span, SpanKinds

# Comment out if you don't have anthropic installed
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic SDK not installed. Install with: pip install anthropic")


def setup_tracing():
    """Set up OpenTelemetry tracing"""
    trace.set_tracer_provider(TracerProvider())

    # Use console exporter for demo - replace with OTLP exporter for production
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return trace.get_tracer(__name__)


def claude_chat_with_observability(client, model: str, messages: list, workflow_id: str = None):
    """
    Claude chat call with full Last9 observability

    Args:
        client: Anthropic client
        model: Model name (e.g., 'claude-3-5-sonnet')
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
            # Make the actual Anthropic API call
            if ANTHROPIC_AVAILABLE:
                response = client.messages.create(
                    model=model, max_tokens=1000, temperature=0.7, messages=messages
                )

                # Calculate response time
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000

                # Extract usage information from Anthropic response
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

                # Extract response content
                response_content = ""
                for content_block in response.content:
                    if hasattr(content_block, "text"):
                        response_content += content_block.text

                # Add response attributes
                l9_genai.add_standard_llm_attributes(
                    span,
                    model,
                    request_params={"max_tokens": 1000, "temperature": 0.7},
                    response_data={
                        "id": response.id,
                        "model": response.model,
                        "finish_reason": response.stop_reason,
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
                    l9_genai.add_workflow_attributes(span, workflow_id, "claude_chat")

                span.set_status(trace.Status(trace.StatusCode.OK))

                return response, cost

            else:
                # Mock response for demo when Anthropic SDK is not available
                print("📝 Simulating Anthropic response (SDK not installed)")

                mock_usage = {
                    "input_tokens": len(prompt_content.split()) * 2,
                    "output_tokens": 150,
                    "total_tokens": len(prompt_content.split()) * 2 + 150,
                }

                cost = l9_genai.add_llm_cost_attributes(span, model, mock_usage, workflow_id)

                mock_response = {
                    "content": "This is a simulated Claude response.",
                    "usage": mock_usage,
                    "id": "msg_mock_123",
                    "model": model,
                }

                return mock_response, cost

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def anthropic_workflow_example():
    """Complete workflow example with Anthropic Claude"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    # Initialize Anthropic client (you'll need your API key)
    if ANTHROPIC_AVAILABLE:
        # Replace with your actual API key or use environment variable
        client = anthropic.Anthropic(
            api_key="your-api-key-here"
        )  # or os.getenv("ANTHROPIC_API_KEY")
    else:
        client = None

    workflow_id = "customer_support_workflow"

    print("🔄 Starting customer support workflow with Claude...")

    # Step 1: Initial customer query processing
    with tracer.start_span("support_query_analysis") as span:
        l9_genai.set_span_kind(span, SpanKinds.LLM)
        l9_genai.add_workflow_attributes(
            span, workflow_id, workflow_type="customer_support", user_id="customer_456"
        )

        messages = [
            {
                "role": "user",
                "content": "I'm having trouble with my account login. It says my password is incorrect but I'm sure it's right.",
            }
        ]

        try:
            response1, cost1 = claude_chat_with_observability(
                client, "claude-3-5-sonnet", messages, workflow_id
            )
            print(f"✅ Query analysis completed - Cost: ${cost1.total:.6f}")

        except Exception as e:
            print(f"❌ Error in query analysis: {e}")

    # Step 2: Generate detailed response
    with tracer.start_span("response_generation") as span:
        l9_genai.set_span_kind(span, SpanKinds.LLM)

        messages = [
            {
                "role": "user",
                "content": "Based on the login issue, provide a comprehensive troubleshooting guide.",
            }
        ]

        try:
            response2, cost2 = claude_chat_with_observability(
                client,
                "claude-3-haiku",
                messages,
                workflow_id,  # Using cheaper model for detailed response
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
                "content": "Suggest proactive steps to prevent similar login issues in the future.",
            }
        ]

        try:
            response3, cost3 = claude_chat_with_observability(
                client, "claude-3-haiku", messages, workflow_id
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


def prompt_versioning_example():
    """Example of prompt template versioning with Claude"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    print("\n🔄 Prompt versioning example...")

    # Define versioned prompt template
    prompt_template = """
You are a helpful customer service AI assistant for TechCorp.

Customer Information:
- Account Status: {account_status}
- Subscription: {subscription_type}
- Previous Interactions: {interaction_history}

Customer Query: {query}

Please provide a helpful, professional response addressing their concern.
Use empathy and offer concrete solutions when possible.
"""

    with tracer.start_span("versioned_prompt_processing") as span:
        l9_genai.set_span_kind(span, SpanKinds.PROMPT)

        # Add prompt versioning
        prompt_hash = l9_genai.add_prompt_versioning(
            span, prompt_template, template_id="customer_service_v3", version="3.1.2"
        )

        # Fill in the template (in real scenario, this would come from your system)
        filled_prompt = prompt_template.format(
            account_status="Premium",
            subscription_type="Pro Plan",
            interaction_history="None",
            query="Having trouble accessing premium features",
        )

        print(f"✅ Prompt versioning added")
        print(f"   Template ID: customer_service_v3")
        print(f"   Version: 3.1.2")
        print(f"   Hash: {prompt_hash}")

        # Now use this versioned prompt with Claude
        messages = [{"role": "user", "content": filled_prompt}]

        if ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key="your-api-key-here")
            try:
                response, cost = claude_chat_with_observability(
                    client, "claude-3-5-sonnet", messages, "prompt_versioning_workflow"
                )
                print(f"   Response cost: ${cost.total:.6f}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("   (Would use prompt with Claude API)")


if __name__ == "__main__":
    print("Last9 GenAI Attributes - Anthropic Integration Example")
    print("=" * 60)

    if not ANTHROPIC_AVAILABLE:
        print("⚠️  Running in simulation mode (Anthropic SDK not installed)")
        print("   Install with: pip install anthropic")
        print("   Set ANTHROPIC_API_KEY environment variable")

    try:
        anthropic_workflow_example()
        prompt_versioning_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n✅ Anthropic integration examples completed!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
