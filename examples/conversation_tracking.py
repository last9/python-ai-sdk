#!/usr/bin/env python3
"""
Conversation tracking example with content events

This example demonstrates the conversation tracking and content events functionality
that matches the Node.js agent, allowing users to track multi-turn conversations
with full input/output prompt visibility as span events.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import (
    ModelPricing,
    Last9GenAI,
    ConversationTracker,
    global_conversation_tracker,
    SpanKinds,
)


def setup_tracing():
    """Set up OpenTelemetry tracing with console export for demo"""
    trace.set_tracer_provider(TracerProvider())

    # Use console exporter for demo - replace with OTLP exporter for production
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return trace.get_tracer(__name__)


def simulate_llm_call(model: str, user_message: str) -> dict:
    """Simulate an LLM API call with conversation context"""
    time.sleep(0.1)  # Simulate API latency

    # Mock responses based on user input
    responses = {
        "Hello": "Hello! I'm Claude, an AI assistant. How can I help you today?",
        "What's the weather like?": "I don't have access to real-time weather data, but I can help you find weather information. What city are you interested in?",
        "San Francisco": "San Francisco typically has mild weather year-round with cool summers and mild winters. For current conditions, I'd recommend checking a weather service like weather.com or your local weather app.",
        "Thank you": "You're welcome! Is there anything else I can help you with?",
        "default": "I understand you're asking about that topic. Let me help you with some information and suggestions.",
    }

    assistant_message = responses.get(user_message, responses["default"])

    # Calculate token usage (rough estimation)
    input_tokens = len(user_message.split()) * 2
    output_tokens = len(assistant_message.split()) * 2

    return {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "response": {
            "id": f"msg_{hash(user_message + assistant_message) % 1000000}",
            "model": model,
            "finish_reason": "stop",
        },
        "content": assistant_message,
    }


def basic_conversation_example():
    """Basic conversation tracking example"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    print("🔄 Basic conversation tracking example...")

    conversation_id = "user_123_conversation"
    user_id = "user_123"
    model = "claude-3-5-sonnet"

    # Conversation turns
    turns = ["Hello", "What's the weather like?", "San Francisco", "Thank you"]

    for i, user_message in enumerate(turns, 1):
        print(f"\n   Turn {i}: {user_message}")

        # Create conversation-aware span
        with l9_genai.create_conversation_span(
            tracer, conversation_id, model, user_id=user_id, turn_number=i
        ) as span:

            # Simulate LLM call
            result = simulate_llm_call(model, user_message)
            assistant_message = result["content"]

            print(f"   Response: {assistant_message[:50]}...")

            # Add content events (input/output prompts as span events)
            l9_genai.add_content_events(
                span,
                prompt=user_message,
                completion=assistant_message,
                truncate_length=200,  # Truncate long content
            )

            # Add cost tracking
            cost = l9_genai.add_llm_cost_attributes(span, model, result["usage"])

            # Track the conversation turn
            global_conversation_tracker.add_turn(
                conversation_id, user_message, assistant_message, model, result["usage"], cost
            )

            print(f"   Cost: ${cost.total:.6f}")

    # Get conversation statistics
    stats = global_conversation_tracker.get_conversation_stats(conversation_id)
    print(f"\n✅ Conversation completed!")
    print(f"   Total turns: {stats['turn_count']}")
    print(f"   Total cost: ${stats['total_cost']:.6f}")
    print(f"   Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}")


def advanced_conversation_with_tool_calls():
    """Advanced example with tool calls and content events"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    print("\n🔄 Advanced conversation with tool calls...")

    conversation_id = "advanced_conversation_456"
    user_id = "user_456"
    model = "gpt-4o"

    # Turn 1: Initial question
    with l9_genai.create_conversation_span(
        tracer, conversation_id, model, user_id=user_id, turn_number=1
    ) as span:

        user_message = "I need to check my account balance and recent transactions"
        assistant_message = (
            "I'll help you check your account information. Let me fetch that data for you."
        )

        # Add content events
        l9_genai.add_content_events(span, user_message, assistant_message)

        # Simulate tool call within the conversation
        with tracer.start_span("gen_ai.tool.account_lookup") as tool_span:
            l9_genai.add_tool_attributes(
                tool_span,
                "account_lookup",
                tool_type="financial_api",
                description="Look up user account information",
            )

            # Add tool call events
            l9_genai.add_tool_call_events(
                tool_span,
                "account_lookup",
                tool_arguments={"user_id": user_id, "include_transactions": True},
                tool_result={"balance": 1250.45, "recent_transactions": 5},
            )

        # Add LLM cost
        usage = {"input_tokens": 50, "output_tokens": 30}
        cost = l9_genai.add_llm_cost_attributes(span, model, usage)

        print(f"   Turn 1 - User: {user_message}")
        print(f"   Turn 1 - Assistant: {assistant_message}")
        print(f"   Turn 1 - Cost: ${cost.total:.6f}")

    # Turn 2: Follow-up with results
    with l9_genai.create_conversation_span(
        tracer, conversation_id, model, user_id=user_id, turn_number=2
    ) as span:

        user_message = "What about my spending this month?"
        assistant_message = """Based on your account data, here's your spending summary:
        - Current balance: $1,250.45
        - Recent transactions: 5 in the last week
        - Monthly spending appears to be within your normal range

        Would you like me to break down the spending by category?"""

        # Add content events with longer content
        l9_genai.add_content_events(
            span,
            user_message,
            assistant_message,
            truncate_length=150,  # This will truncate the longer response
        )

        usage = {"input_tokens": 80, "output_tokens": 120}
        cost = l9_genai.add_llm_cost_attributes(span, model, usage)

        print(f"   Turn 2 - User: {user_message}")
        print(f"   Turn 2 - Assistant: {assistant_message[:50]}...")
        print(f"   Turn 2 - Cost: ${cost.total:.6f}")
        print(f"   Turn 2 - Content truncated due to length > 150 chars")


def multi_model_conversation():
    """Example showing conversation across multiple models"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    print("\n🔄 Multi-model conversation example...")

    conversation_id = "multi_model_conversation"
    user_id = "user_789"

    # Turn 1: Complex analysis with expensive model
    with l9_genai.create_conversation_span(
        tracer, conversation_id, "claude-3-5-sonnet", user_id=user_id, turn_number=1
    ) as span:

        user_message = "Analyze the pros and cons of renewable energy adoption"
        assistant_message = "Renewable energy adoption presents several key advantages and challenges. On the positive side, renewables offer environmental benefits, energy independence, and long-term cost savings..."

        l9_genai.add_content_events(span, user_message, assistant_message)

        usage = {"input_tokens": 200, "output_tokens": 400}
        cost1 = l9_genai.add_llm_cost_attributes(span, "claude-3-5-sonnet", usage)

        print(f"   Turn 1 (Claude): Analysis request - ${cost1.total:.6f}")

    # Turn 2: Simple follow-up with cheaper model
    with l9_genai.create_conversation_span(
        tracer, conversation_id, "claude-3-haiku", user_id=user_id, turn_number=2
    ) as span:

        user_message = "Can you summarize that in 3 bullet points?"
        assistant_message = """• Environmental: Reduced carbon emissions and cleaner air
• Economic: Long-term savings despite high initial investment
• Energy Security: Reduced dependence on fossil fuel imports"""

        l9_genai.add_content_events(span, user_message, assistant_message)

        usage = {"input_tokens": 50, "output_tokens": 80}
        cost2 = l9_genai.add_llm_cost_attributes(span, "claude-3-haiku", usage)

        print(f"   Turn 2 (Haiku): Summary request - ${cost2.total:.6f}")

    # Show conversation cost optimization
    print(f"   💰 Cost optimization: Used expensive model for analysis (${cost1.total:.6f})")
    print(f"                        Used cheaper model for summary (${cost2.total:.6f})")
    print(f"   Total conversation cost: ${cost1.total + cost2.total:.6f}")


def conversation_with_workflow_integration():
    """Show conversation tracking integrated with workflow management"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    print("\n🔄 Conversation + workflow integration...")

    conversation_id = "support_conversation_999"
    workflow_id = "customer_support_workflow"
    user_id = "customer_999"

    # Start workflow span
    with tracer.start_span("customer_support_workflow") as workflow_span:
        l9_genai.add_workflow_attributes(
            workflow_span, workflow_id, workflow_type="customer_support", user_id=user_id
        )

        # Conversation Turn 1: Problem identification
        with l9_genai.create_conversation_span(
            tracer, conversation_id, "gpt-4o", user_id=user_id, turn_number=1
        ) as span:

            user_message = "I can't log into my account. It says invalid password."
            assistant_message = "I understand you're having trouble logging in. Let me help you troubleshoot this issue step by step."

            l9_genai.add_content_events(span, user_message, assistant_message)

            # Add workflow tracking to this conversation turn
            l9_genai.add_workflow_attributes(span, workflow_id)

            usage = {"input_tokens": 40, "output_tokens": 60}
            cost = l9_genai.add_llm_cost_attributes(span, "gpt-4o", usage, workflow_id)

            print(f"   Problem identification - Cost: ${cost.total:.6f}")

        # Tool call: Account verification
        with tracer.start_span("gen_ai.tool.account_verification") as tool_span:
            l9_genai.add_tool_attributes(
                tool_span, "account_verification", tool_type="auth_system", workflow_id=workflow_id
            )
            print(f"   Account verification tool called")

        # Conversation Turn 2: Solution provision
        with l9_genai.create_conversation_span(
            tracer, conversation_id, "gpt-4o", user_id=user_id, turn_number=2
        ) as span:

            user_message = "I tried resetting but didn't get the email"
            assistant_message = "I see the issue. Let me send a password reset to your verified email address and also check your spam folder."

            l9_genai.add_content_events(span, user_message, assistant_message)
            l9_genai.add_workflow_attributes(span, workflow_id)

            usage = {"input_tokens": 60, "output_tokens": 80}
            cost = l9_genai.add_llm_cost_attributes(span, "gpt-4o", usage, workflow_id)

            print(f"   Solution provision - Cost: ${cost.total:.6f}")

    # Check workflow totals
    workflow_cost = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
    print(f"   Total workflow cost: ${workflow_cost.total_cost:.6f}")
    print(f"   LLM calls: {workflow_cost.llm_calls}, Tool calls: {workflow_cost.tool_calls}")


if __name__ == "__main__":
    print("Last9 GenAI Attributes - Conversation Tracking with Content Events")
    print("=" * 70)

    try:
        # Run all conversation examples
        basic_conversation_example()
        advanced_conversation_with_tool_calls()
        multi_model_conversation()
        conversation_with_workflow_integration()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n✅ All conversation tracking examples completed!")
        print("\n📊 Key Features Demonstrated:")
        print("   • Conversation ID tracking across multiple turns")
        print("   • Content events for input/output prompts")
        print("   • Multi-model conversations with cost optimization")
        print("   • Tool call integration within conversations")
        print("   • Workflow + conversation tracking")
        print("   • Automatic content truncation for large prompts")
        print("\n🎯 This matches Node.js agent functionality:")
        print("   • gen_ai.conversation.id attribute")
        print("   • gen_ai.content.prompt span events")
        print("   • gen_ai.content.completion span events")
        print("   • Conversation cost aggregation")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
