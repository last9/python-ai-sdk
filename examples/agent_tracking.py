#!/usr/bin/env python3
"""
Agent identity tracking example

Demonstrates tracking agent identity using OTel GenAI semantic conventions
(gen_ai.agent.id, gen_ai.agent.name, gen_ai.agent.version).

This is useful for multi-agent systems where you need to attribute spans
to specific agents and correlate their interactions.
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
    agent_context,
    conversation_context,
    workflow_context,
)


def setup_tracing():
    """Set up OpenTelemetry tracing with Last9 auto-enrichment"""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    custom_pricing = {
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-4o-mini": ModelPricing(input=0.15, output=0.60),
    }
    l9_processor = Last9SpanProcessor(custom_pricing=custom_pricing)
    provider.add_span_processor(l9_processor)

    return trace.get_tracer(__name__)


def simulate_llm_call(tracer, model: str, prompt: str) -> dict:
    """Simulate an LLM API call"""
    with tracer.start_span("gen_ai.chat.completions") as span:
        time.sleep(0.05)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.usage.input_tokens", len(prompt.split()) * 2)
        span.set_attribute("gen_ai.usage.output_tokens", 50)
        return {"response": f"Response to: {prompt[:40]}..."}


def single_agent_example():
    """Basic agent context example"""
    tracer = setup_tracing()

    print("\n--- Example 1: Single agent tracking ---\n")

    with agent_context(agent_id="support_bot_v2", agent_name="Support Bot", agent_version="2.0"):
        result = simulate_llm_call(tracer, "gpt-4o", "Help me with my order")
        print(f"   Response: {result['response']}")

    print("\n   Span attributes:")
    print("      gen_ai.agent.id = 'support_bot_v2'")
    print("      gen_ai.agent.name = 'Support Bot'")
    print("      gen_ai.agent.version = '2.0'")


def multi_agent_routing_example():
    """Multi-agent system with routing"""
    tracer = setup_tracing()

    print("\n--- Example 2: Multi-agent routing ---\n")

    with conversation_context(conversation_id="session_abc", user_id="user_42"):
        # Router agent classifies intent
        with agent_context(agent_id="router_v1", agent_name="Router Agent"):
            intent = simulate_llm_call(tracer, "gpt-4o-mini", "Classify: refund my order")
            print(f"   Router: {intent['response']}")

        # Specialist agent handles the request
        with agent_context(agent_id="refund_agent_v3", agent_name="Refund Agent", agent_version="3.1"):
            response = simulate_llm_call(tracer, "gpt-4o", "Process refund for order #12345")
            print(f"   Refund Agent: {response['response']}")

    print("\n   Router spans: gen_ai.agent.id='router_v1', conversation_id='session_abc'")
    print("   Refund spans: gen_ai.agent.id='refund_agent_v3', conversation_id='session_abc'")


def agent_with_workflow_example():
    """Agent context nested with workflow context"""
    tracer = setup_tracing()

    print("\n--- Example 3: Agent + workflow nesting ---\n")

    with conversation_context(conversation_id="session_xyz"):
        with agent_context(agent_id="rag_agent", agent_name="RAG Agent", agent_version="1.0"):
            with workflow_context(workflow_id="retrieval_pipeline", workflow_type="rag"):
                simulate_llm_call(tracer, "gpt-4o-mini", "Expand query: best restaurants")
                simulate_llm_call(tracer, "gpt-4o", "Synthesize answer from documents")
                print("   RAG pipeline completed")

    print("\n   All spans have:")
    print("      gen_ai.conversation.id = 'session_xyz'")
    print("      gen_ai.agent.id = 'rag_agent'")
    print("      workflow.id = 'retrieval_pipeline'")


if __name__ == "__main__":
    print("Last9 GenAI - Agent Identity Tracking (OTel Semantic Conventions)")
    print("=" * 70)

    try:
        single_agent_example()
        multi_agent_routing_example()
        agent_with_workflow_example()

        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n" + "=" * 70)
        print("All agent tracking examples completed!")
        print("\nAttributes follow OTel GenAI semantic conventions:")
        print("   gen_ai.agent.id          - Unique agent identifier")
        print("   gen_ai.agent.name        - Human-readable name")
        print("   gen_ai.agent.version     - Agent version")
        print("\nSee: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
