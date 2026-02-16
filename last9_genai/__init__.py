"""
Last9 GenAI Attributes for Python OpenTelemetry

This package provides Last9-specific gen_ai attributes that complement the standard
OpenTelemetry gen_ai semantic conventions. It adds cost tracking, workflow management,
and enhanced observability features for LLM applications.

Usage:
    # Option 1: Manual instrumentation
    from last9_genai import Last9GenAI, ModelPricing

    l9_genai = Last9GenAI(custom_pricing={
        "gpt-4o": ModelPricing(input=2.50, output=10.0)
    })

    with tracer.start_span("llm_call") as span:
        usage = {"input_tokens": 100, "output_tokens": 200}
        cost = l9_genai.add_llm_cost_attributes(span, "gpt-4o", usage)

    # Option 2: Automatic context-based tracking (recommended)
    from last9_genai import conversation_context, workflow_context

    with conversation_context(conversation_id="user_session_123", user_id="user_123"):
        # All LLM calls automatically tagged with conversation_id
        response = client.chat.completions.create(...)

    # Option 3: Auto-enrichment with span processor
    from last9_genai import Last9SpanProcessor
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    provider.add_span_processor(Last9SpanProcessor(custom_pricing={...}))

For more information, see: https://github.com/last9/python-ai-sdk
"""

__version__ = "1.0.0"
__author__ = "Last9 Inc."
__license__ = "MIT"

# Import main classes and functions for easy access
from last9_genai.core import (
    Last9GenAI,
    GenAIAttributes,
    Last9Attributes,
    SpanKinds,
    create_llm_span,
    ModelPricing,
    calculate_llm_cost,
    CostBreakdown,
    WorkflowCostTracker,
    ConversationTracker,
    global_workflow_tracker,
    global_conversation_tracker,
)

# Import context management (auto-tracking)
from last9_genai.context import (
    propagate_attributes,
    conversation_context,
    workflow_context,
    get_current_context,
    clear_context,
)

# Import span processor (auto-enrichment)
from last9_genai.processor import Last9SpanProcessor

# Import decorators (auto-tracking)
from last9_genai.decorators import observe

__all__ = [
    # Version
    "__version__",
    # Main class
    "Last9GenAI",
    # Attributes
    "GenAIAttributes",
    "Last9Attributes",
    "SpanKinds",
    # Helper functions
    "create_llm_span",
    # Costing
    "ModelPricing",
    "calculate_llm_cost",
    "CostBreakdown",
    # Tracking
    "WorkflowCostTracker",
    "ConversationTracker",
    "global_workflow_tracker",
    "global_conversation_tracker",
    # Context management
    "propagate_attributes",
    "conversation_context",
    "workflow_context",
    "get_current_context",
    "clear_context",
    # Span processor
    "Last9SpanProcessor",
    # Decorators (NEW)
    "observe",
]
