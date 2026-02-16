"""
Span processor for automatic attribute enrichment.

This processor automatically adds Last9 attributes (cost, conversation, workflow)
to OpenTelemetry spans based on context variables and span data.
"""

from typing import Optional, Dict
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.context import Context

from .context import get_current_context
from .core import GenAIAttributes, Last9Attributes, calculate_llm_cost, ModelPricing


class Last9SpanProcessor(SpanProcessor):
    """
    Span processor that automatically enriches spans with Last9 context attributes.

    This processor automatically adds:
    1. Conversation ID (from conversation_context)
    2. Workflow ID (from workflow_context)
    3. User ID (from context)
    4. Custom attributes (from propagate_attributes)

    Note: Cost attributes cannot be added automatically by a SpanProcessor because
    spans are immutable by the time usage data is available. For automatic cost
    tracking, use the Last9GenAI class methods or add them in your code.

    The processor DOES track workflow costs internally for aggregation.

    Example:
        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from last9_genai import Last9SpanProcessor, conversation_context

        provider = TracerProvider()
        processor = Last9SpanProcessor(
            custom_pricing={"gpt-4o": ModelPricing(input=2.50, output=10.0)}
        )
        provider.add_span_processor(processor)

        # Now all spans automatically get context attributes
        with conversation_context(conversation_id="session_123"):
            # Spans created here automatically have conversation_id
            ...
        ```
    """

    def __init__(
        self,
        custom_pricing: Optional[Dict[str, ModelPricing]] = None,
        enable_cost_tracking: bool = True,
        workflow_tracker=None,
    ):
        """
        Initialize the span processor.

        Args:
            custom_pricing: Dictionary of model pricing
            enable_cost_tracking: Whether to calculate and add cost attributes
            workflow_tracker: Optional workflow cost tracker instance
        """
        self.custom_pricing = custom_pricing
        self.enable_cost_tracking = enable_cost_tracking
        self.workflow_tracker = workflow_tracker

    def on_start(self, span: "Span", parent_context: Optional[Context] = None) -> None:
        """
        Called when a span starts. Add context attributes here while span is mutable.

        Args:
            span: The span that is starting (mutable)
            parent_context: The parent context
        """
        # Add context attributes from contextvars
        self._add_context_attributes_on_start(span)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends. Calculate costs here (read-only).

        Note: span is immutable at this point (ReadableSpan), so we can only
        read attributes, not modify them. Context attributes are already added
        in on_start().

        Args:
            span: The span that just ended (read-only)
        """
        if not span.attributes:
            return

        # Check if this is an LLM span
        is_llm_span = self._is_llm_span(span.attributes)

        if not is_llm_span:
            return

        # Track workflow cost if enabled
        if self.enable_cost_tracking and self.custom_pricing and self.workflow_tracker:
            self._track_workflow_cost(span)

    def _is_llm_span(self, attributes: Dict) -> bool:
        """
        Check if a span is an LLM span based on attributes.

        Args:
            attributes: Span attributes

        Returns:
            True if this is an LLM span
        """
        # LLM spans typically have gen_ai.request.model or gen_ai.provider.name
        return (
            GenAIAttributes.REQUEST_MODEL in attributes
            or GenAIAttributes.PROVIDER_NAME in attributes
            or "gen_ai.request.model" in attributes
            or "gen_ai.provider.name" in attributes
        )

    def _add_context_attributes_on_start(self, span: "Span") -> None:
        """
        Add context attributes from contextvars to the span at start time.

        Args:
            span: The span to enrich (mutable)
        """
        context = get_current_context()

        # Add conversation attributes
        if "conversation_id" in context:
            span.set_attribute(GenAIAttributes.CONVERSATION_ID, context["conversation_id"])

        if "turn_number" in context:
            span.set_attribute("gen_ai.conversation.turn_number", context["turn_number"])

        if "user_id" in context:
            span.set_attribute("user.id", context["user_id"])

        # Add workflow attributes
        if "workflow_id" in context:
            span.set_attribute("workflow.id", context["workflow_id"])

        if "workflow_type" in context:
            span.set_attribute("workflow.type", context["workflow_type"])

        # Add any custom attributes
        for key, value in context.items():
            if key not in [
                "conversation_id",
                "turn_number",
                "user_id",
                "workflow_id",
                "workflow_type",
            ]:
                span.set_attribute(f"custom.{key}", str(value))

    def _track_workflow_cost(self, span: ReadableSpan) -> None:
        """
        Track workflow cost based on span usage (read-only).

        Note: We can't add cost attributes to the span here because it's immutable.
        Users should add cost attributes manually or use a span exporter.

        Args:
            span: The span that just ended (read-only)
        """
        if not span.attributes:
            return

        # Extract model and usage from span attributes
        model = span.attributes.get(GenAIAttributes.REQUEST_MODEL) or span.attributes.get(
            "gen_ai.request.model"
        )
        if not model:
            return

        # Extract token usage
        usage = self._extract_usage(span.attributes)
        if not usage:
            return

        # Calculate cost
        cost = calculate_llm_cost(model, usage, self.custom_pricing)
        if not cost:
            return

        # Track workflow cost if workflow_id in context
        context = get_current_context()
        workflow_id = context.get("workflow_id")

        if workflow_id and self.workflow_tracker:
            self.workflow_tracker.add_llm_call(workflow_id, cost.total)

    def _extract_usage(self, attributes: Dict) -> Optional[Dict[str, int]]:
        """
        Extract token usage from span attributes.

        Args:
            attributes: Span attributes

        Returns:
            Dictionary with input_tokens, output_tokens, or None
        """
        # Try standard attribute names
        input_tokens = (
            attributes.get(GenAIAttributes.USAGE_INPUT_TOKENS)
            or attributes.get("gen_ai.usage.input_tokens")
            or attributes.get(GenAIAttributes.USAGE_PROMPT_TOKENS)
            or attributes.get("gen_ai.usage.prompt_tokens")
        )

        output_tokens = (
            attributes.get(GenAIAttributes.USAGE_OUTPUT_TOKENS)
            or attributes.get("gen_ai.usage.output_tokens")
            or attributes.get(GenAIAttributes.USAGE_COMPLETION_TOKENS)
            or attributes.get("gen_ai.usage.completion_tokens")
        )

        if input_tokens is None or output_tokens is None:
            return None

        return {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}

    def shutdown(self) -> None:
        """Called when the SDK shuts down."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any buffered data.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful
        """
        return True
