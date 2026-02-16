"""
Decorators for automatic LLM tracking.

This module provides the @observe() decorator that automatically tracks:
- Input/output (as span events)
- Latency (as span duration)
- Cost (calculated from usage)
- Metadata (from context)
"""

import time
import functools
import inspect
from typing import Optional, Dict, Any, Callable
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .context import get_current_context
from .core import (
    GenAIAttributes,
    Last9Attributes,
    calculate_llm_cost,
    ModelPricing,
    SpanKinds,
)


def observe(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    as_type: str = "llm",
    pricing: Optional[Dict[str, ModelPricing]] = None,
    capture_args: bool = True,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
):
    """
    Decorator for automatic LLM call tracking.

    Automatically tracks:
    - Input/output as span events (gen_ai.content.prompt/completion)
    - Latency as span duration
    - Cost per generation (if pricing provided and usage available)
    - Metadata from context (conversation_id, workflow_id, etc.)

    Args:
        name: Optional span name (defaults to function name)
        capture_input: Whether to capture function input as span event
        capture_output: Whether to capture function output as span event
        as_type: Span type - "llm", "tool", "chain" (sets gen_ai.l9.span.kind)
        pricing: Optional pricing dictionary for cost calculation
        capture_args: Whether to capture function arguments in span

    Example:
        ```python
        from last9_genai import observe, ModelPricing

        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

        @observe(pricing=pricing)
        def call_openai(prompt: str):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response

        # Automatically tracks:
        # - Input (prompt)
        # - Output (response)
        # - Latency
        # - Cost
        # - Context (conversation_id, workflow_id, etc.)
        ```

        With context:
        ```python
        with conversation_context(conversation_id="session_123"):
            response = call_openai("Hello!")
            # Span automatically has conversation_id
        ```
    """

    def decorator(func: Callable) -> Callable:
        # Determine if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _observe_async(
                    func,
                    args,
                    kwargs,
                    name or func.__name__,
                    capture_input,
                    capture_output,
                    as_type,
                    pricing,
                    capture_args,
                    tags,
                    metadata,
                )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return _observe_sync(
                    func,
                    args,
                    kwargs,
                    name or func.__name__,
                    capture_input,
                    capture_output,
                    as_type,
                    pricing,
                    capture_args,
                    tags,
                    metadata,
                )

            return sync_wrapper

    return decorator


def _observe_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    span_name: str,
    capture_input: bool,
    capture_output: bool,
    as_type: str,
    pricing: Optional[Dict[str, ModelPricing]],
    capture_args: bool,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
):
    """Synchronous observation wrapper."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(span_name) as span:
        try:
            # Add span kind
            span_kind = _get_span_kind(as_type)
            span.set_attribute(Last9Attributes.L9_SPAN_KIND, span_kind)

            # Add context attributes (conversation_id, workflow_id, etc.)
            _add_context_attributes(span)

            # Add tags as array attribute (OTel standard)
            if tags:
                span.set_attribute("tags", tags)

            # Add metadata as individual attributes
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))

                # Special handling: emit metadata.category as user.category for Last9 dashboard
                if "category" in metadata:
                    span.set_attribute("user.category", str(metadata["category"]))

            # Capture function arguments
            if capture_args:
                _add_function_arguments(span, func, args, kwargs)

            # Capture input as span event
            if capture_input:
                input_str = _format_input(args, kwargs)
                span.add_event(
                    "gen_ai.content.prompt",
                    {"gen_ai.prompt": input_str[:10000]},  # Limit to 10KB
                )

            # Execute function and measure latency
            start_time = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Add latency
            span.set_attribute("latency_ms", latency_ms)

            # Capture output as span event
            if capture_output:
                output_str = _format_output(result)
                span.add_event(
                    "gen_ai.content.completion",
                    {"gen_ai.completion": output_str[:10000]},  # Limit to 10KB
                )

            # Extract and add LLM attributes (model, usage, cost)
            _add_llm_attributes(span, result, pricing, kwargs)

            # Mark span as successful
            span.set_status(Status(StatusCode.OK))

            return result

        except Exception as e:
            # Record exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


async def _observe_async(
    func: Callable,
    args: tuple,
    kwargs: dict,
    span_name: str,
    capture_input: bool,
    capture_output: bool,
    as_type: str,
    pricing: Optional[Dict[str, ModelPricing]],
    capture_args: bool,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
):
    """Asynchronous observation wrapper."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(span_name) as span:
        try:
            # Add span kind
            span_kind = _get_span_kind(as_type)
            span.set_attribute(Last9Attributes.L9_SPAN_KIND, span_kind)

            # Add context attributes
            _add_context_attributes(span)

            # Add tags as array attribute (OTel standard)
            if tags:
                span.set_attribute("tags", tags)

            # Add metadata as individual attributes
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))

                # Special handling: emit metadata.category as user.category for Last9 dashboard
                if "category" in metadata:
                    span.set_attribute("user.category", str(metadata["category"]))

            # Capture function arguments
            if capture_args:
                _add_function_arguments(span, func, args, kwargs)

            # Capture input
            if capture_input:
                input_str = _format_input(args, kwargs)
                span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": input_str[:10000]})

            # Execute function and measure latency
            start_time = time.time()
            result = await func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Add latency
            span.set_attribute("latency_ms", latency_ms)

            # Capture output
            if capture_output:
                output_str = _format_output(result)
                span.add_event(
                    "gen_ai.content.completion", {"gen_ai.completion": output_str[:10000]}
                )

            # Extract and add LLM attributes
            _add_llm_attributes(span, result, pricing)

            # Mark span as successful
            span.set_status(Status(StatusCode.OK))

            return result

        except Exception as e:
            # Record exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def _get_span_kind(as_type: str) -> str:
    """Map as_type to span kind."""
    type_map = {
        "llm": SpanKinds.LLM,
        "tool": SpanKinds.TOOL,
        "chain": SpanKinds.CHAIN,
        "agent": SpanKinds.AGENT,
        "prompt": SpanKinds.PROMPT,
    }
    return type_map.get(as_type.lower(), SpanKinds.LLM)


def _add_context_attributes(span):
    """Add attributes from context (conversation_id, workflow_id, etc.)."""
    context = get_current_context()

    if "conversation_id" in context:
        span.set_attribute(GenAIAttributes.CONVERSATION_ID, context["conversation_id"])

    if "turn_number" in context:
        span.set_attribute("gen_ai.conversation.turn_number", context["turn_number"])

    if "user_id" in context:
        span.set_attribute("user.id", context["user_id"])

    if "workflow_id" in context:
        span.set_attribute("workflow.id", context["workflow_id"])

    if "workflow_type" in context:
        span.set_attribute("workflow.type", context["workflow_type"])

    # Add custom attributes
    for key, value in context.items():
        if key not in ["conversation_id", "turn_number", "user_id", "workflow_id", "workflow_type"]:
            span.set_attribute(f"custom.{key}", str(value))


def _add_function_arguments(span, func: Callable, args: tuple, kwargs: dict):
    """Add function arguments as span attributes."""
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Add each argument as attribute
        for param_name, param_value in bound_args.arguments.items():
            # Preserve type for primitives, convert others to string
            if isinstance(param_value, (int, float, bool, str)):
                span.set_attribute(f"function.arg.{param_name}", param_value)
            else:
                # Convert complex objects to string (with size limit)
                value_str = str(param_value)
                if len(value_str) < 1000:
                    span.set_attribute(f"function.arg.{param_name}", value_str)
    except Exception:
        # If argument binding fails, just skip
        pass


def _format_input(args: tuple, kwargs: dict) -> str:
    """Format function input for logging."""
    # If single string argument, return it directly (most common case)
    if len(args) == 1 and not kwargs and isinstance(args[0], str):
        return args[0]

    # If messages array (common for chat APIs)
    if len(args) == 1 and not kwargs and isinstance(args[0], list):
        try:
            # Format chat messages
            messages = args[0]
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
                formatted = []
                for msg in messages:
                    formatted.append(f"{msg['role']}: {msg['content']}")
                return "\n".join(formatted)
        except Exception:
            pass

    # Check if 'prompt' or 'message' in kwargs
    if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
        return kwargs["prompt"]
    if "message" in kwargs and isinstance(kwargs["message"], str):
        return kwargs["message"]
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        try:
            messages = kwargs["messages"]
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
                formatted = []
                for msg in messages:
                    formatted.append(f"{msg['role']}: {msg['content']}")
                return "\n".join(formatted)
        except Exception:
            pass

    # Fallback: show args and kwargs but cleaner
    parts = []
    if args:
        # Show args without 'args=' prefix if single arg
        if len(args) == 1:
            parts.append(str(args[0]))
        else:
            parts.append(f"args={args}")

    if kwargs:
        # Show kwargs items cleanly
        for k, v in kwargs.items():
            parts.append(f"{k}={v}")

    return ", ".join(parts) if parts else "(no input)"


def _format_output(result: Any) -> str:
    """Format function output for logging."""
    if result is None:
        return "(no output)"

    # Try to get text content from common LLM response types
    try:
        # OpenAI response
        if hasattr(result, "choices") and len(result.choices) > 0:
            if hasattr(result.choices[0], "message"):
                return result.choices[0].message.content or ""

        # Anthropic response
        if hasattr(result, "content") and isinstance(result.content, list):
            texts = [block.text for block in result.content if hasattr(block, "text")]
            return " ".join(texts)

        # Plain string
        if isinstance(result, str):
            return result

        # Fallback to string representation
        return str(result)
    except Exception:
        return "(output capture failed)"


def _add_llm_attributes(
    span, result: Any, pricing: Optional[Dict[str, ModelPricing]], kwargs: dict = None
):
    """Extract and add LLM-specific attributes from result."""
    try:
        # Extract model
        model = _extract_model(result)
        if model:
            span.set_attribute(GenAIAttributes.RESPONSE_MODEL, model)

        # Extract system/provider
        system = _extract_system(result, model)
        if system:
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, system)

        # Extract response ID
        response_id = _extract_response_id(result)
        if response_id:
            span.set_attribute(GenAIAttributes.RESPONSE_ID, response_id)

        # Extract model parameters from kwargs (if provided in request)
        if kwargs:
            _extract_request_parameters(span, kwargs)

        # Extract usage
        usage = _extract_usage(result)
        if usage:
            span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, usage["input_tokens"])
            span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, usage["output_tokens"])

            # Calculate cost if pricing provided
            if pricing and model:
                cost = calculate_llm_cost(model, usage, pricing)
                if cost:
                    span.set_attribute(GenAIAttributes.USAGE_COST_USD, cost.total)
                    span.set_attribute(GenAIAttributes.USAGE_COST_INPUT_USD, cost.input)
                    span.set_attribute(GenAIAttributes.USAGE_COST_OUTPUT_USD, cost.output)

        # Extract finish reason
        finish_reason = _extract_finish_reason(result)
        if finish_reason:
            span.set_attribute(GenAIAttributes.RESPONSE_FINISH_REASONS, finish_reason)

        # Extract tool calls (if present)
        _extract_tool_calls(span, result)

    except Exception:
        # If attribute extraction fails, just skip
        pass


def _extract_model(result: Any) -> Optional[str]:
    """Extract model name from LLM response."""
    # OpenAI
    if hasattr(result, "model"):
        return result.model

    # Anthropic
    if hasattr(result, "model"):
        return result.model

    return None


def _extract_system(result: Any, model: Optional[str]) -> Optional[str]:
    """Extract system/provider from response or model name."""
    # Try to infer from model name
    if model:
        if "gpt" in model.lower() or "o1" in model.lower():
            return "openai"
        if "claude" in model.lower():
            return "anthropic"
        if "gemini" in model.lower():
            return "google"
        if "command" in model.lower():
            return "cohere"

    # Try to infer from response type
    if hasattr(result, "__class__"):
        class_name = result.__class__.__name__.lower()
        if "openai" in class_name:
            return "openai"
        if "anthropic" in class_name:
            return "anthropic"

    return None


def _extract_usage(result: Any) -> Optional[Dict[str, int]]:
    """Extract token usage from LLM response."""
    # OpenAI format
    if hasattr(result, "usage"):
        usage = result.usage
        if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            return {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            return {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

    return None


def _extract_finish_reason(result: Any) -> Optional[str]:
    """Extract finish reason from LLM response."""
    # OpenAI
    if hasattr(result, "choices") and len(result.choices) > 0:
        if hasattr(result.choices[0], "finish_reason"):
            return result.choices[0].finish_reason

    # Anthropic
    if hasattr(result, "stop_reason"):
        return result.stop_reason

    return None


def _extract_response_id(result: Any) -> Optional[str]:
    """Extract response ID from LLM response."""
    # OpenAI - response.id
    if hasattr(result, "id"):
        return result.id

    # Anthropic - response.id
    if hasattr(result, "id"):
        return result.id

    return None


def _extract_request_parameters(span, kwargs: dict) -> None:
    """Extract request parameters from function kwargs."""
    # Temperature
    if "temperature" in kwargs:
        span.set_attribute(GenAIAttributes.REQUEST_TEMPERATURE, float(kwargs["temperature"]))

    # Max tokens
    if "max_tokens" in kwargs:
        span.set_attribute(GenAIAttributes.REQUEST_MAX_TOKENS, int(kwargs["max_tokens"]))

    # Top P
    if "top_p" in kwargs:
        span.set_attribute(GenAIAttributes.REQUEST_TOP_P, float(kwargs["top_p"]))

    # Frequency penalty
    if "frequency_penalty" in kwargs:
        span.set_attribute(
            GenAIAttributes.REQUEST_FREQUENCY_PENALTY, float(kwargs["frequency_penalty"])
        )

    # Presence penalty
    if "presence_penalty" in kwargs:
        span.set_attribute(
            GenAIAttributes.REQUEST_PRESENCE_PENALTY, float(kwargs["presence_penalty"])
        )


def _extract_tool_calls(span, result: Any) -> None:
    """Extract and add tool call information from LLM response."""
    try:
        # OpenAI format - response.choices[0].message.tool_calls
        if hasattr(result, "choices") and len(result.choices) > 0:
            message = result.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Add tool call count
                span.set_attribute("gen_ai.tool_calls.count", len(message.tool_calls))

                # Add each tool call as an event with structure
                for i, tool_call in enumerate(message.tool_calls):
                    tool_call_data = {
                        "tool_call.index": i,
                        "tool_call.id": tool_call.id,
                        "tool_call.type": tool_call.type,
                        "tool_call.function.name": tool_call.function.name,
                        "tool_call.function.arguments": tool_call.function.arguments,
                    }
                    span.add_event(f"gen_ai.tool.call", tool_call_data)

                # Also add as attributes for easy filtering
                # First tool call name (most common use case)
                span.set_attribute(GenAIAttributes.TOOL_NAME, message.tool_calls[0].function.name)

                return

        # Anthropic format - response.content with tool_use blocks
        if hasattr(result, "content") and isinstance(result.content, list):
            tool_uses = [
                block
                for block in result.content
                if hasattr(block, "type") and block.type == "tool_use"
            ]

            if tool_uses:
                # Add tool call count
                span.set_attribute("gen_ai.tool_calls.count", len(tool_uses))

                # Add each tool call as an event
                for i, tool_use in enumerate(tool_uses):
                    tool_call_data = {
                        "tool_call.index": i,
                        "tool_call.id": tool_use.id,
                        "tool_call.name": tool_use.name,
                        "tool_call.input": str(tool_use.input),
                    }
                    span.add_event(f"gen_ai.tool.call", tool_call_data)

                # First tool name
                span.set_attribute(GenAIAttributes.TOOL_NAME, tool_uses[0].name)

    except Exception:
        # If extraction fails, just skip
        pass
