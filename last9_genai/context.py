"""
Context-based tracking for conversations and workflows.

This module provides automatic context propagation using Python's contextvars
for thread-safe attribute tracking across LLM calls.
"""

from contextvars import ContextVar
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Context variables for automatic propagation
_conversation_id: ContextVar[Optional[str]] = ContextVar("conversation_id", default=None)
_conversation_turn: ContextVar[Optional[int]] = ContextVar("conversation_turn", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_workflow_id: ContextVar[Optional[str]] = ContextVar("workflow_id", default=None)
_workflow_type: ContextVar[Optional[str]] = ContextVar("workflow_type", default=None)
_custom_attributes: ContextVar[Dict[str, Any]] = ContextVar("custom_attributes", default={})


@contextmanager
def propagate_attributes(**custom_attrs):
    """
    Context manager for propagating custom attributes to all child spans.

    This allows setting arbitrary attributes that will be automatically
    added to all spans created within the context.

    Args:
        **custom_attrs: Custom attributes to propagate (added with 'custom.' prefix)

    Example:
        ```python
        with propagate_attributes(environment="production", version="1.0.0"):
            # All LLM calls will have custom.environment and custom.version
            response = client.chat.completions.create(...)
        ```
    """
    # Save previous custom attributes
    prev_custom = _custom_attributes.get()

    try:
        # Merge with existing custom attributes
        merged_attrs = prev_custom.copy() if prev_custom else {}
        merged_attrs.update(custom_attrs)
        _custom_attributes.set(merged_attrs)

        yield

    finally:
        # Restore previous custom attributes
        _custom_attributes.set(prev_custom)


def get_current_context() -> Dict[str, Any]:
    """
    Get all current context values.

    Returns:
        Dictionary of current context values
    """
    context = {}

    conversation_id = _conversation_id.get()
    if conversation_id is not None:
        context["conversation_id"] = conversation_id

    user_id = _user_id.get()
    if user_id is not None:
        context["user_id"] = user_id

    workflow_id = _workflow_id.get()
    if workflow_id is not None:
        context["workflow_id"] = workflow_id

    workflow_type = _workflow_type.get()
    if workflow_type is not None:
        context["workflow_type"] = workflow_type

    turn_number = _conversation_turn.get()
    if turn_number is not None:
        context["turn_number"] = turn_number

    custom = _custom_attributes.get()
    if custom:
        context.update(custom)

    return context


def clear_context() -> None:
    """Clear all context variables."""
    _conversation_id.set(None)
    _user_id.set(None)
    _workflow_id.set(None)
    _workflow_type.set(None)
    _conversation_turn.set(None)
    _custom_attributes.set({})


@contextmanager
def conversation_context(
    conversation_id: str,
    user_id: Optional[str] = None,
    turn_number: Optional[int] = None,
    **custom_attrs,
):
    """
    Context manager for conversation tracking.

    All LLM spans created within this context will automatically
    have conversation attributes added.

    Args:
        conversation_id: Conversation identifier
        user_id: Optional user identifier
        turn_number: Optional turn number
        **custom_attrs: Additional custom attributes

    Example:
        ```python
        with conversation_context(conversation_id="user_123_session", user_id="user_123"):
            # All LLM calls automatically tagged with conversation_id
            response1 = client.chat.completions.create(...)
            response2 = client.chat.completions.create(...)
        ```
    """
    # Save previous values
    prev_conv_id = _conversation_id.get()
    prev_user_id = _user_id.get()
    prev_turn = _conversation_turn.get()
    prev_custom = _custom_attributes.get()

    try:
        # Set new values
        _conversation_id.set(conversation_id)
        if user_id is not None:
            _user_id.set(user_id)
        if turn_number is not None:
            _conversation_turn.set(turn_number)
        if custom_attrs:
            _custom_attributes.set(custom_attrs)

        yield

    finally:
        # Restore previous values
        _conversation_id.set(prev_conv_id)
        _user_id.set(prev_user_id)
        _conversation_turn.set(prev_turn)
        _custom_attributes.set(prev_custom)


@contextmanager
def workflow_context(
    workflow_id: str,
    workflow_type: Optional[str] = None,
    user_id: Optional[str] = None,
    **custom_attrs,
):
    """
    Context manager for workflow tracking.

    All LLM and tool spans created within this context will automatically
    have workflow attributes added.

    Args:
        workflow_id: Workflow identifier
        workflow_type: Type of workflow (e.g., "customer_support", "rag_search")
        user_id: Optional user identifier
        **custom_attrs: Additional custom attributes

    Example:
        ```python
        with workflow_context(workflow_id="search_workflow", workflow_type="rag_search"):
            # All operations automatically tagged with workflow_id
            docs = retrieve_documents(query)
            context = generate_context(docs)
            response = generate_response(context)
        ```

        # Can be nested with conversation_context:
        ```python
        with conversation_context(conversation_id="session_123"):
            with workflow_context(workflow_id="search_products"):
                # Both conversation AND workflow tracked
                results = search_and_recommend()
        ```
    """
    # Save previous values
    prev_wf_id = _workflow_id.get()
    prev_wf_type = _workflow_type.get()
    prev_user_id = _user_id.get()
    prev_custom = _custom_attributes.get()

    try:
        # Set new values
        _workflow_id.set(workflow_id)
        if workflow_type is not None:
            _workflow_type.set(workflow_type)
        if user_id is not None:
            _user_id.set(user_id)
        if custom_attrs:
            _custom_attributes.set(custom_attrs)

        yield

    finally:
        # Restore previous values
        _workflow_id.set(prev_wf_id)
        _workflow_type.set(prev_wf_type)
        _user_id.set(prev_user_id)
        _custom_attributes.set(prev_custom)
