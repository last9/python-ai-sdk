"""Tests for context managers (conversation_context, workflow_context, propagate_attributes)"""

import pytest
import threading
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from last9_genai import (
    conversation_context,
    workflow_context,
    propagate_attributes,
    get_current_context,
    GenAIAttributes,
)


class TestConversationContext:
    """Test conversation_context() context manager"""

    def test_conversation_context_basic(self, tracer_setup):
        """Test basic conversation_context usage"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="conv_123"):
            with tracer.start_as_current_span("test_span") as span:
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_123"

    def test_conversation_context_with_user_id(self, tracer_setup):
        """Test conversation_context with user_id"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="conv_456", user_id="user_789"):
            with tracer.start_as_current_span("test_span") as span:
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_456"
        assert spans[0].attributes["user.id"] == "user_789"

    def test_conversation_context_propagates_to_nested_spans(self, tracer_setup):
        """Test that conversation context propagates to all nested spans"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="nested_conv"):
            with tracer.start_as_current_span("parent") as parent:
                with tracer.start_as_current_span("child") as child:
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2

        # Both spans should have conversation_id
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "nested_conv"
        assert spans[1].attributes[GenAIAttributes.CONVERSATION_ID] == "nested_conv"

    def test_conversation_context_cleanup(self, tracer_setup):
        """Test that context is cleaned up after exit"""
        tracer, memory_exporter = tracer_setup

        # Inside context
        with conversation_context(conversation_id="temp_conv"):
            context = get_current_context()
            assert "conversation_id" in context
            assert context["conversation_id"] == "temp_conv"

        # Outside context - should be cleaned up
        context = get_current_context()
        assert "conversation_id" not in context or context.get("conversation_id") != "temp_conv"

    def test_multiple_conversation_contexts(self, tracer_setup):
        """Test multiple separate conversation contexts"""
        tracer, memory_exporter = tracer_setup

        # First conversation
        with conversation_context(conversation_id="conv_1"):
            with tracer.start_as_current_span("span_1"):
                pass

        # Second conversation
        with conversation_context(conversation_id="conv_2"):
            with tracer.start_as_current_span("span_2"):
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_1"
        assert spans[1].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_2"

    def test_conversation_context_override(self, tracer_setup):
        """Test that inner context overrides outer context"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="outer_conv"):
            with conversation_context(conversation_id="inner_conv"):
                with tracer.start_as_current_span("test_span"):
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        # Inner context should take precedence
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "inner_conv"


class TestWorkflowContext:
    """Test workflow_context() context manager"""

    def test_workflow_context_basic(self, tracer_setup):
        """Test basic workflow_context usage"""
        tracer, memory_exporter = tracer_setup

        with workflow_context(workflow_id="wf_123"):
            with tracer.start_as_current_span("test_span"):
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["workflow.id"] == "wf_123"

    def test_workflow_context_with_type(self, tracer_setup):
        """Test workflow_context with workflow_type"""
        tracer, memory_exporter = tracer_setup

        with workflow_context(workflow_id="wf_456", workflow_type="rag"):
            with tracer.start_as_current_span("test_span"):
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["workflow.id"] == "wf_456"
        assert spans[0].attributes["workflow.type"] == "rag"

    def test_workflow_context_propagates_to_nested_spans(self, tracer_setup):
        """Test that workflow context propagates to nested spans"""
        tracer, memory_exporter = tracer_setup

        with workflow_context(workflow_id="nested_wf"):
            with tracer.start_as_current_span("step_1"):
                with tracer.start_as_current_span("step_2"):
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2

        # Both spans should have workflow_id
        assert spans[0].attributes["workflow.id"] == "nested_wf"
        assert spans[1].attributes["workflow.id"] == "nested_wf"

    def test_workflow_context_cleanup(self, tracer_setup):
        """Test that workflow context is cleaned up"""
        tracer, memory_exporter = tracer_setup

        with workflow_context(workflow_id="temp_wf"):
            context = get_current_context()
            assert "workflow_id" in context
            assert context["workflow_id"] == "temp_wf"

        # Outside context
        context = get_current_context()
        assert "workflow_id" not in context or context.get("workflow_id") != "temp_wf"


class TestNestedContexts:
    """Test nested conversation and workflow contexts"""

    def test_conversation_with_nested_workflow(self, tracer_setup):
        """Test workflow nested inside conversation"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="conv_main", user_id="user_1"):
            # Span before workflow
            with tracer.start_as_current_span("before_workflow"):
                pass

            # Workflow inside conversation
            with workflow_context(workflow_id="wf_sub", workflow_type="search"):
                with tracer.start_as_current_span("workflow_span"):
                    pass

            # Span after workflow
            with tracer.start_as_current_span("after_workflow"):
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 3

        # First span: conversation only
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_main"
        assert spans[0].attributes["user.id"] == "user_1"
        assert "workflow.id" not in spans[0].attributes

        # Second span: both conversation and workflow
        assert spans[1].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_main"
        assert spans[1].attributes["user.id"] == "user_1"
        assert spans[1].attributes["workflow.id"] == "wf_sub"
        assert spans[1].attributes["workflow.type"] == "search"

        # Third span: conversation only (workflow exited)
        assert spans[2].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_main"
        assert spans[2].attributes["user.id"] == "user_1"
        assert "workflow.id" not in spans[2].attributes

    def test_multiple_workflows_in_conversation(self, tracer_setup):
        """Test multiple workflows within same conversation"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="conv_123"):
            # First workflow
            with workflow_context(workflow_id="wf_1"):
                with tracer.start_as_current_span("wf1_span"):
                    pass

            # Second workflow
            with workflow_context(workflow_id="wf_2"):
                with tracer.start_as_current_span("wf2_span"):
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2

        # Both have conversation, but different workflows
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_123"
        assert spans[0].attributes["workflow.id"] == "wf_1"

        assert spans[1].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_123"
        assert spans[1].attributes["workflow.id"] == "wf_2"

    def test_deeply_nested_contexts(self, tracer_setup):
        """Test deeply nested conversation and workflow contexts"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="level_0"):
            with workflow_context(workflow_id="wf_level_1"):
                with conversation_context(conversation_id="level_2"):
                    with tracer.start_as_current_span("deep_span"):
                        pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Innermost context values should be used
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "level_2"
        assert spans[0].attributes["workflow.id"] == "wf_level_1"


class TestPropagateAttributes:
    """Test propagate_attributes() context manager"""

    def test_propagate_attributes_basic(self, tracer_setup):
        """Test basic propagate_attributes usage"""
        tracer, memory_exporter = tracer_setup

        with propagate_attributes(environment="production", version="1.0.0"):
            with tracer.start_as_current_span("test_span"):
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["custom.environment"] == "production"
        assert spans[0].attributes["custom.version"] == "1.0.0"

    def test_propagate_attributes_with_standard_contexts(self, tracer_setup):
        """Test propagate_attributes combined with conversation_context"""
        tracer, memory_exporter = tracer_setup

        with conversation_context(conversation_id="conv_abc"):
            with propagate_attributes(region="us-east-1", tier="premium"):
                with tracer.start_as_current_span("test_span"):
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Should have both conversation and custom attributes
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_abc"
        assert spans[0].attributes["custom.region"] == "us-east-1"
        assert spans[0].attributes["custom.tier"] == "premium"

    def test_propagate_attributes_cleanup(self, tracer_setup):
        """Test that custom attributes are cleaned up"""
        tracer, memory_exporter = tracer_setup

        with propagate_attributes(temp_attr="temp_value"):
            context = get_current_context()
            assert "temp_attr" in context
            assert context["temp_attr"] == "temp_value"

        # Outside context
        context = get_current_context()
        assert "temp_attr" not in context or context.get("temp_attr") != "temp_value"


class TestGetCurrentContext:
    """Test get_current_context() helper"""

    def test_get_current_context_empty(self, tracer_setup):
        """Test get_current_context when no context is set"""
        tracer, memory_exporter = tracer_setup

        """Test get_current_context when no context is set"""
        context = get_current_context()
        assert isinstance(context, dict)

    def test_get_current_context_with_conversation(self, tracer_setup):
        """Test get_current_context inside conversation"""
        tracer, memory_exporter = tracer_setup

        """Test get_current_context inside conversation"""
        with conversation_context(conversation_id="test_conv", user_id="test_user"):
            context = get_current_context()
            assert context["conversation_id"] == "test_conv"
            assert context["user_id"] == "test_user"

    def test_get_current_context_with_workflow(self, tracer_setup):
        """Test get_current_context inside workflow"""
        tracer, memory_exporter = tracer_setup

        """Test get_current_context inside workflow"""
        with workflow_context(workflow_id="test_wf", workflow_type="test"):
            context = get_current_context()
            assert context["workflow_id"] == "test_wf"
            assert context["workflow_type"] == "test"

    def test_get_current_context_with_both(self, tracer_setup):
        """Test get_current_context with both conversation and workflow"""
        tracer, memory_exporter = tracer_setup

        """Test get_current_context with both conversation and workflow"""
        with conversation_context(conversation_id="conv_1"):
            with workflow_context(workflow_id="wf_1"):
                context = get_current_context()
                assert context["conversation_id"] == "conv_1"
                assert context["workflow_id"] == "wf_1"


class TestThreadSafety:
    """Test thread safety of context managers"""

    def test_context_isolation_across_threads(self, tracer_setup):
        """Test that contexts are isolated across threads"""
        tracer, memory_exporter = tracer_setup

        """Test that contexts are isolated across threads"""
        results = []

        def thread_function(thread_id: int):
            with conversation_context(conversation_id=f"conv_{thread_id}"):
                with tracer.start_as_current_span(f"span_{thread_id}"):
                    context = get_current_context()
                    results.append(
                        {"thread_id": thread_id, "conversation_id": context.get("conversation_id")}
                    )

        # Create multiple threads with different contexts
        threads = []
        for i in range(5):
            t = threading.Thread(target=thread_function, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Each thread should have its own conversation_id
        assert len(results) == 5
        for i, result in enumerate(results):
            # Results might be in different order due to threading
            # Just check that each has a unique conversation_id
            assert result["conversation_id"] in [f"conv_{j}" for j in range(5)]

        # Check spans - should have 5 spans with different conversation IDs
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 5

        conv_ids = [span.attributes[GenAIAttributes.CONVERSATION_ID] for span in spans]
        expected_ids = [f"conv_{i}" for i in range(5)]
        assert sorted(conv_ids) == sorted(expected_ids)


class TestContextEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_conversation_id(self, tracer_setup):
        """Test handling of empty conversation_id"""
        tracer, memory_exporter = tracer_setup

        """Test handling of empty conversation_id"""
        with conversation_context(conversation_id=""):
            with tracer.start_as_current_span("test_span"):
                pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        # Empty string should still be set
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == ""

    def test_special_characters_in_ids(self, tracer_setup):
        """Test IDs with special characters"""
        tracer, memory_exporter = tracer_setup

        """Test IDs with special characters"""
        special_conv_id = "conv-123_test@domain.com"
        special_wf_id = "wf:2024-01-01/test"

        with conversation_context(conversation_id=special_conv_id):
            with workflow_context(workflow_id=special_wf_id):
                with tracer.start_as_current_span("test_span"):
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == special_conv_id
        assert spans[0].attributes["workflow.id"] == special_wf_id

    def test_context_with_no_spans(self, tracer_setup):
        """Test that context managers work even if no spans are created"""
        tracer, memory_exporter = tracer_setup

        """Test that context managers work even if no spans are created"""
        # This should not raise an error
        with conversation_context(conversation_id="no_spans"):
            pass

        with workflow_context(workflow_id="no_spans"):
            pass

        # No spans should have been created
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_context_reentry(self, tracer_setup):
        """Test entering same context multiple times"""
        tracer, memory_exporter = tracer_setup

        """Test entering same context multiple times"""
        with conversation_context(conversation_id="reentry_test"):
            with tracer.start_as_current_span("span_1"):
                pass

            # Exit and re-enter same context
            with conversation_context(conversation_id="reentry_test"):
                with tracer.start_as_current_span("span_2"):
                    pass

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "reentry_test"
        assert spans[1].attributes[GenAIAttributes.CONVERSATION_ID] == "reentry_test"
