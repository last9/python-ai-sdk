"""Tests for @observe() decorator"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from last9_genai import observe, ModelPricing, GenAIAttributes, Last9Attributes


class MockOpenAIResponse:
    """Mock OpenAI API response"""

    def __init__(
        self,
        response_id="chatcmpl-123",
        model="gpt-4o-mini",
        content="Hello!",
        input_tokens=10,
        output_tokens=20,
    ):
        self.id = response_id
        self.model = model
        self.choices = [
            type(
                "Choice",
                (),
                {"message": type("Message", (), {"content": content})(), "finish_reason": "stop"},
            )()
        ]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )()


class MockAnthropicResponse:
    """Mock Anthropic API response"""

    def __init__(
        self,
        response_id="msg_123",
        model="claude-3-haiku",
        content="Hello!",
        input_tokens=10,
        output_tokens=20,
    ):
        self.id = response_id
        self.model = model
        self.content = [type("Content", (), {"text": content})()]
        self.usage = type(
            "Usage", (), {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )()


class TestObserveDecorator:
    """Test @observe() decorator"""

    def test_basic_decorator(self, tracer_setup):
        """Test basic @observe() decorator"""
        tracer, memory_exporter = tracer_setup

        @observe()
        def simple_function(x: int) -> int:
            return x * 2

        result = simple_function(5)
        assert result == 10

        # Check span was created
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "simple_function"

    def test_decorator_with_custom_name(self, tracer_setup):
        """Test @observe() with custom span name"""
        tracer, memory_exporter = tracer_setup

        @observe(name="custom_span_name")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "custom_span_name"

    def test_decorator_captures_input_output(self, tracer_setup):
        """Test that decorator captures input and output"""
        tracer, memory_exporter = tracer_setup

        @observe(capture_input=True, capture_output=True)
        def echo_function(message: str) -> str:
            return f"Echo: {message}"

        result = echo_function("Hello")
        assert result == "Echo: Hello"

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Check events for input/output
        events = spans[0].events
        event_names = [e.name for e in events]
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names

    def test_decorator_with_tags(self, tracer_setup):
        """Test @observe() with tags"""
        tracer, memory_exporter = tracer_setup

        @observe(tags=["production", "critical", "openai"])
        def tagged_function():
            return "result"

        tagged_function()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Check tags attribute
        assert "tags" in spans[0].attributes
        tags = spans[0].attributes["tags"]
        assert tags == ("production", "critical", "openai")

    def test_decorator_with_metadata(self, tracer_setup):
        """Test @observe() with metadata"""
        tracer, memory_exporter = tracer_setup

        @observe(metadata={"version": "1.0.0", "environment": "production"})
        def metadata_function():
            return "result"

        metadata_function()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Check metadata attributes
        attrs = spans[0].attributes
        assert attrs["metadata.version"] == "1.0.0"
        assert attrs["metadata.environment"] == "production"

    def test_decorator_with_category_emits_user_category(self, tracer_setup):
        """Test that metadata.category also emits user.category"""
        tracer, memory_exporter = tracer_setup

        @observe(metadata={"category": "customer_support", "priority": "high"})
        def support_function():
            return "result"

        support_function()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Check both metadata.category and user.category are set
        assert attrs["metadata.category"] == "customer_support"
        assert attrs["user.category"] == "customer_support"
        assert attrs["metadata.priority"] == "high"

    def test_decorator_span_kind(self, tracer_setup):
        """Test @observe() sets correct span kind"""
        tracer, memory_exporter = tracer_setup

        @observe(as_type="llm")
        def llm_function():
            return MockOpenAIResponse()

        llm_function()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == "llm"

    def test_decorator_with_openai_response(self, tracer_setup):
        """Test decorator extracts OpenAI response attributes"""
        tracer, memory_exporter = tracer_setup

        pricing = {"gpt-4o-mini": ModelPricing(input=0.15, output=0.60)}

        @observe(pricing=pricing)
        def call_openai(prompt: str):
            return MockOpenAIResponse(
                response_id="chatcmpl-abc123",
                model="gpt-4o-mini",
                content="Hello from OpenAI!",
                input_tokens=10,
                output_tokens=25,
            )

        result = call_openai("Say hello")
        assert result.id == "chatcmpl-abc123"

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Check response ID
        assert attrs[GenAIAttributes.RESPONSE_ID] == "chatcmpl-abc123"
        # Check model
        assert attrs[GenAIAttributes.RESPONSE_MODEL] == "gpt-4o-mini"
        # Check tokens
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 10
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 25
        # Check cost was calculated
        assert GenAIAttributes.USAGE_COST_USD in attrs

    def test_decorator_with_anthropic_response(self, tracer_setup):
        """Test decorator extracts Anthropic response attributes"""
        tracer, memory_exporter = tracer_setup

        pricing = {"claude-3-haiku": ModelPricing(input=0.25, output=1.25)}

        @observe(pricing=pricing)
        def call_anthropic(prompt: str):
            return MockAnthropicResponse(
                response_id="msg_xyz789",
                model="claude-3-haiku",
                content="Hello from Claude!",
                input_tokens=15,
                output_tokens=30,
            )

        result = call_anthropic("Say hello")
        assert result.id == "msg_xyz789"

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Check response ID
        assert attrs[GenAIAttributes.RESPONSE_ID] == "msg_xyz789"
        # Check model
        assert attrs[GenAIAttributes.RESPONSE_MODEL] == "claude-3-haiku"
        # Check tokens
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 15
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 30
        # Check cost was calculated
        assert GenAIAttributes.USAGE_COST_USD in attrs

    def test_decorator_extracts_model_parameters(self, tracer_setup):
        """Test decorator extracts model parameters from kwargs"""
        tracer, memory_exporter = tracer_setup

        @observe()
        def llm_with_params(prompt: str, temperature=0.7, max_tokens=100, top_p=0.9):
            return MockOpenAIResponse()

        llm_with_params("test", temperature=0.8, max_tokens=150, top_p=0.95)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        assert attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.8
        assert attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 150
        assert attrs[GenAIAttributes.REQUEST_TOP_P] == 0.95

    def test_decorator_error_handling(self, tracer_setup):
        """Test that decorator handles exceptions properly"""
        tracer, memory_exporter = tracer_setup

        @observe()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Check span recorded the error
        span = spans[0]
        assert span.status.status_code.name == "ERROR"

        # Check exception was recorded in events
        events = span.events
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1  # May have multiple from decorator + processor

    def test_decorator_with_capture_args(self, tracer_setup):
        """Test decorator captures function arguments"""
        tracer, memory_exporter = tracer_setup

        @observe(capture_args=True)
        def function_with_args(x: int, y: str, z: float = 3.14):
            return f"{x}-{y}-{z}"

        function_with_args(42, "hello", z=2.71)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        assert attrs["function.arg.x"] == 42
        assert attrs["function.arg.y"] == "hello"
        assert attrs["function.arg.z"] == 2.71

    def test_decorator_without_capture_args(self, tracer_setup):
        """Test decorator respects capture_args=False"""
        tracer, memory_exporter = tracer_setup

        @observe(capture_args=False)
        def function_with_args(secret_key: str, data: str):
            return data

        function_with_args("secret123", "public_data")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Function args should NOT be captured
        assert "function.arg.secret_key" not in attrs
        assert "function.arg.data" not in attrs

    def test_decorator_without_capture_input(self, tracer_setup):
        """Test decorator respects capture_input=False"""
        tracer, memory_exporter = tracer_setup

        @observe(capture_input=False, capture_output=True)
        def sensitive_input(secret: str):
            return "output"

        sensitive_input("secret_data")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        events = spans[0].events
        event_names = [e.name for e in events]
        # Input event should NOT be present
        assert "gen_ai.content.prompt" not in event_names
        # But output should be captured
        assert "gen_ai.content.completion" in event_names

    def test_decorator_without_capture_output(self, tracer_setup):
        """Test decorator respects capture_output=False"""
        tracer, memory_exporter = tracer_setup

        @observe(capture_input=True, capture_output=False)
        def sensitive_output(data: str):
            return "sensitive_result"

        sensitive_output("input")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        events = spans[0].events
        event_names = [e.name for e in events]
        # Input should be captured
        assert "gen_ai.content.prompt" in event_names
        # But output event should NOT be present
        assert "gen_ai.content.completion" not in event_names

    def test_decorator_with_different_span_kinds(self, tracer_setup):
        """Test decorator with different as_type values"""
        tracer, memory_exporter = tracer_setup

        @observe(as_type="tool")
        def tool_function():
            return "result"

        @observe(as_type="chain")
        def chain_function():
            return "result"

        tool_function()
        chain_function()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2

        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == "tool"
        assert spans[1].attributes[Last9Attributes.L9_SPAN_KIND] == "chain"

    def test_decorator_cost_calculation(self, tracer_setup):
        """Test that decorator calculates cost correctly"""
        tracer, memory_exporter = tracer_setup

        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

        @observe(pricing=pricing)
        def expensive_call():
            return MockOpenAIResponse(model="gpt-4o", input_tokens=1000, output_tokens=500)

        expensive_call()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Check cost calculation
        # Input: 1000 tokens * $2.50 per million = $0.0025
        # Output: 500 tokens * $10.0 per million = $0.005
        # Total: $0.0075
        total_cost = attrs[GenAIAttributes.USAGE_COST_USD]
        assert total_cost == pytest.approx(0.0075, rel=1e-6)

    def test_decorator_without_pricing(self, tracer_setup):
        """Test decorator works without pricing (no cost calculation)"""
        tracer, memory_exporter = tracer_setup

        @observe()
        def call_without_pricing():
            return MockOpenAIResponse()

        call_without_pricing()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Should have usage tokens
        assert GenAIAttributes.USAGE_INPUT_TOKENS in attrs
        assert GenAIAttributes.USAGE_OUTPUT_TOKENS in attrs
        # But no cost attributes
        assert GenAIAttributes.USAGE_COST_USD not in attrs


@pytest.mark.asyncio
class TestObserveDecoratorAsync:
    """Test @observe() decorator with async functions"""

    async def test_async_decorator_basic(self, tracer_setup):
        """Test @observe() with async function"""
        tracer, memory_exporter = tracer_setup

        @observe()
        async def async_function(x: int):
            return x * 2

        result = await async_function(5)
        assert result == 10

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async_function"

    async def test_async_decorator_with_llm_response(self, tracer_setup):
        """Test async decorator with LLM response"""
        tracer, memory_exporter = tracer_setup

        pricing = {"gpt-4o-mini": ModelPricing(input=0.15, output=0.60)}

        @observe(pricing=pricing, tags=["async", "test"])
        async def async_llm_call(prompt: str):
            # Simulate async API call
            return MockOpenAIResponse(response_id="chatcmpl-async123", content="Async response")

        result = await async_llm_call("test prompt")
        assert result.id == "chatcmpl-async123"

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        assert attrs[GenAIAttributes.RESPONSE_ID] == "chatcmpl-async123"
        assert attrs["tags"] == ("async", "test")

    async def test_async_decorator_error_handling(self, tracer_setup):
        """Test async decorator handles errors properly"""
        tracer, memory_exporter = tracer_setup

        @observe()
        async def failing_async():
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError, match="Async error"):
            await failing_async()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"


class TestDecoratorIntegration:
    """Integration tests for decorator with context managers"""

    def test_decorator_with_conversation_context(self, tracer_setup):
        """Test decorator picks up conversation context"""
        tracer, memory_exporter = tracer_setup

        from last9_genai import conversation_context

        @observe()
        def llm_call():
            return MockOpenAIResponse()

        with conversation_context(conversation_id="test_conv_123", user_id="user_456"):
            llm_call()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Should automatically have conversation context
        assert attrs[GenAIAttributes.CONVERSATION_ID] == "test_conv_123"
        assert attrs["user.id"] == "user_456"

    def test_decorator_with_workflow_context(self, tracer_setup):
        """Test decorator picks up workflow context"""
        tracer, memory_exporter = tracer_setup

        from last9_genai import workflow_context

        @observe()
        def workflow_step():
            return MockOpenAIResponse()

        with workflow_context(workflow_id="test_workflow", workflow_type="rag"):
            workflow_step()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = spans[0].attributes
        # Should automatically have workflow context
        assert attrs["workflow.id"] == "test_workflow"
        assert attrs["workflow.type"] == "rag"

    def test_nested_decorated_functions(self, tracer_setup):
        """Test nested decorated functions create proper span hierarchy"""
        tracer, memory_exporter = tracer_setup

        @observe(as_type="chain")
        def parent_chain():
            child_llm()
            return "parent_result"

        @observe(as_type="llm")
        def child_llm():
            return MockOpenAIResponse()

        parent_chain()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2

        # Check parent-child relationship
        child_span = spans[0]
        parent_span = spans[1]

        assert child_span.parent.span_id == parent_span.context.span_id
        assert parent_span.attributes[Last9Attributes.L9_SPAN_KIND] == "chain"
        assert child_span.attributes[Last9Attributes.L9_SPAN_KIND] == "llm"
