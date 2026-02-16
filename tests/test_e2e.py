"""
End-to-end integration tests for last9-genai SDK

Tests complete workflows including:
- OpenTelemetry setup with Last9SpanProcessor
- Conversation and workflow context managers
- @observe() decorator with real LLM-like responses
- Cost tracking and aggregation
- Span hierarchy and attribute propagation
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from last9_genai import (
    Last9SpanProcessor,
    conversation_context,
    workflow_context,
    observe,
    ModelPricing,
    GenAIAttributes,
    Last9Attributes,
)


# Mock LLM response objects
class MockOpenAIResponse:
    """Mock OpenAI chat completion response"""

    def __init__(
        self,
        content: str,
        model: str = "gpt-4o",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ):
        self.id = "chatcmpl-123"
        self.model = model
        self.choices = [
            type(
                "obj",
                (object,),
                {
                    "message": type("obj", (object,), {"content": content})(),
                    "finish_reason": "stop",
                },
            )()
        ]
        self.usage = type(
            "obj",
            (object,),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )()


class MockAnthropicResponse:
    """Mock Anthropic message response"""

    def __init__(
        self,
        content: str,
        model: str = "claude-3-5-sonnet-20241022",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        self.id = "msg_123"
        self.model = model
        self.content = [type("obj", (object,), {"text": content})()]
        self.usage = type(
            "obj", (object,), {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )()
        self.stop_reason = "end_turn"


# E2E tests use the tracer_setup fixture from conftest.py


class TestE2EConversationTracking:
    """End-to-end tests for conversation tracking"""

    def test_multi_turn_conversation_with_openai(self, tracer_setup):
        """Test complete multi-turn conversation workflow with OpenAI"""
        tracer, memory_exporter = tracer_setup

        # Define model pricing for cost tracking
        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }

        # Define conversation functions with @observe decorator
        @observe(pricing=pricing, capture_input=True, capture_output=True)
        def call_openai(prompt: str) -> MockOpenAIResponse:
            """Simulate OpenAI API call"""
            return MockOpenAIResponse(
                content=f"Response to: {prompt}",
                model="gpt-4o",
                prompt_tokens=len(prompt) // 4,
                completion_tokens=50,
            )

        # Run multi-turn conversation
        conversation_id = "conv_e2e_test_123"
        user_id = "user_alice"

        with conversation_context(conversation_id=conversation_id, user_id=user_id):
            # Turn 1
            response1 = call_openai("What is the capital of France?")

            # Turn 2
            response2 = call_openai("What's the population?")

            # Turn 3
            response3 = call_openai("Tell me about its history")

        # Verify spans
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 3, "Should have 3 LLM call spans"

        # Verify all spans have conversation tracking
        for span in spans:
            assert span.attributes[GenAIAttributes.CONVERSATION_ID] == conversation_id
            assert span.attributes["user.id"] == user_id
            assert GenAIAttributes.RESPONSE_MODEL in span.attributes
            assert span.attributes[GenAIAttributes.RESPONSE_MODEL] == "gpt-4o"

            # Verify cost tracking
            assert GenAIAttributes.USAGE_COST_USD in span.attributes
            assert span.attributes[GenAIAttributes.USAGE_COST_USD] > 0

            # Verify span kind
            assert span.attributes[Last9Attributes.L9_SPAN_KIND] == "llm"

        # Verify input/output capture
        first_span = spans[0]
        assert "function.arg.prompt" in first_span.attributes
        assert "France" in first_span.attributes["function.arg.prompt"]
        # Output is available in response object, not as separate attribute

    def test_conversation_with_anthropic(self, tracer_setup):
        """Test conversation tracking with Anthropic Claude"""
        tracer, memory_exporter = tracer_setup

        pricing = {
            "claude-3-5-sonnet-20241022": ModelPricing(input=3.0, output=15.0),
        }

        @observe(pricing=pricing)
        def call_claude(prompt: str) -> MockAnthropicResponse:
            """Simulate Anthropic API call"""
            return MockAnthropicResponse(
                content=f"Claude response: {prompt}",
                model="claude-3-5-sonnet-20241022",
                input_tokens=100,
                output_tokens=150,
            )

        with conversation_context(conversation_id="conv_claude_test"):
            response = call_claude("Explain quantum computing")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.attributes[GenAIAttributes.CONVERSATION_ID] == "conv_claude_test"
        assert span.attributes[GenAIAttributes.RESPONSE_MODEL] == "claude-3-5-sonnet-20241022"
        assert span.attributes[GenAIAttributes.USAGE_INPUT_TOKENS] == 100
        assert span.attributes[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 150

        # Verify Claude-specific cost calculation
        expected_cost = (100 / 1_000_000) * 3.0 + (150 / 1_000_000) * 15.0
        assert span.attributes[GenAIAttributes.USAGE_COST_USD] == pytest.approx(
            expected_cost, rel=1e-6
        )


class TestE2EWorkflowTracking:
    """End-to-end tests for workflow tracking"""

    def test_rag_workflow_with_multiple_steps(self, tracer_setup):
        """Test complete RAG workflow with retrieval, context building, and generation"""
        tracer, memory_exporter = tracer_setup

        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }

        # Define workflow steps
        @observe(as_type="tool", name="retrieve_documents")
        def retrieve_documents(query: str) -> list:
            """Simulate document retrieval"""
            return [
                {"id": 1, "text": "Paris is the capital of France"},
                {"id": 2, "text": "Paris has a population of 2.1 million"},
            ]

        @observe(as_type="tool", name="build_context")
        def build_context(documents: list) -> str:
            """Build context from retrieved documents"""
            return "\n".join([doc["text"] for doc in documents])

        @observe(pricing=pricing, name="generate_response")
        def generate_response(query: str, context: str) -> MockOpenAIResponse:
            """Generate final response using LLM"""
            prompt = f"Context: {context}\n\nQuestion: {query}"
            return MockOpenAIResponse(
                content="Paris is the capital of France with 2.1 million people.",
                model="gpt-4o",
                prompt_tokens=len(prompt) // 4,
                completion_tokens=30,
            )

        # Execute RAG workflow
        workflow_id = "rag_workflow_001"

        with workflow_context(workflow_id=workflow_id, workflow_type="rag_search"):
            query = "What is Paris?"

            # Step 1: Retrieve
            docs = retrieve_documents(query)

            # Step 2: Build context
            context = build_context(docs)

            # Step 3: Generate
            response = generate_response(query, context)

        # Verify spans
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 3, "Should have 3 spans (retrieve, build, generate)"

        # Verify all spans have workflow tracking
        for span in spans:
            assert span.attributes["workflow.id"] == workflow_id
            assert span.attributes["workflow.type"] == "rag_search"

        # Verify span kinds
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == "tool"  # retrieve
        assert spans[1].attributes[Last9Attributes.L9_SPAN_KIND] == "tool"  # build_context
        assert spans[2].attributes[Last9Attributes.L9_SPAN_KIND] == "llm"  # generate

        # Verify only LLM span has cost
        assert GenAIAttributes.USAGE_COST_USD not in spans[0].attributes
        assert GenAIAttributes.USAGE_COST_USD not in spans[1].attributes
        assert GenAIAttributes.USAGE_COST_USD in spans[2].attributes

    def test_nested_conversation_and_workflow(self, tracer_setup):
        """Test conversation nested inside workflow"""
        tracer, memory_exporter = tracer_setup

        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

        @observe(pricing=pricing)
        def chat(message: str) -> MockOpenAIResponse:
            return MockOpenAIResponse(content=f"Reply: {message}", model="gpt-4o")

        @observe(as_type="tool", name="search_products")
        def search_products(query: str) -> list:
            return [{"name": "Product A"}, {"name": "Product B"}]

        # Nested context: conversation inside workflow
        with conversation_context(conversation_id="conv_nested", user_id="user_bob"):
            with workflow_context(workflow_id="wf_shopping", workflow_type="product_search"):
                # Tool call within workflow and conversation
                products = search_products("laptops")

                # LLM call within workflow and conversation
                response = chat("Show me laptops")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 2

        # Both spans should have both conversation AND workflow context
        for span in spans:
            assert span.attributes[GenAIAttributes.CONVERSATION_ID] == "conv_nested"
            assert span.attributes["user.id"] == "user_bob"
            assert span.attributes["workflow.id"] == "wf_shopping"
            assert span.attributes["workflow.type"] == "product_search"


class TestE2ECostTracking:
    """End-to-end tests for cost tracking and aggregation"""

    def test_cost_aggregation_across_workflow(self, tracer_setup):
        """Test that costs are properly aggregated across multiple LLM calls"""
        tracer, memory_exporter = tracer_setup

        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
            "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
        }

        @observe(pricing=pricing)
        def expensive_call() -> MockOpenAIResponse:
            return MockOpenAIResponse(
                content="Expensive response",
                model="gpt-4o",
                prompt_tokens=1000,
                completion_tokens=500,
            )

        @observe(pricing=pricing)
        def cheap_call() -> MockOpenAIResponse:
            return MockOpenAIResponse(
                content="Cheap response",
                model="gpt-3.5-turbo",
                prompt_tokens=100,
                completion_tokens=50,
            )

        with workflow_context(workflow_id="wf_cost_test"):
            resp1 = expensive_call()
            resp2 = cheap_call()
            resp3 = expensive_call()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 3

        # Calculate expected costs
        expensive_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.0
        cheap_cost = (100 / 1_000_000) * 0.50 + (50 / 1_000_000) * 1.50

        # Verify individual costs
        assert spans[0].attributes[GenAIAttributes.USAGE_COST_USD] == pytest.approx(
            expensive_cost, rel=1e-6
        )
        assert spans[1].attributes[GenAIAttributes.USAGE_COST_USD] == pytest.approx(
            cheap_cost, rel=1e-6
        )
        assert spans[2].attributes[GenAIAttributes.USAGE_COST_USD] == pytest.approx(
            expensive_cost, rel=1e-6
        )

        # Total cost should be: 2 * expensive + 1 * cheap
        total_cost = sum(span.attributes[GenAIAttributes.USAGE_COST_USD] for span in spans)
        expected_total = 2 * expensive_cost + cheap_cost
        assert total_cost == pytest.approx(expected_total, rel=1e-6)

    def test_no_cost_without_pricing(self, tracer_setup):
        """Test that system works without pricing (no cost tracking)"""
        tracer, memory_exporter = tracer_setup

        # No pricing provided
        @observe()
        def call_without_pricing() -> MockOpenAIResponse:
            return MockOpenAIResponse(content="Response", model="gpt-4o")

        with conversation_context(conversation_id="conv_no_cost"):
            response = call_without_pricing()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Should have tokens but no cost
        assert GenAIAttributes.USAGE_INPUT_TOKENS in spans[0].attributes
        assert GenAIAttributes.USAGE_OUTPUT_TOKENS in spans[0].attributes
        assert GenAIAttributes.USAGE_COST_USD not in spans[0].attributes


class TestE2EAttributePropagation:
    """End-to-end tests for context propagation and attribute inheritance"""

    def test_custom_attributes_propagate(self, tracer_setup):
        """Test that custom attributes propagate through context managers"""
        tracer, memory_exporter = tracer_setup

        from last9_genai import propagate_attributes

        @observe()
        def nested_call() -> str:
            return "result"

        with propagate_attributes(environment="production", version="1.0.0"):
            with conversation_context(conversation_id="conv_custom"):
                result = nested_call()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        # Should have both conversation and custom attributes
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_custom"
        assert spans[0].attributes["custom.environment"] == "production"
        assert spans[0].attributes["custom.version"] == "1.0.0"

    def test_metadata_and_tags_in_spans(self, tracer_setup):
        """Test that metadata and tags are properly added to spans"""
        tracer, memory_exporter = tracer_setup

        @observe(
            tags=["production", "api-v2"], metadata={"endpoint": "/chat", "category": "support"}
        )
        def tagged_call(message: str) -> str:
            return f"Reply: {message}"

        result = tagged_call("Hello")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        # Tags are stored as tuple
        assert "tags" in span.attributes
        tags = span.attributes["tags"]
        if isinstance(tags, tuple):
            assert set(tags) == {"production", "api-v2"}
        else:
            assert set(tags.split(",")) == {"production", "api-v2"}

        assert span.attributes["metadata.endpoint"] == "/chat"
        assert span.attributes["metadata.category"] == "support"
        assert (
            span.attributes["user.category"] == "support"
        )  # category also emitted as user.category


class TestE2EErrorHandling:
    """End-to-end tests for error handling and exception tracking"""

    def test_exception_tracking_in_decorator(self, tracer_setup):
        """Test that exceptions are properly tracked in spans"""
        tracer, memory_exporter = tracer_setup

        @observe()
        def failing_call() -> str:
            raise ValueError("API rate limit exceeded")

        # Exception should be raised and span should be recorded
        with conversation_context(conversation_id="conv_error"):
            with pytest.raises(ValueError, match="API rate limit exceeded"):
                failing_call()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

        span = spans[0]

        # Should have conversation context even with error
        assert span.attributes[GenAIAttributes.CONVERSATION_ID] == "conv_error"

        # Should have recorded the exception
        events = span.events
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

        # Check exception details
        exc_event = exception_events[0]
        assert "exception.type" in exc_event.attributes
        assert "ValueError" in str(exc_event.attributes["exception.type"])
        assert "API rate limit exceeded" in str(exc_event.attributes.get("exception.message", ""))


class TestE2ERealWorldScenario:
    """End-to-end test simulating a real-world customer support bot"""

    def test_customer_support_chatbot_flow(self, tracer_setup):
        """
        Simulate a complete customer support interaction:
        1. Customer asks question
        2. Search knowledge base (tool call)
        3. Check order status (tool call)
        4. Generate personalized response (LLM call)
        """
        tracer, memory_exporter = tracer_setup

        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

        # Define support bot functions
        @observe(as_type="tool", name="search_knowledge_base")
        def search_kb(query: str) -> list:
            """Search internal knowledge base"""
            return [
                {"article": "Return Policy", "relevance": 0.9},
                {"article": "Shipping Times", "relevance": 0.7},
            ]

        @observe(as_type="tool", name="check_order_status")
        def check_order(order_id: str) -> dict:
            """Check order status in database"""
            return {"order_id": order_id, "status": "shipped", "tracking": "1Z999AA10123456784"}

        @observe(pricing=pricing, name="generate_support_response", capture_input=True)
        def generate_response(context: dict, query: str) -> MockOpenAIResponse:
            """Generate personalized support response"""
            prompt = f"Context: {context}\nUser question: {query}"
            return MockOpenAIResponse(
                content="Your order has been shipped! Tracking: 1Z999AA10123456784",
                model="gpt-4o",
                prompt_tokens=len(prompt) // 4,
                completion_tokens=25,
            )

        # Simulate complete support interaction
        conversation_id = "support_conv_456"
        user_id = "customer_john"
        workflow_id = "support_ticket_789"

        with conversation_context(conversation_id=conversation_id, user_id=user_id):
            with workflow_context(workflow_id=workflow_id, workflow_type="customer_support"):
                # Customer question
                user_query = "Where is my order #12345?"

                # Step 1: Search knowledge base
                kb_results = search_kb(user_query)

                # Step 2: Check order status
                order_info = check_order("12345")

                # Step 3: Generate personalized response
                context = {"kb_results": kb_results, "order_info": order_info}
                response = generate_response(context, user_query)

        # Verify complete span hierarchy and attributes
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 3, "Should have 3 spans: KB search, order check, LLM response"

        # All spans should have full context
        for span in spans:
            # Conversation context
            assert span.attributes[GenAIAttributes.CONVERSATION_ID] == conversation_id
            assert span.attributes["user.id"] == user_id

            # Workflow context
            assert span.attributes["workflow.id"] == workflow_id
            assert span.attributes["workflow.type"] == "customer_support"

        # Verify span kinds
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == "tool"  # KB search
        assert spans[1].attributes[Last9Attributes.L9_SPAN_KIND] == "tool"  # Order check
        assert spans[2].attributes[Last9Attributes.L9_SPAN_KIND] == "llm"  # LLM response

        # Verify LLM span has full tracking
        llm_span = spans[2]
        assert llm_span.attributes[GenAIAttributes.RESPONSE_MODEL] == "gpt-4o"
        assert GenAIAttributes.USAGE_COST_USD in llm_span.attributes
        assert llm_span.attributes[GenAIAttributes.USAGE_INPUT_TOKENS] > 0
        assert llm_span.attributes[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 25

        # Verify input capture
        assert "function.arg.query" in llm_span.attributes
        assert "order" in llm_span.attributes["function.arg.query"].lower()

        print(f"\n✅ Customer support flow complete:")
        print(f"   - Conversation: {conversation_id}")
        print(f"   - User: {user_id}")
        print(f"   - Workflow: {workflow_id}")
        print(f"   - Total spans: {len(spans)}")
        print(f"   - LLM cost: ${llm_span.attributes[GenAIAttributes.USAGE_COST_USD]:.6f}")


def test_e2e_quick_sanity_check(tracer_setup):
    """Quick sanity check that everything works together"""
    tracer, memory_exporter = tracer_setup

    pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}

    @observe(pricing=pricing)
    def quick_test() -> MockOpenAIResponse:
        return MockOpenAIResponse(content="Hello!", model="gpt-4o")

    with conversation_context(conversation_id="sanity"):
        quick_test()

    spans = memory_exporter.get_finished_spans()

    # Basic assertions
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "sanity"
    assert spans[0].attributes[GenAIAttributes.RESPONSE_MODEL] == "gpt-4o"
    assert GenAIAttributes.USAGE_COST_USD in spans[0].attributes
    assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == "llm"

    print("\n✅ E2E sanity check passed!")
