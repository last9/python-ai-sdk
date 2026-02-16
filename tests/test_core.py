"""Tests for core.py - cost calculation, tracking, and Last9GenAI class"""

import pytest
import hashlib
from datetime import datetime
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from last9_genai.core import (
    calculate_llm_cost,
    detect_ai_provider,
    estimate_tokens,
    WorkflowCostTracker,
    ConversationTracker,
    Last9GenAI,
    create_llm_span,
    create_tool_span,
    ModelPricing,
    CostBreakdown,
    GenAIAttributes,
    Last9Attributes,
    SpanKinds,
    Operations,
    Providers,
    EventNames,
)


class TestCostCalculation:
    """Test calculate_llm_cost() function"""

    def test_calculate_cost_with_pricing(self):
        """Test cost calculation with custom pricing"""
        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }
        usage = {"input_tokens": 1000, "output_tokens": 500}

        cost = calculate_llm_cost("gpt-4o", usage, pricing)

        assert cost is not None
        assert cost.input == pytest.approx(0.0025)  # (1000 / 1M) * 2.50
        assert cost.output == pytest.approx(0.005)  # (500 / 1M) * 10.0
        assert cost.total == pytest.approx(0.0075)

    def test_calculate_cost_old_naming_convention(self):
        """Test cost calculation with old token naming (prompt_tokens, completion_tokens)"""
        pricing = {
            "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
        }
        usage = {"prompt_tokens": 2000, "completion_tokens": 1000}

        cost = calculate_llm_cost("gpt-3.5-turbo", usage, pricing)

        assert cost is not None
        assert cost.input == pytest.approx(0.001)  # (2000 / 1M) * 0.50
        assert cost.output == pytest.approx(0.0015)  # (1000 / 1M) * 1.50
        assert cost.total == pytest.approx(0.0025)

    def test_calculate_cost_no_pricing(self):
        """Test cost calculation without pricing returns None"""
        usage = {"input_tokens": 1000, "output_tokens": 500}

        cost = calculate_llm_cost("gpt-4o", usage, None)

        assert cost is None

    def test_calculate_cost_model_not_found(self):
        """Test cost calculation when model not in pricing dict"""
        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }
        usage = {"input_tokens": 1000, "output_tokens": 500}

        cost = calculate_llm_cost("unknown-model", usage, pricing)

        assert cost is None

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }
        usage = {"input_tokens": 0, "output_tokens": 0}

        cost = calculate_llm_cost("gpt-4o", usage, pricing)

        assert cost is not None
        assert cost.input == 0.0
        assert cost.output == 0.0
        assert cost.total == 0.0

    def test_calculate_cost_large_numbers(self):
        """Test cost calculation with large token counts"""
        pricing = {
            "gpt-4o": ModelPricing(input=2.50, output=10.0),
        }
        usage = {"input_tokens": 10_000_000, "output_tokens": 5_000_000}

        cost = calculate_llm_cost("gpt-4o", usage, pricing)

        assert cost is not None
        assert cost.input == pytest.approx(25.0)  # (10M / 1M) * 2.50
        assert cost.output == pytest.approx(50.0)  # (5M / 1M) * 10.0
        assert cost.total == pytest.approx(75.0)


class TestProviderDetection:
    """Test detect_ai_provider() function"""

    def test_detect_openai(self):
        """Test OpenAI model detection"""
        assert detect_ai_provider("gpt-4o") == Providers.OPENAI
        assert detect_ai_provider("gpt-3.5-turbo") == Providers.OPENAI
        assert detect_ai_provider("GPT-4") == Providers.OPENAI

    def test_detect_anthropic(self):
        """Test Anthropic model detection"""
        assert detect_ai_provider("claude-3-5-sonnet") == Providers.ANTHROPIC
        assert detect_ai_provider("claude-opus-4-6") == Providers.ANTHROPIC
        assert detect_ai_provider("CLAUDE-2") == Providers.ANTHROPIC

    def test_detect_google(self):
        """Test Google model detection"""
        assert detect_ai_provider("gemini-pro") == Providers.GOOGLE
        assert detect_ai_provider("gemini-1.5-flash") == Providers.GOOGLE

    def test_detect_cohere(self):
        """Test Cohere model detection"""
        assert detect_ai_provider("command-r-plus") == Providers.COHERE
        assert detect_ai_provider("command-light") == Providers.COHERE

    def test_detect_unknown_provider(self):
        """Test unknown model returns None"""
        assert detect_ai_provider("unknown-model-123") is None
        assert detect_ai_provider("llama-2") is None

    def test_detect_empty_model(self):
        """Test empty model returns None"""
        assert detect_ai_provider("") is None
        assert detect_ai_provider(None) is None


class TestTokenEstimation:
    """Test estimate_tokens() function"""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation"""
        text = "Hello, world!"
        tokens = estimate_tokens(text)
        assert tokens == len(text) // 4

    def test_estimate_tokens_long_text(self):
        """Test token estimation with longer text"""
        text = "This is a longer text with many words that should be estimated correctly"
        tokens = estimate_tokens(text)
        assert tokens == len(text) // 4

    def test_estimate_tokens_empty(self):
        """Test token estimation with empty string"""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        """Test token estimation with very short text"""
        assert estimate_tokens("Hi") == 0  # 2 // 4 = 0


class TestWorkflowCostTracker:
    """Test WorkflowCostTracker class"""

    def test_initialize_workflow(self):
        """Test workflow initialization"""
        tracker = WorkflowCostTracker()
        tracker.initialize_workflow("wf_123", {"user_id": "user_1"})

        workflow = tracker.get_workflow_cost("wf_123")
        assert workflow is not None
        assert workflow.workflow_id == "wf_123"
        assert workflow.metadata["user_id"] == "user_1"
        assert workflow.total_cost == 0.0
        assert workflow.llm_calls == 0
        assert workflow.tool_calls == 0

    def test_add_cost_llm(self):
        """Test adding LLM cost to workflow"""
        tracker = WorkflowCostTracker()
        cost = CostBreakdown(input=0.01, output=0.02, total=0.03)

        tracker.add_cost("wf_456", cost, "llm")

        workflow = tracker.get_workflow_cost("wf_456")
        assert workflow is not None
        assert workflow.total_cost == 0.03
        assert workflow.llm_calls == 1
        assert workflow.tool_calls == 0
        assert len(workflow.costs) == 1

    def test_add_cost_tool(self):
        """Test adding tool cost to workflow"""
        tracker = WorkflowCostTracker()
        cost = CostBreakdown(input=0.0, output=0.0, total=0.0)

        tracker.add_cost("wf_789", cost, "tool")

        workflow = tracker.get_workflow_cost("wf_789")
        assert workflow is not None
        assert workflow.total_cost == 0.0
        assert workflow.llm_calls == 0
        assert workflow.tool_calls == 1

    def test_add_multiple_costs(self):
        """Test adding multiple costs to same workflow"""
        tracker = WorkflowCostTracker()
        cost1 = CostBreakdown(input=0.01, output=0.02, total=0.03)
        cost2 = CostBreakdown(input=0.02, output=0.03, total=0.05)

        tracker.add_cost("wf_multi", cost1, "llm")
        tracker.add_cost("wf_multi", cost2, "llm")

        workflow = tracker.get_workflow_cost("wf_multi")
        assert workflow is not None
        assert workflow.total_cost == 0.08  # 0.03 + 0.05
        assert workflow.llm_calls == 2
        assert len(workflow.costs) == 2

    def test_get_all_workflows(self):
        """Test getting all workflows"""
        tracker = WorkflowCostTracker()
        tracker.initialize_workflow("wf_1")
        tracker.initialize_workflow("wf_2")

        all_workflows = tracker.get_all_workflows()
        assert len(all_workflows) == 2
        assert "wf_1" in all_workflows
        assert "wf_2" in all_workflows

    def test_delete_workflow(self):
        """Test deleting workflow"""
        tracker = WorkflowCostTracker()
        tracker.initialize_workflow("wf_delete")

        # Workflow exists
        assert tracker.get_workflow_cost("wf_delete") is not None

        # Delete workflow
        result = tracker.delete_workflow("wf_delete")
        assert result is True
        assert tracker.get_workflow_cost("wf_delete") is None

        # Try deleting non-existent workflow
        result = tracker.delete_workflow("wf_nonexistent")
        assert result is False


class TestConversationTracker:
    """Test ConversationTracker class"""

    def test_start_conversation(self):
        """Test starting a new conversation"""
        tracker = ConversationTracker()
        tracker.start_conversation("conv_123", user_id="user_1", metadata={"app": "chatbot"})

        conversation = tracker.get_conversation("conv_123")
        assert conversation is not None
        assert len(conversation) == 0

        # Stats returns None for empty conversation (no turns yet)
        stats = tracker.get_conversation_stats("conv_123")
        assert stats is None

    def test_add_turn(self):
        """Test adding turn to conversation"""
        tracker = ConversationTracker()
        cost = CostBreakdown(input=0.01, output=0.02, total=0.03)
        usage = {"input_tokens": 100, "output_tokens": 200}

        turn_number = tracker.add_turn(
            "conv_456",
            user_message="Hello",
            assistant_message="Hi there!",
            model="gpt-4o",
            usage=usage,
            cost=cost,
        )

        assert turn_number == 1

        conversation = tracker.get_conversation("conv_456")
        assert len(conversation) == 1
        assert conversation[0].user_message == "Hello"
        assert conversation[0].assistant_message == "Hi there!"
        assert conversation[0].model == "gpt-4o"

    def test_add_multiple_turns(self):
        """Test adding multiple turns"""
        tracker = ConversationTracker()
        cost = CostBreakdown(input=0.01, output=0.02, total=0.03)
        usage = {"input_tokens": 100, "output_tokens": 200}

        turn1 = tracker.add_turn("conv_multi", "Hello", "Hi!", "gpt-4o", usage, cost)
        turn2 = tracker.add_turn("conv_multi", "How are you?", "I'm good!", "gpt-4o", usage, cost)

        assert turn1 == 1
        assert turn2 == 2

        conversation = tracker.get_conversation("conv_multi")
        assert len(conversation) == 2

    def test_get_conversation_cost(self):
        """Test getting total conversation cost"""
        tracker = ConversationTracker()
        cost1 = CostBreakdown(input=0.01, output=0.02, total=0.03)
        cost2 = CostBreakdown(input=0.02, output=0.03, total=0.05)
        usage = {"input_tokens": 100, "output_tokens": 200}

        tracker.add_turn("conv_cost", "Msg1", "Reply1", "gpt-4o", usage, cost1)
        tracker.add_turn("conv_cost", "Msg2", "Reply2", "gpt-4o", usage, cost2)

        total_cost = tracker.get_conversation_cost("conv_cost")
        assert total_cost == 0.08  # 0.03 + 0.05

    def test_get_conversation_stats(self):
        """Test getting conversation statistics"""
        tracker = ConversationTracker()
        tracker.start_conversation("conv_stats", user_id="user_1")

        cost = CostBreakdown(input=0.01, output=0.02, total=0.03)
        usage = {"input_tokens": 100, "output_tokens": 200}

        tracker.add_turn("conv_stats", "Hello", "Hi!", "gpt-4o", usage, cost)
        tracker.add_turn("conv_stats", "How are you?", "Good!", "gpt-3.5-turbo", usage, cost)

        stats = tracker.get_conversation_stats("conv_stats")
        assert stats is not None
        assert stats["turn_count"] == 2
        assert stats["total_cost"] == 0.06
        assert stats["total_input_tokens"] == 200
        assert stats["total_output_tokens"] == 400
        assert set(stats["models_used"]) == {"gpt-4o", "gpt-3.5-turbo"}
        assert stats["user_id"] == "user_1"

    def test_get_conversation_nonexistent(self):
        """Test getting non-existent conversation"""
        tracker = ConversationTracker()
        conversation = tracker.get_conversation("nonexistent")
        assert conversation is None

        stats = tracker.get_conversation_stats("nonexistent")
        assert stats is None


class TestLast9GenAI:
    """Test Last9GenAI main class"""

    def test_init_default(self):
        """Test Last9GenAI initialization with defaults"""
        l9 = Last9GenAI()
        assert l9.model_pricing is None
        assert l9.enable_cost_tracking is True
        assert l9.workflow_tracker is not None

    def test_init_with_pricing(self):
        """Test initialization with custom pricing"""
        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}
        l9 = Last9GenAI(custom_pricing=pricing)
        assert l9.model_pricing == pricing

    def test_set_span_kind(self, tracer_setup):
        """Test setting span kind"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.set_span_kind(span, SpanKinds.LLM)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == SpanKinds.LLM

    def test_set_span_kind_invalid(self, tracer_setup):
        """Test setting invalid span kind logs warning"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.set_span_kind(span, "invalid_kind")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        # Invalid kind should not be set
        assert Last9Attributes.L9_SPAN_KIND not in spans[0].attributes

    def test_add_llm_cost_attributes(self, tracer_setup):
        """Test adding LLM cost attributes"""
        tracer, memory_exporter = tracer_setup
        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}
        l9 = Last9GenAI(custom_pricing=pricing)

        with tracer.start_as_current_span("test") as span:
            usage = {"input_tokens": 1000, "output_tokens": 500}
            cost = l9.add_llm_cost_attributes(span, "gpt-4o", usage)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.USAGE_COST_USD] == pytest.approx(0.0075)
        assert spans[0].attributes[GenAIAttributes.USAGE_COST_INPUT_USD] == pytest.approx(0.0025)
        assert spans[0].attributes[GenAIAttributes.USAGE_COST_OUTPUT_USD] == pytest.approx(0.005)
        assert cost is not None
        assert cost.total == pytest.approx(0.0075)

    def test_add_llm_cost_attributes_no_pricing(self, tracer_setup):
        """Test adding LLM cost attributes without pricing"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()  # No pricing

        with tracer.start_as_current_span("test") as span:
            usage = {"input_tokens": 1000, "output_tokens": 500}
            cost = l9.add_llm_cost_attributes(span, "gpt-4o", usage)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        # No cost attributes should be set
        assert GenAIAttributes.USAGE_COST_USD not in spans[0].attributes
        assert cost is None

    def test_add_llm_cost_attributes_with_workflow(self, tracer_setup):
        """Test adding LLM cost with workflow tracking"""
        tracer, memory_exporter = tracer_setup
        pricing = {"gpt-4o": ModelPricing(input=2.50, output=10.0)}
        tracker = WorkflowCostTracker()
        l9 = Last9GenAI(custom_pricing=pricing, workflow_tracker=tracker)

        with tracer.start_as_current_span("test") as span:
            usage = {"input_tokens": 1000, "output_tokens": 500}
            l9.add_llm_cost_attributes(span, "gpt-4o", usage, workflow_id="wf_123")

        # Check workflow was updated
        workflow = tracker.get_workflow_cost("wf_123")
        assert workflow is not None
        assert workflow.total_cost == pytest.approx(0.0075)
        assert workflow.llm_calls == 1

    def test_add_workflow_attributes(self, tracer_setup):
        """Test adding workflow attributes"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_workflow_attributes(
                span,
                workflow_id="wf_456",
                workflow_type="rag",
                user_id="user_1",
                session_id="session_1",
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[Last9Attributes.WORKFLOW_ID] == "wf_456"
        assert spans[0].attributes[Last9Attributes.WORKFLOW_TYPE] == "rag"
        assert spans[0].attributes[Last9Attributes.WORKFLOW_USER_ID] == "user_1"
        assert spans[0].attributes[Last9Attributes.WORKFLOW_SESSION_ID] == "session_1"

    def test_add_prompt_versioning(self, tracer_setup):
        """Test adding prompt versioning"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        prompt_template = "You are a helpful assistant. Question: {question}"

        with tracer.start_as_current_span("test") as span:
            prompt_hash = l9.add_prompt_versioning(
                span, prompt_template, template_id="assistant_v1", version="1.0.0"
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.PROMPT_TEMPLATE] == prompt_template
        assert spans[0].attributes[GenAIAttributes.PROMPT_HASH] == prompt_hash
        assert spans[0].attributes[GenAIAttributes.PROMPT_TEMPLATE_ID] == "assistant_v1"
        assert spans[0].attributes[GenAIAttributes.PROMPT_VERSION] == "1.0.0"

        # Verify hash is correct
        expected_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
        assert prompt_hash == expected_hash

    def test_add_tool_attributes(self, tracer_setup):
        """Test adding tool attributes"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_tool_attributes(
                span,
                tool_name="database_query",
                tool_type="datastore",
                description="Query user data",
                arguments={"table": "users", "id": 123},
                result="Found 1 record",
                duration_ms=45.2,
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.TOOL_NAME] == "database_query"
        assert spans[0].attributes[GenAIAttributes.TOOL_TYPE] == "datastore"
        assert spans[0].attributes[GenAIAttributes.TOOL_DESCRIPTION] == "Query user data"
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == SpanKinds.TOOL
        assert '"table": "users"' in spans[0].attributes[Last9Attributes.FUNCTION_CALL_ARGUMENTS]
        assert spans[0].attributes[Last9Attributes.FUNCTION_CALL_RESULT] == "Found 1 record"
        assert spans[0].attributes[Last9Attributes.FUNCTION_CALL_DURATION_MS] == 45.2

    def test_add_performance_attributes(self, tracer_setup):
        """Test adding performance attributes"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_performance_attributes(
                span,
                response_time_ms=125.5,
                request_size_bytes=1024,
                response_size_bytes=2048,
                quality_score=0.95,
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[Last9Attributes.RESPONSE_TIME_MS] == 125.5
        assert spans[0].attributes[Last9Attributes.REQUEST_SIZE_BYTES] == 1024
        assert spans[0].attributes[Last9Attributes.RESPONSE_SIZE_BYTES] == 2048
        assert spans[0].attributes[Last9Attributes.QUALITY_SCORE] == 0.95

    def test_add_standard_llm_attributes(self, tracer_setup):
        """Test adding standard OpenTelemetry GenAI attributes"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_standard_llm_attributes(
                span,
                model="gpt-4o",
                operation=Operations.CHAT_COMPLETIONS,
                conversation_id="conv_123",
                request_params={"max_tokens": 1000, "temperature": 0.7},
                response_data={"id": "resp_456", "model": "gpt-4o", "finish_reason": "stop"},
                usage={"input_tokens": 100, "output_tokens": 200},
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4o"
        assert spans[0].attributes[GenAIAttributes.OPERATION_NAME] == Operations.CHAT_COMPLETIONS
        assert spans[0].attributes[GenAIAttributes.PROVIDER_NAME] == Providers.OPENAI
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_123"
        assert spans[0].attributes[GenAIAttributes.REQUEST_MAX_TOKENS] == 1000
        assert spans[0].attributes[GenAIAttributes.REQUEST_TEMPERATURE] == 0.7
        assert spans[0].attributes[GenAIAttributes.RESPONSE_ID] == "resp_456"
        assert spans[0].attributes[GenAIAttributes.RESPONSE_MODEL] == "gpt-4o"
        assert spans[0].attributes[GenAIAttributes.USAGE_INPUT_TOKENS] == 100
        assert spans[0].attributes[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 200
        assert spans[0].attributes[GenAIAttributes.USAGE_TOTAL_TOKENS] == 300

    def test_add_conversation_tracking(self, tracer_setup):
        """Test adding conversation tracking attributes"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_conversation_tracking(
                span,
                conversation_id="conv_789",
                user_id="user_1",
                session_id="session_1",
                turn_number=5,
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_789"
        assert spans[0].attributes[Last9Attributes.WORKFLOW_USER_ID] == "user_1"
        assert spans[0].attributes[Last9Attributes.WORKFLOW_SESSION_ID] == "session_1"
        assert spans[0].attributes["gen_ai.conversation.turn_number"] == 5

    def test_add_content_events(self, tracer_setup):
        """Test adding content events for prompt and completion"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_content_events(
                span, prompt="What is the capital of France?", completion="The capital is Paris."
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        events = spans[0].events
        assert len(events) == 2

        # Check prompt event
        prompt_event = events[0]
        assert prompt_event.name == EventNames.GEN_AI_CONTENT_PROMPT
        assert prompt_event.attributes[GenAIAttributes.PROMPT] == "What is the capital of France?"
        assert prompt_event.attributes["gen_ai.prompt.length"] == 30
        assert prompt_event.attributes["gen_ai.prompt.truncated"] is False

        # Check completion event
        completion_event = events[1]
        assert completion_event.name == EventNames.GEN_AI_CONTENT_COMPLETION
        assert completion_event.attributes[GenAIAttributes.COMPLETION] == "The capital is Paris."
        assert completion_event.attributes["gen_ai.completion.length"] == 21
        assert completion_event.attributes["gen_ai.completion.truncated"] is False

    def test_add_content_events_truncation(self, tracer_setup):
        """Test content events with truncation"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        long_text = "x" * 2000

        with tracer.start_as_current_span("test") as span:
            l9.add_content_events(span, prompt=long_text, truncate_length=100)

        spans = memory_exporter.get_finished_spans()
        events = spans[0].events
        prompt_event = events[0]

        assert prompt_event.attributes["gen_ai.prompt.length"] == 2000
        assert prompt_event.attributes["gen_ai.prompt.truncated"] is True
        assert len(prompt_event.attributes[GenAIAttributes.PROMPT]) == 103  # 100 + "..."

    def test_add_tool_call_events(self, tracer_setup):
        """Test adding tool call events"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        with tracer.start_as_current_span("test") as span:
            l9.add_tool_call_events(
                span,
                tool_name="search",
                tool_arguments={"query": "python tutorial"},
                tool_result="Found 10 results",
            )

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1

        events = spans[0].events
        assert len(events) == 2

        # Check tool call event
        call_event = events[0]
        assert call_event.name == EventNames.GEN_AI_TOOL_CALL
        assert call_event.attributes[GenAIAttributes.TOOL_NAME] == "search"
        assert (
            '"query": "python tutorial"'
            in call_event.attributes[Last9Attributes.FUNCTION_CALL_ARGUMENTS]
        )

        # Check tool result event
        result_event = events[1]
        assert result_event.name == EventNames.GEN_AI_TOOL_RESULT
        assert result_event.attributes[GenAIAttributes.TOOL_NAME] == "search"
        assert result_event.attributes[Last9Attributes.FUNCTION_CALL_RESULT] == "Found 10 results"

    def test_create_conversation_span(self, tracer_setup):
        """Test creating conversation span"""
        tracer, memory_exporter = tracer_setup
        l9 = Last9GenAI()

        span = l9.create_conversation_span(
            tracer, conversation_id="conv_test", model="gpt-4o", user_id="user_1", turn_number=3
        )
        span.end()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4o"
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_test"
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == SpanKinds.LLM
        assert spans[0].attributes[Last9Attributes.WORKFLOW_USER_ID] == "user_1"
        assert spans[0].attributes["gen_ai.conversation.turn_number"] == 3


class TestCreateLLMSpan:
    """Test create_llm_span() convenience function"""

    def test_create_llm_span_basic(self, tracer_setup):
        """Test creating basic LLM span"""
        tracer, memory_exporter = tracer_setup

        span = create_llm_span(tracer, "test_llm", "gpt-4o")
        span.end()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4o"
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == SpanKinds.LLM

    def test_create_llm_span_with_workflow(self, tracer_setup):
        """Test creating LLM span with workflow"""
        tracer, memory_exporter = tracer_setup

        span = create_llm_span(
            tracer,
            "test_llm",
            "gpt-4o",
            workflow_id="wf_test",
            conversation_id="conv_test",
        )
        span.end()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[Last9Attributes.WORKFLOW_ID] == "wf_test"
        assert spans[0].attributes[GenAIAttributes.CONVERSATION_ID] == "conv_test"


class TestCreateToolSpan:
    """Test create_tool_span() convenience function"""

    def test_create_tool_span_basic(self, tracer_setup):
        """Test creating basic tool span"""
        tracer, memory_exporter = tracer_setup

        span = create_tool_span(tracer, "database_query")
        span.end()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.TOOL_NAME] == "database_query"
        assert spans[0].attributes[Last9Attributes.L9_SPAN_KIND] == SpanKinds.TOOL

    def test_create_tool_span_with_workflow(self, tracer_setup):
        """Test creating tool span with workflow"""
        tracer, memory_exporter = tracer_setup

        span = create_tool_span(tracer, "api_call", tool_type="external_api", workflow_id="wf_tool")
        span.end()

        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[GenAIAttributes.TOOL_NAME] == "api_call"
        assert spans[0].attributes[GenAIAttributes.TOOL_TYPE] == "external_api"
