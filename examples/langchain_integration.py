#!/usr/bin/env python3
"""
Integration example with LangChain

This example shows how to integrate Last9 GenAI attributes with LangChain
for comprehensive observability and cost tracking in LangChain applications.

Install dependencies:
    pip install langchain langchain-openai langchain-anthropic opentelemetry-api opentelemetry-sdk last9-genai
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import ModelPricing, Last9GenAI, SpanKinds

# Check for LangChain availability
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed. Install with: pip install langchain langchain-openai")


def setup_tracing():
    """Set up OpenTelemetry tracing"""
    trace.set_tracer_provider(TracerProvider())
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    return trace.get_tracer(__name__)


def langchain_chain_with_observability():
    """Example of LangChain chain with Last9 observability"""
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available, skipping example")
        return

    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "langchain_qa_workflow"

    print("🔄 Running LangChain chain with Last9 observability...")

    # Define a simple LangChain prompt template
    template = """You are a helpful AI assistant for a tech company.

    Customer Question: {question}
    Previous Context: {context}

    Please provide a clear and helpful answer."""

    prompt = PromptTemplate(input_variables=["question", "context"], template=template)

    # Create LLM (with your API key from environment)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    )

    # Wrap LangChain execution in our observability span
    with tracer.start_span("langchain.chain.execution") as span:
        start_time = time.time()

        l9_genai.set_span_kind(span, SpanKinds.LLM)
        l9_genai.add_workflow_attributes(
            span, workflow_id, workflow_type="langchain_qa", user_id="customer_123"
        )

        # Add prompt versioning
        l9_genai.add_prompt_versioning(
            span, template, template_id="customer_support_template", version="1.0.0"
        )

        try:
            # Run LangChain chain
            chain = LLMChain(llm=llm, prompt=prompt)

            result = chain.run(
                question="How do I reset my password?",
                context="User previously logged in successfully yesterday",
            )

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            # Estimate token usage (LangChain doesn't always provide this directly)
            # In production, you'd extract this from LangChain callbacks
            estimated_input = len(template) + 100  # rough estimation
            estimated_output = len(result)

            usage = {
                "input_tokens": estimated_input // 4,  # ~4 chars per token
                "output_tokens": estimated_output // 4,
            }

            # Add Last9 cost tracking
            cost = l9_genai.add_llm_cost_attributes(span, "gpt-3.5-turbo", usage, workflow_id)

            # Add performance metrics
            l9_genai.add_performance_attributes(span, response_time_ms=response_time_ms)

            print(f"✅ LangChain execution completed")
            print(f"   Response: {result[:100]}...")
            print(f"   Cost: ${cost.total:.6f} USD")
            print(f"   Response time: {response_time_ms:.2f}ms")

        except Exception as e:
            print(f"❌ Error in LangChain execution: {e}")
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))


def langchain_with_callbacks():
    """Advanced example using LangChain callbacks for precise tracking"""
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available, skipping example")
        return

    print("\n🔄 LangChain with custom callbacks for precise tracking...")

    # In a production setup, you'd create a custom LangChain callback
    # that captures token usage and integrates with Last9 attributes

    print("💡 In production, implement a custom LangChain callback handler:")
    print("""
    from langchain.callbacks.base import BaseCallbackHandler

    class Last9CallbackHandler(BaseCallbackHandler):
        def __init__(self, l9_genai, workflow_id):
            self.l9_genai = l9_genai
            self.workflow_id = workflow_id
            self.tracer = trace.get_tracer(__name__)

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.span = self.tracer.start_span("langchain.llm.call")
            self.l9_genai.set_span_kind(self.span, SpanKinds.LLM)

        def on_llm_end(self, response, **kwargs):
            # Extract real token usage from response
            if hasattr(response, 'llm_output'):
                usage = response.llm_output.get('token_usage', {})
                cost = self.l9_genai.add_llm_cost_attributes(
                    self.span,
                    response.llm_output.get('model_name'),
                    usage,
                    self.workflow_id
                )
            self.span.end()

        def on_llm_error(self, error, **kwargs):
            if self.span:
                self.span.record_exception(error)
                self.span.end()
    """)


def multi_chain_workflow():
    """Example of multi-step LangChain workflow with cost aggregation"""
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available, skipping example")
        return

    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "multi_chain_workflow"

    print("\n🔄 Multi-chain LangChain workflow...")

    with tracer.start_span("langchain.multi_chain_workflow") as workflow_span:
        l9_genai.add_workflow_attributes(
            workflow_span, workflow_id, workflow_type="document_analysis", user_id="analyst_456"
        )

        # Step 1: Summarization chain
        with tracer.start_span("chain.summarize") as span:
            l9_genai.set_span_kind(span, SpanKinds.LLM)

            # Simulate summarization
            mock_usage = {"input_tokens": 500, "output_tokens": 100}
            cost1 = l9_genai.add_llm_cost_attributes(span, "gpt-3.5-turbo", mock_usage, workflow_id)
            print(f"   Step 1 (Summarize): ${cost1.total:.6f}")

        # Step 2: Analysis chain
        with tracer.start_span("chain.analyze") as span:
            l9_genai.set_span_kind(span, SpanKinds.LLM)

            mock_usage = {"input_tokens": 300, "output_tokens": 200}
            cost2 = l9_genai.add_llm_cost_attributes(span, "gpt-4o", mock_usage, workflow_id)
            print(f"   Step 2 (Analyze): ${cost2.total:.6f}")

        # Step 3: Final report generation
        with tracer.start_span("chain.generate_report") as span:
            l9_genai.set_span_kind(span, SpanKinds.LLM)

            mock_usage = {"input_tokens": 400, "output_tokens": 300}
            cost3 = l9_genai.add_llm_cost_attributes(span, "gpt-3.5-turbo", mock_usage, workflow_id)
            print(f"   Step 3 (Report): ${cost3.total:.6f}")

    # Check total workflow cost
    workflow = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
    if workflow:
        print(f"\n💰 Multi-Chain Workflow Summary:")
        print(f"   Total cost: ${workflow.total_cost:.6f} USD")
        print(f"   Total LLM calls: {workflow.llm_calls}")


def langchain_agents_example():
    """Example with LangChain agents and tools"""
    print("\n🔄 LangChain Agents with Tools...")

    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "agent_workflow"

    # Simulate agent workflow with tools
    with tracer.start_span("langchain.agent.execution") as agent_span:
        l9_genai.add_workflow_attributes(
            agent_span, workflow_id, workflow_type="agent_task", user_id="user_789"
        )

        # Agent reasoning step
        with tracer.start_span("agent.reasoning") as span:
            l9_genai.set_span_kind(span, SpanKinds.LLM)

            mock_usage = {"input_tokens": 200, "output_tokens": 50}
            cost = l9_genai.add_llm_cost_attributes(span, "gpt-4o", mock_usage, workflow_id)
            print(f"   Agent reasoning: ${cost.total:.6f}")

        # Tool execution
        with tracer.start_span("agent.tool.search") as span:
            l9_genai.set_span_kind(span, SpanKinds.TOOL)

            l9_genai.add_tool_attributes(
                span,
                tool_name="web_search",
                tool_type="search",
                description="Search the web for information",
                arguments={"query": "latest AI developments"},
                result={"results": ["Article 1", "Article 2"]},
                duration_ms=250.5,
                workflow_id=workflow_id,
            )
            print(f"   Tool execution: web_search")

        # Final response generation
        with tracer.start_span("agent.final_response") as span:
            l9_genai.set_span_kind(span, SpanKinds.LLM)

            mock_usage = {"input_tokens": 300, "output_tokens": 150}
            cost = l9_genai.add_llm_cost_attributes(span, "gpt-4o", mock_usage, workflow_id)
            print(f"   Final response: ${cost.total:.6f}")

    # Check workflow summary
    workflow = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
    if workflow:
        print(f"\n💰 Agent Workflow Summary:")
        print(f"   Total cost: ${workflow.total_cost:.6f} USD")
        print(f"   LLM calls: {workflow.llm_calls}")
        print(f"   Tool calls: {workflow.tool_calls}")


if __name__ == "__main__":
    print("Last9 GenAI Attributes - LangChain Integration Example")
    print("=" * 60)

    if not LANGCHAIN_AVAILABLE:
        print("⚠️  LangChain not installed")
        print("   Install with: pip install langchain langchain-openai")
        print("   Set OPENAI_API_KEY environment variable\n")

    try:
        langchain_chain_with_observability()
        langchain_with_callbacks()
        multi_chain_workflow()
        langchain_agents_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n✅ LangChain integration examples completed!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
