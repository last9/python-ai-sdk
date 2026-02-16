#!/usr/bin/env python3
"""
Tool/Function call integration example

This example demonstrates how to track tool and function calls within AI workflows
using Last9 GenAI attributes, including database queries, API calls, and other tools.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import json
import random
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from last9_genai import ModelPricing, Last9GenAI, create_tool_span, SpanKinds


def setup_tracing():
    """Set up OpenTelemetry tracing"""
    trace.set_tracer_provider(TracerProvider())
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    return trace.get_tracer(__name__)


# Mock tool implementations
def mock_database_query(table: str, filters: dict) -> dict:
    """Mock database query tool"""
    time.sleep(random.uniform(0.1, 0.3))  # Simulate query time
    return {
        "results": [
            {"id": 1, "name": "John Doe", "status": "active"},
            {"id": 2, "name": "Jane Smith", "status": "pending"},
        ],
        "count": 2,
        "query_time_ms": 45.2,
    }


def mock_api_call(endpoint: str, params: dict) -> dict:
    """Mock external API call tool"""
    time.sleep(random.uniform(0.2, 0.5))  # Simulate API call time
    return {
        "status": 200,
        "data": {"weather": "sunny", "temperature": 72},
        "response_time_ms": 234.5,
    }


def mock_file_processor(file_path: str, operation: str) -> dict:
    """Mock file processing tool"""
    time.sleep(random.uniform(0.05, 0.15))
    return {
        "processed": True,
        "file_size_bytes": 1024 * 50,
        "lines_processed": 150,
        "operation": operation,
    }


def database_tool_example():
    """Example of database query tool tracking"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "user_lookup_workflow"

    print("🔄 Database tool example...")

    with tracer.start_span("gen_ai.tool.database_query") as span:
        start_time = time.time()

        # Tool parameters
        tool_args = {
            "table": "users",
            "filters": {"status": "active", "created_after": "2024-01-01"},
        }

        # Execute the tool
        result = mock_database_query(tool_args["table"], tool_args["filters"])

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Add Last9 tool attributes
        l9_genai.add_tool_attributes(
            span,
            tool_name="database_query",
            tool_type="datastore",
            description="Query users table with filters",
            arguments=tool_args,
            result=result,
            duration_ms=duration_ms,
            workflow_id=workflow_id,
        )

        print(f"✅ Database query completed")
        print(f"   Duration: {duration_ms:.2f}ms")
        print(f"   Results: {result['count']} records found")


def api_call_tool_example():
    """Example of external API call tool tracking"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "weather_lookup_workflow"

    print("\n🔄 API call tool example...")

    with tracer.start_span("gen_ai.tool.weather_api") as span:
        start_time = time.time()

        # Tool parameters
        tool_args = {
            "endpoint": "https://api.weather.com/v1/current",
            "params": {"location": "San Francisco, CA", "units": "fahrenheit"},
        }

        # Execute the tool
        result = mock_api_call(tool_args["endpoint"], tool_args["params"])

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Add Last9 tool attributes
        l9_genai.add_tool_attributes(
            span,
            tool_name="weather_api",
            tool_type="api",
            description="Fetch current weather data",
            arguments=tool_args,
            result=result,
            duration_ms=duration_ms,
            workflow_id=workflow_id,
        )

        # Add API-specific performance metrics
        l9_genai.add_performance_attributes(
            span, response_time_ms=duration_ms, response_size_bytes=len(json.dumps(result).encode())
        )

        print(f"✅ Weather API call completed")
        print(f"   Status: {result['status']}")
        print(f"   Duration: {duration_ms:.2f}ms")
        print(f"   Weather: {result['data']['weather']}, {result['data']['temperature']}°F")


def file_processing_tool_example():
    """Example of file processing tool tracking"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "document_processing_workflow"

    print("\n🔄 File processing tool example...")

    with tracer.start_span("gen_ai.tool.file_processor") as span:
        start_time = time.time()

        # Tool parameters
        tool_args = {"file_path": "/data/documents/report.txt", "operation": "extract_text"}

        # Execute the tool
        result = mock_file_processor(tool_args["file_path"], tool_args["operation"])

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Add Last9 tool attributes
        l9_genai.add_tool_attributes(
            span,
            tool_name="file_processor",
            tool_type="file_system",
            description="Extract text content from document",
            arguments=tool_args,
            result=result,
            duration_ms=duration_ms,
            workflow_id=workflow_id,
        )

        print(f"✅ File processing completed")
        print(f"   File size: {result['file_size_bytes']} bytes")
        print(f"   Lines processed: {result['lines_processed']}")
        print(f"   Duration: {duration_ms:.2f}ms")


def multi_tool_workflow_example():
    """Complex workflow example with multiple tools"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "customer_analysis_workflow"

    print("\n🔄 Multi-tool workflow example...")

    # Initialize workflow
    with tracer.start_span("customer_analysis_workflow") as workflow_span:
        l9_genai.add_workflow_attributes(
            workflow_span, workflow_id, workflow_type="customer_analytics", user_id="analyst_123"
        )

        # Step 1: Database lookup
        with create_tool_span(tracer, "customer_lookup", "datastore", workflow_id) as span:
            start_time = time.time()
            result1 = mock_database_query("customers", {"segment": "premium"})
            duration = (time.time() - start_time) * 1000

            l9_genai.add_tool_attributes(
                span,
                "customer_lookup",
                "datastore",
                description="Lookup premium customers",
                arguments={"table": "customers", "segment": "premium"},
                result=result1,
                duration_ms=duration,
                workflow_id=workflow_id,
            )
            print(f"   Step 1: Found {result1['count']} premium customers")

        # Step 2: External data enrichment
        with create_tool_span(tracer, "data_enrichment", "api", workflow_id) as span:
            start_time = time.time()
            result2 = mock_api_call("https://api.enrichment.com/v1/customers", {"ids": [1, 2]})
            duration = (time.time() - start_time) * 1000

            l9_genai.add_tool_attributes(
                span,
                "data_enrichment",
                "api",
                description="Enrich customer data with external sources",
                arguments={"api_endpoint": "enrichment", "customer_ids": [1, 2]},
                result=result2,
                duration_ms=duration,
                workflow_id=workflow_id,
            )
            print(f"   Step 2: Data enrichment completed with status {result2['status']}")

        # Step 3: Report generation
        with create_tool_span(tracer, "report_generator", "file_system", workflow_id) as span:
            start_time = time.time()
            result3 = mock_file_processor("/reports/customer_analysis.pdf", "generate")
            duration = (time.time() - start_time) * 1000

            l9_genai.add_tool_attributes(
                span,
                "report_generator",
                "file_system",
                description="Generate customer analysis report",
                arguments={"output_path": "/reports/customer_analysis.pdf"},
                result=result3,
                duration_ms=duration,
                workflow_id=workflow_id,
            )
            print(f"   Step 3: Report generated ({result3['file_size_bytes']} bytes)")

        # Check workflow summary
        workflow = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
        print(f"\n✅ Multi-tool workflow completed!")
        print(f"   Total tool calls: {workflow.tool_calls}")
        print(f"   Workflow cost: ${workflow.total_cost:.6f} (tools typically have no direct cost)")


def function_call_with_llm_example():
    """Example combining LLM calls with tool usage"""
    tracer = setup_tracing()
    # Add pricing for cost tracking (optional - without this, only tokens tracked)
    custom_pricing = {
        "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
        "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
        "gpt-4o": ModelPricing(input=2.50, output=10.0),
        "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
    }
    l9_genai = Last9GenAI(custom_pricing=custom_pricing)

    workflow_id = "ai_assisted_analysis"

    print("\n🔄 AI-assisted analysis workflow (LLM + Tools)...")

    # Step 1: LLM determines what data to fetch
    with tracer.start_span("gen_ai.chat.completions") as llm_span:
        l9_genai.set_span_kind(llm_span, SpanKinds.LLM)
        l9_genai.add_workflow_attributes(llm_span, workflow_id, "ai_assisted_analysis")

        # Simulate LLM cost for the planning call
        mock_usage = {"input_tokens": 100, "output_tokens": 50}
        cost = l9_genai.add_llm_cost_attributes(llm_span, "claude-3-haiku", mock_usage, workflow_id)
        print(f"   LLM planning call: ${cost.total:.6f}")

    # Step 2: Execute the data fetch based on LLM decision
    with create_tool_span(tracer, "data_fetch", "datastore", workflow_id) as tool_span:
        start_time = time.time()
        result = mock_database_query("analytics", {"date_range": "last_30_days"})
        duration = (time.time() - start_time) * 1000

        l9_genai.add_tool_attributes(
            tool_span,
            "data_fetch",
            "datastore",
            description="Fetch analytics data as determined by LLM",
            arguments={"query": "last 30 days analytics"},
            result=result,
            duration_ms=duration,
            workflow_id=workflow_id,
        )
        print(f"   Data fetch completed: {result['count']} records")

    # Step 3: LLM analyzes the fetched data
    with tracer.start_span("gen_ai.chat.completions") as llm_span:
        l9_genai.set_span_kind(llm_span, SpanKinds.LLM)

        # Simulate larger LLM call for analysis
        mock_usage = {"input_tokens": 500, "output_tokens": 300}
        cost = l9_genai.add_llm_cost_attributes(
            llm_span, "claude-3-5-sonnet", mock_usage, workflow_id
        )
        print(f"   LLM analysis call: ${cost.total:.6f}")

    # Final workflow summary
    workflow = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)
    print(f"\n✅ AI-assisted workflow completed!")
    print(f"   LLM calls: {workflow.llm_calls}")
    print(f"   Tool calls: {workflow.tool_calls}")
    print(f"   Total cost: ${workflow.total_cost:.6f}")


if __name__ == "__main__":
    print("Last9 GenAI Attributes - Tool Integration Examples")
    print("=" * 55)

    try:
        # Run all tool examples
        database_tool_example()
        api_call_tool_example()
        file_processing_tool_example()
        multi_tool_workflow_example()
        function_call_with_llm_example()

        # Force export of spans
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        print("\n✅ All tool integration examples completed!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
