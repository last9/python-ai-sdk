#!/usr/bin/env python3
"""
FastAPI Integration Example with Last9 GenAI Attributes

This example shows how to integrate Last9 observability into a FastAPI application
that uses LLMs, with automatic request tracing and cost tracking.

Install dependencies:
    pip install fastapi uvicorn openai anthropic opentelemetry-api opentelemetry-sdk \
                opentelemetry-instrumentation-fastapi last9-genai

Run:
    python examples/fastapi_app.py

    # Or with uvicorn directly:
    uvicorn examples.fastapi_app:app --reload
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from typing import Optional, List
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Last9 imports
from last9_genai import ModelPricing, Last9GenAI, SpanKinds

# Optional: LLM client imports
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ============================================================================
# Setup OpenTelemetry and FastAPI
# ============================================================================


def setup_telemetry():
    """Set up OpenTelemetry tracing with FastAPI instrumentation"""
    trace.set_tracer_provider(TracerProvider())

    # Check if OTLP endpoint is configured (for Last9 or other OTLP backends)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otlp_endpoint:
        # Use OTLP exporter for production (Last9, Datadog, New Relic, etc.)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            print(f"🚀 Exporting traces to: {otlp_endpoint}")

            # Parse headers from environment
            # Note: gRPC metadata keys must be lowercase
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
            headers = {}
            if headers_str:
                for header in headers_str.split(","):
                    if "=" in header:
                        key, value = header.split("=", 1)
                        headers[key.strip().lower()] = value.strip()

            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, headers=headers)
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            print("   ✅ OTLP exporter configured")

        except ImportError:
            print(
                "   ⚠️  OTLP exporter not available. Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
            )
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
    else:
        # Use console exporter for local testing/demo
        print("📝 Using ConsoleSpanExporter (set OTEL_EXPORTER_OTLP_ENDPOINT for production)")
        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

    return trace.get_tracer(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Last9 GenAI Demo API",
    description="Demo API showing Last9 observability with LLM endpoints",
    version="1.0.0",
)

# Setup telemetry
tracer = setup_telemetry()

# Add pricing for cost tracking (optional - without this, only tokens tracked)
custom_pricing = {
    "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
    "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
    "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
}
l9_genai = Last9GenAI(custom_pricing=custom_pricing)


# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# ============================================================================
# Models
# ============================================================================


class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    cost_usd: float
    tokens_used: dict
    conversation_id: str


class SummarizeRequest(BaseModel):
    text: str
    model: str = "claude-3-haiku"


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    cost_usd: float
    original_length: int
    summary_length: int


# ============================================================================
# Helper Functions
# ============================================================================


def call_openai(message: str, model: str, conversation_id: str) -> tuple:
    """Call OpenAI with observability"""
    with tracer.start_span("gen_ai.openai.chat") as span:
        start_time = time.time()

        l9_genai.set_span_kind(span, SpanKinds.LLM)
        l9_genai.add_standard_llm_attributes(span, model, conversation_id=conversation_id)

        if OPENAI_AVAILABLE:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-key"))

            try:
                response = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": message}], max_tokens=500
                )

                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000

                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                content = response.choices[0].message.content

                # Add Last9 cost tracking
                cost = l9_genai.add_llm_cost_attributes(span, model, usage, conversation_id)

                # Add performance metrics
                l9_genai.add_performance_attributes(
                    span,
                    response_time_ms=response_time_ms,
                    request_size_bytes=len(message.encode()),
                    response_size_bytes=len(content.encode()),
                )

                return content, cost, usage

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        else:
            # Mock response when OpenAI is not available
            mock_content = f"Mock response to: {message[:50]}..."
            mock_usage = {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}
            cost = l9_genai.add_llm_cost_attributes(span, model, mock_usage, conversation_id)
            return mock_content, cost, mock_usage


def call_anthropic(text: str, model: str) -> tuple:
    """Call Anthropic with observability"""
    with tracer.start_span("gen_ai.anthropic.messages") as span:
        start_time = time.time()

        l9_genai.set_span_kind(span, SpanKinds.LLM)
        l9_genai.add_standard_llm_attributes(span, model)

        if ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "your-key"))

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=300,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Summarize the following text concisely:\n\n{text}",
                        }
                    ],
                )

                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

                summary = response.content[0].text

                # Add Last9 cost tracking
                cost = l9_genai.add_llm_cost_attributes(span, model, usage)

                # Add performance metrics
                l9_genai.add_performance_attributes(span, response_time_ms=response_time_ms)

                return summary, cost, usage

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise HTTPException(status_code=500, detail=f"Anthropic API error: {str(e)}")
        else:
            # Mock response when Anthropic is not available
            mock_summary = f"Mock summary of text: {text[:50]}..."
            mock_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            cost = l9_genai.add_llm_cost_attributes(span, model, mock_usage)
            return mock_summary, cost, mock_usage


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Last9 GenAI Demo API",
        "version": "1.0.0",
        "endpoints": [
            "/chat - Chat with AI (OpenAI)",
            "/summarize - Summarize text (Anthropic)",
            "/health - Health check",
        ],
        "observability": "Last9 + OpenTelemetry",
        "openai_available": OPENAI_AVAILABLE,
        "anthropic_available": ANTHROPIC_AVAILABLE,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "openai": OPENAI_AVAILABLE, "anthropic": ANTHROPIC_AVAILABLE}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with OpenAI integration and cost tracking

    This endpoint demonstrates:
    - OpenTelemetry tracing
    - Last9 cost tracking
    - Conversation ID tracking
    - Performance metrics
    """
    with tracer.start_span("api.chat") as span:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"

        # Add API-level attributes
        span.set_attribute("api.endpoint", "/chat")
        span.set_attribute("api.user_id", request.user_id or "anonymous")

        # Call LLM with observability
        content, cost, usage = call_openai(request.message, request.model, conversation_id)

        # Return response with cost information
        return ChatResponse(
            response=content,
            model=request.model,
            cost_usd=cost.total,
            tokens_used=usage,
            conversation_id=conversation_id,
        )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarization endpoint with Anthropic integration

    This endpoint demonstrates:
    - Multi-model support (Anthropic)
    - Cost tracking for different operations
    - Performance comparison
    """
    with tracer.start_span("api.summarize") as span:
        span.set_attribute("api.endpoint", "/summarize")
        span.set_attribute("text.original_length", len(request.text))

        # Call Anthropic for summarization
        summary, cost, usage = call_anthropic(request.text, request.model)

        span.set_attribute("text.summary_length", len(summary))

        return SummarizeResponse(
            summary=summary,
            model=request.model,
            cost_usd=cost.total,
            original_length=len(request.text),
            summary_length=len(summary),
        )


@app.post("/workflow/customer-support")
async def customer_support_workflow(request: ChatRequest):
    """
    Complete customer support workflow with multiple LLM calls

    This endpoint demonstrates:
    - Workflow cost aggregation
    - Multi-step AI operations
    - Combined cost tracking
    """
    workflow_id = f"support_{int(time.time())}"

    with tracer.start_span("workflow.customer_support") as workflow_span:
        l9_genai.add_workflow_attributes(
            workflow_span,
            workflow_id=workflow_id,
            workflow_type="customer_support",
            user_id=request.user_id or "anonymous",
        )

        # Step 1: Classify the query
        classify_prompt = f"Classify this support query in one word (billing/technical/general): {request.message}"
        classification, cost1, _ = call_openai(classify_prompt, "gpt-3.5-turbo", workflow_id)

        # Step 2: Generate response based on classification
        response_prompt = (
            f"Provide a helpful response to this {classification.strip()} query: {request.message}"
        )
        response, cost2, usage = call_openai(response_prompt, request.model, workflow_id)

        # Get total workflow cost
        workflow = l9_genai.workflow_tracker.get_workflow_cost(workflow_id)

        return {
            "response": response,
            "classification": classification.strip(),
            "workflow_id": workflow_id,
            "total_cost_usd": workflow.total_cost if workflow else cost1.total + cost2.total,
            "llm_calls": workflow.llm_calls if workflow else 2,
            "tokens_used": usage,
        }


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting Last9 GenAI Demo API...")
    print(f"   OpenAI available: {OPENAI_AVAILABLE}")
    print(f"   Anthropic available: {ANTHROPIC_AVAILABLE}")
    print("   OpenTelemetry tracing: ✅")
    print("   Last9 cost tracking: ✅")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Flush remaining spans
    trace.get_tracer_provider().force_flush(timeout_millis=5000)
    print("👋 Shutting down Last9 GenAI Demo API...")


# ============================================================================
# Run the app
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Last9 GenAI Attributes - FastAPI Integration Example")
    print("=" * 60)
    print("\n📝 API Endpoints:")
    print("   GET  /          - API information")
    print("   GET  /health    - Health check")
    print("   POST /chat      - Chat with AI")
    print("   POST /summarize - Summarize text")
    print("   POST /workflow/customer-support - Full workflow demo")
    print("\n🔧 Example requests:")
    print('   curl -X POST "http://localhost:8000/chat" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"message": "Hello!", "model": "gpt-3.5-turbo"}\'')
    print("\n🌐 Starting server on http://localhost:8000")
    print("   Docs available at: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
