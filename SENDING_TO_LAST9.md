# Sending Telemetry Data to Last9

This guide shows how to verify that your GenAI telemetry data is actually being sent to Last9.

## Quick Start

### 1. Install with OTLP Support

```bash
pip install last9-genai[otlp]
```

### 2. Configure Last9 Endpoint

Set environment variables for your Last9 OTLP endpoint:

```bash
export LAST9_OTLP_ENDPOINT="https://otlp.last9.io"  # Your Last9 OTLP endpoint
export LAST9_API_KEY="your-api-key"                 # Your Last9 API key
```

**Note:** Contact Last9 support for your specific OTLP endpoint URL and authentication method.

### 3. Setup OpenTelemetry with OTLP Exporter

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from last9_genai import Last9SpanProcessor, conversation_context, observe, ModelPricing

# Create resource with service identification
resource = Resource.create({
    SERVICE_NAME: "my-ai-app",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

# Create tracer provider
provider = TracerProvider(resource=resource)

# Configure OTLP exporter with Last9 endpoint
otlp_exporter = OTLPSpanExporter(
    endpoint="https://otlp.last9.io",  # Your Last9 endpoint
    headers=(("authorization", "Basic your-api-key"),),  # Authentication
)

# Add Last9SpanProcessor for automatic context enrichment
provider.add_span_processor(Last9SpanProcessor())

# Add BatchSpanProcessor for efficient export
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Set as global provider
trace.set_tracer_provider(provider)
```

### 4. Use the SDK Normally

```python
# Track a conversation
with conversation_context(conversation_id="user_123_session", user_id="user_123"):
    @observe(pricing={"gpt-4o": ModelPricing(input=2.50, output=10.0)})
    def chat(prompt: str):
        # Your LLM API call here
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response

    response = chat("Hello, how are you?")
```

### 5. Verify Data in Last9

After running your application, check Last9 dashboard:

1. **Filter by Service Name**: Look for `my-ai-app` (or your configured service name)

2. **Search for GenAI Attributes**:
   - `gen_ai.conversation.id`: Conversation tracking
   - `gen_ai.usage.cost_usd`: Cost tracking
   - `gen_ai.l9.span.kind`: Span classification (llm, tool, etc.)
   - `workflow.id`: Workflow tracking

3. **Expected Span Attributes**:
   ```json
   {
     "service.name": "my-ai-app",
     "gen_ai.conversation.id": "user_123_session",
     "gen_ai.response.model": "gpt-4o",
     "gen_ai.usage.input_tokens": 15,
     "gen_ai.usage.output_tokens": 25,
     "gen_ai.usage.cost_usd": 0.000288,
     "gen_ai.l9.span.kind": "llm",
     "user.id": "user_123"
   }
   ```

## Complete Example

See `examples/send_to_last9.py` for a complete working example:

```bash
# Set credentials
export LAST9_OTLP_ENDPOINT="https://otlp.last9.io"
export LAST9_API_KEY="your-api-key"

# Run example
python examples/send_to_last9.py
```

The example demonstrates:
- ✅ Multi-turn conversation tracking
- ✅ RAG workflow with tool calls
- ✅ Nested conversation + workflow contexts
- ✅ Cost tracking and aggregation
- ✅ Automatic context propagation

## Troubleshooting

### Data Not Appearing in Last9?

**1. Check OTLP Exporter Installation**
```bash
pip show opentelemetry-exporter-otlp-proto-grpc
```

**2. Enable Debug Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**3. Force Flush Before Exit**
```python
# At end of your application
trace.get_tracer_provider().force_flush(timeout_millis=10000)
```

**4. Verify Endpoint and Authentication**
- Confirm OTLP endpoint URL with Last9 support
- Check authentication method (Basic, Bearer, API key header)
- Test with `curl` or `grpcurl` if possible

**5. Check Network Connectivity**
```bash
# Test if endpoint is reachable
curl -v https://otlp.last9.io
```

### Common Issues

**Issue: "Connection refused" or "timeout"**
- Firewall blocking outbound gRPC connections (port 443 or 4317)
- Incorrect endpoint URL
- Network proxy not configured

**Issue: "Unauthorized" or "Forbidden"**
- Incorrect API key
- Wrong authentication header format
- API key expired or revoked

**Issue: "Spans not exported"**
- Application exits before BatchSpanProcessor flushes
- Add `force_flush()` before exit
- Use shorter batch timeout in development

## Test Configuration

For testing, you can use `ConsoleSpanExporter` to verify spans locally:

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Use console exporter for local testing
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(BatchSpanProcessor(console_exporter))
```

This will print spans to stdout so you can verify the data before sending to Last9.

## Production Best Practices

1. **Use BatchSpanProcessor** (not SimpleSpanProcessor) for better performance
2. **Set appropriate timeouts** for export
3. **Handle export failures** gracefully
4. **Monitor export queue size** for backpressure
5. **Use secure connections** (TLS/SSL)
6. **Rotate API keys** regularly
7. **Set resource attributes** for service identification

## Next Steps

- Review [examples/](examples/) for more integration patterns
- Check [README.md](README.md) for SDK features
- Contact Last9 support for endpoint configuration
- Set up alerts in Last9 for cost thresholds
