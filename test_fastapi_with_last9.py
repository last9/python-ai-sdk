#!/usr/bin/env python3
"""
Test FastAPI app with Last9 telemetry export

This script tests the FastAPI app and exports real traces to Last9
"""

import os
import sys
import time
import requests
from threading import Thread

# Verify required environment variables are set
# Set these before running:
# export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic YOUR_KEY"
# export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-aps1.last9.io:443"
if "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ:
    print("❌ Error: OTEL_EXPORTER_OTLP_HEADERS environment variable not set")
    print("   Run: export OTEL_EXPORTER_OTLP_HEADERS=\"Authorization=Basic YOUR_KEY\"")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))


def start_fastapi_server():
    """Start FastAPI server in a thread"""
    import uvicorn
    from examples.fastapi_app import app

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


def test_endpoints():
    """Test FastAPI endpoints"""
    base_url = "http://127.0.0.1:8000"

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server started successfully!\n")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("❌ Server failed to start")
        return False

    print("=" * 60)
    print("Testing FastAPI with Last9 Telemetry Export")
    print("=" * 60)

    # Test 1: Health check
    print("\n1️⃣ Testing Health Check Endpoint")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Root endpoint
    print("\n2️⃣ Testing Root Endpoint")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Name: {data['name']}")
        print(f"   Endpoints: {len(data['endpoints'])}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Chat endpoint (mock mode - no real API calls)
    print("\n3️⃣ Testing Chat Endpoint (Mock Mode)")
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={
                "message": "Test message for Last9 telemetry",
                "model": "gpt-3.5-turbo",
                "user_id": "test_user_123",
            },
            timeout=10,
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Response: {data['response'][:50]}...")
        print(f"   Cost: ${data['cost_usd']:.6f} USD")
        print(f"   Tokens: {data['tokens_used']}")
        print(f"   Conversation ID: {data['conversation_id']}")
        print(f"   ✅ Trace sent to Last9!")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 4: Summarize endpoint (mock mode)
    print("\n4️⃣ Testing Summarize Endpoint (Mock Mode)")
    try:
        response = requests.post(
            f"{base_url}/summarize",
            json={"text": "This is a test document that needs summarization for Last9 telemetry testing.", "model": "claude-3-haiku"},
            timeout=10,
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Summary: {data['summary'][:50]}...")
        print(f"   Cost: ${data['cost_usd']:.6f} USD")
        print(f"   Original length: {data['original_length']} chars")
        print(f"   ✅ Trace sent to Last9!")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 5: Workflow endpoint
    print("\n5️⃣ Testing Multi-Step Workflow Endpoint")
    try:
        response = requests.post(
            f"{base_url}/workflow/customer-support",
            json={"message": "I need help with my account", "model": "gpt-3.5-turbo", "user_id": "workflow_test_user"},
            timeout=15,
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Classification: {data['classification']}")
        print(f"   Response: {data['response'][:50]}...")
        print(f"   Total Cost: ${data['total_cost_usd']:.6f} USD")
        print(f"   LLM Calls: {data['llm_calls']}")
        print(f"   Workflow ID: {data['workflow_id']}")
        print(f"   ✅ Workflow traces sent to Last9!")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print(f"🚀 Check your Last9 dashboard for traces:")
    print(f"   Environment: local")
    print(f"   Service: python-ai-sdk-test")
    print("=" * 60 + "\n")

    return True


if __name__ == "__main__":
    # Start server in background thread
    server_thread = Thread(target=start_fastapi_server, daemon=True)
    server_thread.start()

    # Run tests
    try:
        success = test_endpoints()

        # Keep server running briefly to flush traces
        if success:
            print("⏳ Flushing traces to Last9...")
            time.sleep(3)

        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
        sys.exit(0)
