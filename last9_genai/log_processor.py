"""
Log-to-span bridge for OTel GenAI semantic conventions.

OpenTelemetry's newer GenAI instrumentations (e.g. opentelemetry-instrumentation-openai-v2)
emit request messages, responses, and tool calls as OTel log events — not as span
attributes. Last9's LLM dashboard reads span attributes / events, so without this bridge
those payloads never reach the dashboard.

This processor promotes well-known GenAI log events onto the currently active span using
the flat + span-event scheme the Last9 LLM dashboard parses:
  - Span attribute `gen_ai.prompt`       : JSON array of prompt messages
  - Span attribute `gen_ai.completion`   : JSON array of completion choices
  - Span event `gen_ai.content.prompt`   : { content: <json array> }
  - Span event `gen_ai.content.completion`: { completion: <json array> }

Indexed attributes (`gen_ai.prompt.{i}.*`) are also emitted for compatibility with
AgentOps / Traceloop-style consumers.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List

from opentelemetry import trace
from opentelemetry.sdk._logs import LogRecordProcessor, ReadWriteLogRecord

GEN_AI_PROMPT_EVENTS = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}
GEN_AI_CHOICE_EVENT = "gen_ai.choice"


class Last9LogToSpanProcessor(LogRecordProcessor):
    """Promote GenAI log events to span attributes + events on the active span.

    Writes flat JSON-array attributes (what the Last9 LLM dashboard parses) and
    indexed attributes (AgentOps/Traceloop convention) so downstream renderers
    in either scheme can consume the payload.
    """

    def __init__(self, max_content_length: int = 4096):
        self._max = max_content_length
        self._state: Dict[int, Dict[str, List[dict]]] = {}
        self._lock = threading.Lock()

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        record = log_record.log_record
        event_name = getattr(record, "event_name", None)
        if not event_name:
            return
        if event_name != GEN_AI_CHOICE_EVENT and event_name not in GEN_AI_PROMPT_EVENTS:
            return

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid or not span.is_recording():
            return

        body = record.body
        if not isinstance(body, dict):
            return

        with self._lock:
            state = self._state.setdefault(ctx.span_id, {"prompts": [], "completions": []})

            if event_name == GEN_AI_CHOICE_EVENT:
                idx = len(state["completions"])
                entry = self._build_completion_entry(body)
                state["completions"].append(entry)
                self._set_completion_indexed(span, idx, entry, body)
                self._set_completion_flat(span, state["completions"])
            else:
                idx = len(state["prompts"])
                default_role = GEN_AI_PROMPT_EVENTS[event_name]
                entry = self._build_prompt_entry(default_role, body)
                state["prompts"].append(entry)
                self._set_prompt_indexed(span, idx, entry, body)
                self._set_prompt_flat(span, state["prompts"])

    def cleanup_span(self, span_id: int) -> None:
        """Release per-span state when the span ends (called from Last9SpanProcessor)."""
        with self._lock:
            self._state.pop(span_id, None)

    def shutdown(self) -> None:
        with self._lock:
            self._state.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _truncate(self, value: Any) -> str:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(s) > self._max:
            return s[: self._max] + "...[truncated]"
        return s

    def _build_prompt_entry(self, default_role: str, body: dict) -> dict:
        entry: Dict[str, Any] = {"role": body.get("role", default_role)}
        content = body.get("content")
        if content is not None:
            entry["content"] = content
        if body.get("tool_calls"):
            entry["tool_calls"] = body["tool_calls"]
        if body.get("id"):
            entry["tool_call_id"] = body["id"]
        return entry

    def _build_completion_entry(self, body: dict) -> dict:
        message = body.get("message") or {}
        entry: Dict[str, Any] = {"role": message.get("role", "assistant")}
        if message.get("content") is not None:
            entry["content"] = message["content"]
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        if body.get("finish_reason") is not None:
            entry["finish_reason"] = body["finish_reason"]
        if body.get("index") is not None:
            entry["index"] = body["index"]
        return entry

    def _set_prompt_indexed(self, span, idx: int, entry: dict, body: dict) -> None:
        span.set_attribute(f"gen_ai.prompt.{idx}.role", entry["role"])
        if "content" in entry:
            span.set_attribute(f"gen_ai.prompt.{idx}.content", self._truncate(entry["content"]))
        if "tool_calls" in entry:
            span.set_attribute(
                f"gen_ai.prompt.{idx}.tool_calls", self._truncate(entry["tool_calls"])
            )
        if "tool_call_id" in entry:
            span.set_attribute(f"gen_ai.prompt.{idx}.tool_call.id", str(entry["tool_call_id"]))

    def _set_completion_indexed(self, span, idx: int, entry: dict, body: dict) -> None:
        span.set_attribute(f"gen_ai.completion.{idx}.role", entry["role"])
        if "content" in entry:
            span.set_attribute(f"gen_ai.completion.{idx}.content", self._truncate(entry["content"]))
        if "tool_calls" in entry:
            span.set_attribute(
                f"gen_ai.completion.{idx}.tool_calls",
                self._truncate(entry["tool_calls"]),
            )
        if "finish_reason" in entry:
            span.set_attribute(f"gen_ai.completion.{idx}.finish_reason", entry["finish_reason"])
        if "index" in entry:
            span.set_attribute(f"gen_ai.completion.{idx}.index", entry["index"])

    def _set_prompt_flat(self, span, prompts: List[dict]) -> None:
        payload = self._truncate(prompts)
        span.set_attribute("gen_ai.prompt", payload)
        span.add_event("gen_ai.content.prompt", {"content": payload})

    def _set_completion_flat(self, span, completions: List[dict]) -> None:
        payload = self._truncate(completions)
        span.set_attribute("gen_ai.completion", payload)
        span.add_event("gen_ai.content.completion", {"completion": payload})
