# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-14

### Added
- Initial open source release of Last9 Python AI SDK
- **Cost Tracking**: Automatic cost calculation for 20+ AI models
  - Anthropic: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
  - OpenAI: GPT-4o, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
  - Google: Gemini Pro, Gemini 1.5 Pro, Gemini 1.5 Flash
  - Cohere: Command R, Command R+
  - And more...
- **Conversation Tracking**: Multi-turn conversation tracking with `gen_ai.conversation.id`
- **Workflow Management**: Cost aggregation across multi-step workflows
- **Span Classification**: `gen_ai.l9.span.kind` for filtering (llm/tool/prompt)
- **Prompt Versioning**: Hash-based prompt template tracking and versioning
- **Tool/Function Tracking**: Enhanced attributes for tool and function calls
- **Performance Metrics**: Response times, request/response sizes, quality scores
- **Content Events**: Input/output prompts as span events
- **Standard Compliance**: Full compatibility with OpenTelemetry GenAI v1.28.0 conventions

### Documentation
- Comprehensive README with usage examples
- Installation guide (INSTALL.md)
- Complete API reference
- Working examples:
  - Basic usage with cost tracking
  - Anthropic Claude SDK integration
  - Conversation tracking with multi-turn support
  - Tool and function call tracking

### Technical Details
- Built on OpenTelemetry Python SDK
- Requires Python >=3.9 (aligned with OpenTelemetry API requirements)
- Zero dependencies beyond OpenTelemetry
- Works with any OTLP-compatible backend
- Feature parity with last9-node-agent

### Integration Support
- OpenAI Python SDK
- Anthropic Python SDK
- LangChain (via examples)
- FastAPI (via examples)
- Framework-agnostic design

---

## Versioning Guidelines

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

For more information, see the [SPEC.md](SPEC.md) document.
