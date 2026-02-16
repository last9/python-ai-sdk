"""Tests for cost calculation functionality"""

import pytest
from last9_genai import calculate_llm_cost, ModelPricing

# Test pricing data
TEST_PRICING = {
    "claude-3-5-sonnet": ModelPricing(input=3.0, output=15.0),
    "gpt-4o": ModelPricing(input=2.50, output=10.0),
    "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50),
}


class TestCostCalculation:
    """Test cost calculation functions"""

    def test_calculate_cost_anthropic(self):
        """Test cost calculation for Anthropic model"""
        usage = {"input_tokens": 1000, "output_tokens": 500}
        cost = calculate_llm_cost("claude-3-5-sonnet", usage, TEST_PRICING)

        # Claude 3.5 Sonnet: $3/million input, $15/million output
        expected_input = (1000 / 1_000_000) * 3.0  # $0.003
        expected_output = (500 / 1_000_000) * 15.0  # $0.0075
        expected_total = expected_input + expected_output  # $0.0105

        assert cost.input == pytest.approx(expected_input, rel=1e-6)
        assert cost.output == pytest.approx(expected_output, rel=1e-6)
        assert cost.total == pytest.approx(expected_total, rel=1e-6)

    def test_calculate_cost_openai(self):
        """Test cost calculation for OpenAI model"""
        usage = {"input_tokens": 2000, "output_tokens": 1000}
        cost = calculate_llm_cost("gpt-4o", usage, TEST_PRICING)

        # GPT-4o: $2.50/million input, $10/million output
        expected_input = (2000 / 1_000_000) * 2.50  # $0.005
        expected_output = (1000 / 1_000_000) * 10.0  # $0.010
        expected_total = expected_input + expected_output  # $0.015

        assert cost.input == pytest.approx(expected_input, rel=1e-6)
        assert cost.output == pytest.approx(expected_output, rel=1e-6)
        assert cost.total == pytest.approx(expected_total, rel=1e-6)

    def test_calculate_cost_with_old_naming(self):
        """Test cost calculation with old token naming convention"""
        usage = {"prompt_tokens": 500, "completion_tokens": 300}
        cost = calculate_llm_cost("gpt-3.5-turbo", usage, TEST_PRICING)

        # Should work with old naming
        assert cost.input > 0
        assert cost.output > 0
        assert cost.total == cost.input + cost.output

    def test_calculate_cost_no_pricing(self):
        """Test that calculate_llm_cost returns None when no pricing provided"""
        usage = {"input_tokens": 1000, "output_tokens": 500}

        # Without custom_pricing
        cost = calculate_llm_cost("gpt-4o", usage, None)
        assert cost is None

        # With empty custom_pricing
        cost = calculate_llm_cost("gpt-4o", usage, {})
        assert cost is None

    def test_calculate_cost_unknown_model(self):
        """Test that unknown model returns None when not in custom_pricing"""
        usage = {"input_tokens": 1000, "output_tokens": 500}
        cost = calculate_llm_cost("unknown-model", usage, TEST_PRICING)

        # Unknown model not in TEST_PRICING should return None
        assert cost is None

    def test_calculate_cost_custom_pricing(self):
        """Test cost calculation with custom pricing"""
        custom_pricing = {"my-model": ModelPricing(input=5.0, output=10.0)}
        usage = {"input_tokens": 1000, "output_tokens": 500}
        cost = calculate_llm_cost("my-model", usage, custom_pricing)

        expected_input = (1000 / 1_000_000) * 5.0
        expected_output = (500 / 1_000_000) * 10.0

        assert cost.input == pytest.approx(expected_input, rel=1e-6)
        assert cost.output == pytest.approx(expected_output, rel=1e-6)

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        usage = {"input_tokens": 0, "output_tokens": 0}
        cost = calculate_llm_cost("gpt-4o", usage, TEST_PRICING)

        assert cost.input == 0.0
        assert cost.output == 0.0
        assert cost.total == 0.0

    def test_model_pricing_structure(self):
        """Test that ModelPricing has correct structure"""
        pricing = ModelPricing(input=3.0, output=15.0)

        assert hasattr(pricing, "input")
        assert hasattr(pricing, "output")
        assert pricing.input == 3.0
        assert pricing.output == 15.0
