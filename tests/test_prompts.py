"""Tests for LLM prompt templates and builder functions."""

from __future__ import annotations

from mcts.llm.prompts import (
    COUNTER_PROPOSE_SCHEMA,
    COUNTER_PROPOSE_TEMPLATE,
    EVALUATE_TRADE_SCHEMA,
    EVALUATE_TRADE_TEMPLATE,
    PROPOSE_TRADE_SCHEMA,
    PROPOSE_TRADE_TEMPLATE,
    SYSTEM_PROMPT,
    build_counter_prompt,
    build_evaluate_prompt,
    build_propose_prompt,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_mentions_monopoly(self) -> None:
        assert "Monopoly" in SYSTEM_PROMPT

    def test_mentions_json(self) -> None:
        assert "JSON" in SYSTEM_PROMPT

    def test_mentions_key_principles(self) -> None:
        assert "Monopoly completion" in SYSTEM_PROMPT
        assert "Blocking trades" in SYSTEM_PROMPT

    def test_mentions_property_tiers(self) -> None:
        assert "Oranges/Reds" in SYSTEM_PROMPT
        assert "Utilities" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_propose_template_has_placeholders(self) -> None:
        assert "{game_state}" in PROPOSE_TRADE_TEMPLATE
        assert "{trade_context}" in PROPOSE_TRADE_TEMPLATE

    def test_evaluate_template_has_placeholders(self) -> None:
        assert "{game_state}" in EVALUATE_TRADE_TEMPLATE
        assert "{trade_description}" in EVALUATE_TRADE_TEMPLATE
        assert "{trade_impact}" in EVALUATE_TRADE_TEMPLATE

    def test_counter_template_has_placeholders(self) -> None:
        assert "{game_state}" in COUNTER_PROPOSE_TEMPLATE
        assert "{previous_trade}" in COUNTER_PROPOSE_TEMPLATE
        assert "{reason}" in COUNTER_PROPOSE_TEMPLATE

    def test_propose_template_mentions_position_numbers(self) -> None:
        assert "POSITION NUMBERS" in PROPOSE_TRADE_TEMPLATE

    def test_evaluate_template_has_decision_options(self) -> None:
        assert "accept" in EVALUATE_TRADE_TEMPLATE
        assert "reject" in EVALUATE_TRADE_TEMPLATE
        assert "counter" in EVALUATE_TRADE_TEMPLATE

    def test_counter_template_has_action_options(self) -> None:
        assert '"counter"' in COUNTER_PROPOSE_TEMPLATE
        assert '"stop"' in COUNTER_PROPOSE_TEMPLATE


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_propose_schema_structure(self) -> None:
        assert PROPOSE_TRADE_SCHEMA["type"] == "object"
        assert "oneOf" in PROPOSE_TRADE_SCHEMA

    def test_evaluate_schema_has_required_fields(self) -> None:
        assert "decision" in EVALUATE_TRADE_SCHEMA["required"]
        assert "reasoning" in EVALUATE_TRADE_SCHEMA["required"]

    def test_counter_schema_has_required_fields(self) -> None:
        assert "action" in COUNTER_PROPOSE_SCHEMA["required"]
        assert "reasoning" in COUNTER_PROPOSE_SCHEMA["required"]


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


class TestBuildPropose:
    def test_includes_game_state(self) -> None:
        result = build_propose_prompt("STATE_TEXT", "CONTEXT_TEXT")
        assert "STATE_TEXT" in result

    def test_includes_trade_context(self) -> None:
        result = build_propose_prompt("STATE_TEXT", "CONTEXT_TEXT")
        assert "CONTEXT_TEXT" in result

    def test_empty_context_gets_fallback(self) -> None:
        result = build_propose_prompt("STATE_TEXT", "")
        assert "no specific trade analysis" in result

    def test_no_raw_placeholders_remain(self) -> None:
        result = build_propose_prompt("STATE", "CTX")
        assert "{game_state}" not in result
        assert "{trade_context}" not in result


class TestBuildEvaluate:
    def test_includes_all_sections(self) -> None:
        result = build_evaluate_prompt("STATE", "TRADE_DESC", "IMPACT")
        assert "STATE" in result
        assert "TRADE_DESC" in result
        assert "IMPACT" in result

    def test_no_raw_placeholders_remain(self) -> None:
        result = build_evaluate_prompt("S", "T", "I")
        assert "{game_state}" not in result
        assert "{trade_description}" not in result
        assert "{trade_impact}" not in result


class TestBuildCounter:
    def test_includes_all_sections(self) -> None:
        result = build_counter_prompt("STATE", "PREV_TRADE", "too expensive")
        assert "STATE" in result
        assert "PREV_TRADE" in result
        assert "too expensive" in result

    def test_no_raw_placeholders_remain(self) -> None:
        result = build_counter_prompt("S", "T", "R")
        assert "{game_state}" not in result
        assert "{previous_trade}" not in result
        assert "{reason}" not in result
