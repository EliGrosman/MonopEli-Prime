"""Prompt templates for LLM-powered Monopoly trade generation and evaluation.

All templates use ``{placeholder}`` syntax for string formatting.
Builder functions assemble the final prompt from game state serializations.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompt (shared across all trade-related LLM calls)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an expert Monopoly trading advisor. Your goal is to propose and evaluate \
trades that maximize your player's chance of winning.

KEY TRADING PRINCIPLES:
1. Monopoly completion is king. A property that completes a color set is worth \
2-5x its face value because it enables building houses/hotels.
2. Never give an opponent a monopoly for free. If your trade completes their set, \
demand significant compensation (cash, properties, or both).
3. Consider the board position. Properties on high-traffic areas (oranges, reds) \
are more valuable than low-traffic areas (greens, dark blues).
4. Cash has diminishing value. Once you have enough to build, extra cash sitting \
idle is worth less than a property.
5. Blocking trades have value. Holding a single property in an opponent's \
near-complete set gives you leverage.
6. Think about ALL players, not just the trade partner. A trade that helps you \
but helps your partner even more can lose you the game.

PROPERTY VALUE TIERS (approximate):
- Oranges/Reds: Highest ROI (best rent-to-cost ratio, high landing frequency)
- Light Blues/Magentas: Good early game value, cheap to develop
- Yellows/Greens: High rent but expensive to develop
- Dark Blues: Highest rent but lowest landing frequency
- Railroads: Steady income, value increases with count owned
- Utilities: Lowest value, inconsistent rent

OUTPUT FORMAT: Always respond with valid JSON matching the requested schema. \
Do not include any text outside the JSON object."""


# ---------------------------------------------------------------------------
# Propose trade template
# ---------------------------------------------------------------------------

PROPOSE_TRADE_TEMPLATE: str = """\
Given the current game state, propose a trade that benefits your position.

{game_state}

TRADE ANALYSIS (provided by game engine):
{trade_context}

Propose a trade with ONE other player. You may offer/request:
- Multiple properties
- Cash in either direction
- Any combination

IMPORTANT: Use board POSITION NUMBERS (not names) for properties. Position numbers \
are shown in brackets in the state above. This avoids ambiguity.

Respond with JSON:
{{
  "to_player": <player_id>,
  "give_properties": [<position_number>, ...],
  "give_money": <amount>,
  "want_properties": [<position_number>, ...],
  "want_money": <amount>,
  "reasoning": "<1-2 sentence explanation>"
}}

If no good trade exists, respond with:
{{"no_trade": true, "reasoning": "<why>"}}"""


# ---------------------------------------------------------------------------
# Evaluate trade template
# ---------------------------------------------------------------------------

EVALUATE_TRADE_TEMPLATE: str = """\
You have received a trade proposal. Evaluate whether to accept, reject, \
or counter-propose.

{game_state}

INCOMING TRADE:
{trade_description}

TRADE IMPACT (provided by game engine):
{trade_impact}

Respond with JSON:
{{
  "decision": "accept" | "reject" | "counter",
  "reasoning": "<1-2 sentence explanation>",
  "counter_proposal": {{
    "give_properties": [<position>, ...],
    "give_money": <amount>,
    "want_properties": [<position>, ...],
    "want_money": <amount>
  }}
}}

Include "counter_proposal" only if decision is "counter"."""


# ---------------------------------------------------------------------------
# Counter-propose template
# ---------------------------------------------------------------------------

COUNTER_PROPOSE_TEMPLATE: str = """\
Your previous trade was rejected. The opponent said: "{reason}"

{game_state}

PREVIOUS OFFER:
{previous_trade}

Propose a modified trade that addresses their concerns, or decide to stop \
negotiating.

Respond with JSON:
{{
  "action": "counter" | "stop",
  "reasoning": "<explanation>",
  "counter_proposal": {{
    "to_player": <player_id>,
    "give_properties": [<position>, ...],
    "give_money": <amount>,
    "want_properties": [<position>, ...],
    "want_money": <amount>
  }}
}}

Include "counter_proposal" only if action is "counter"."""


# ---------------------------------------------------------------------------
# Propose trade choice template (multiple-choice variant)
# ---------------------------------------------------------------------------

PROPOSE_TRADE_CHOICE_TEMPLATE: str = """\
Given the current game state, select the BEST trade from the options below, \
or choose "none".

{game_state}

TRADE OPTIONS:
{trade_options}

Pick the option number of the best trade. If none are worthwhile, pick 0.

Respond with JSON:
{{"choice": <option_number_or_0>, "reasoning": "<1-2 sentences>"}}"""


# ---------------------------------------------------------------------------
# JSON output schemas (for providers that support structured output)
# ---------------------------------------------------------------------------

PROPOSE_TRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "properties": {
                "to_player": {"type": "integer"},
                "give_properties": {
                    "type": "array", "items": {"type": "integer"},
                },
                "give_money": {"type": "integer", "minimum": 0},
                "want_properties": {
                    "type": "array", "items": {"type": "integer"},
                },
                "want_money": {"type": "integer", "minimum": 0},
                "reasoning": {"type": "string"},
            },
            "required": [
                "to_player", "give_properties", "give_money",
                "want_properties", "want_money", "reasoning",
            ],
        },
        {
            "properties": {
                "no_trade": {"type": "boolean", "const": True},
                "reasoning": {"type": "string"},
            },
            "required": ["no_trade", "reasoning"],
        },
    ],
}

EVALUATE_TRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject", "counter"]},
        "reasoning": {"type": "string"},
        "counter_proposal": {
            "type": "object",
            "properties": {
                "give_properties": {
                    "type": "array", "items": {"type": "integer"},
                },
                "give_money": {"type": "integer", "minimum": 0},
                "want_properties": {
                    "type": "array", "items": {"type": "integer"},
                },
                "want_money": {"type": "integer", "minimum": 0},
            },
        },
    },
    "required": ["decision", "reasoning"],
}

COUNTER_PROPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["counter", "stop"]},
        "reasoning": {"type": "string"},
        "counter_proposal": {
            "type": "object",
            "properties": {
                "to_player": {"type": "integer"},
                "give_properties": {
                    "type": "array", "items": {"type": "integer"},
                },
                "give_money": {"type": "integer", "minimum": 0},
                "want_properties": {
                    "type": "array", "items": {"type": "integer"},
                },
                "want_money": {"type": "integer", "minimum": 0},
            },
        },
    },
    "required": ["action", "reasoning"],
}

PROPOSE_TRADE_CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "minimum": 0},
        "reasoning": {"type": "string"},
    },
    "required": ["choice", "reasoning"],
}


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_propose_prompt(game_state_text: str, trade_context: str) -> str:
    """Build the user prompt for proposing a trade.

    Args:
        game_state_text: Output of ``serialize_game_state()``.
        trade_context: Output of ``suggest_valuable_trades()`` or similar,
            formatted as readable text.
    """
    return PROPOSE_TRADE_TEMPLATE.format(
        game_state=game_state_text,
        trade_context=trade_context or "(no specific trade analysis available)",
    )


def build_evaluate_prompt(
    game_state_text: str, trade_text: str, impact_text: str,
) -> str:
    """Build the user prompt for evaluating an incoming trade.

    Args:
        game_state_text: Output of ``serialize_game_state()``.
        trade_text: Output of ``serialize_trade_proposal()``.
        impact_text: Output of ``trade_impact()`` formatted as text.
    """
    return EVALUATE_TRADE_TEMPLATE.format(
        game_state=game_state_text,
        trade_description=trade_text,
        trade_impact=impact_text,
    )


def build_counter_prompt(
    game_state_text: str, trade_text: str, reason: str,
) -> str:
    """Build the user prompt for counter-proposing after rejection.

    Args:
        game_state_text: Output of ``serialize_game_state()``.
        trade_text: The previous trade proposal text.
        reason: The opponent's reason for rejection.
    """
    return COUNTER_PROPOSE_TEMPLATE.format(
        game_state=game_state_text,
        previous_trade=trade_text,
        reason=reason,
    )


def build_propose_choice_prompt(
    game_state_text: str, trade_options_text: str,
) -> str:
    """Build the user prompt for choosing from pre-computed trade candidates.

    Args:
        game_state_text: Output of ``serialize_game_state()``.
        trade_options_text: Numbered list of candidate trades, e.g.
            ``"1. Give [Baltic Ave] to Player 2 for $200\\n2. ..."``.
    """
    return PROPOSE_TRADE_CHOICE_TEMPLATE.format(
        game_state=game_state_text,
        trade_options=trade_options_text,
    )
