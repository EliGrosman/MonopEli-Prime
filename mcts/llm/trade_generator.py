"""Trade proposal generator using LLM.

Orchestrates: serialize state -> build prompt -> call LLM -> parse -> validate.
Uses heuristic trade candidates from ``trade_utils`` to provide context
in the prompt, helping the LLM focus on the most promising trades.
"""

from __future__ import annotations

import logging
from typing import Any

from monopoly_engine.actions import ProposeTrade
from monopoly_engine.game import MonopolyGame

from ..trade_utils import TradeCandidate, suggest_valuable_trades
from .client import LLMClient
from .prompts import SYSTEM_PROMPT, build_propose_prompt
from .state_prompt import serialize_game_state

logger = logging.getLogger(__name__)


class TradeGenerator:
    """Generates trade proposals using an LLM.

    The generator serializes the game state, enriches the prompt with
    heuristic trade candidates from :func:`suggest_valuable_trades`,
    calls the LLM for a decision, and validates the result against
    the engine's trade rules before returning.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def generate_proposal(
        self,
        game: MonopolyGame,
        player_id: int,
        trade_context: str = "",
    ) -> ProposeTrade | None:
        """Generate a trade proposal for the given player.

        Args:
            game: Current game instance.
            player_id: The player generating the trade.
            trade_context: Optional pre-built context string. If empty,
                auto-generated from :func:`suggest_valuable_trades`.

        Returns:
            A validated ``ProposeTrade`` action, or ``None`` if the LLM
            decides no good trade exists or the proposal fails validation.
        """
        # 1. Serialize game state from player's perspective
        state_text = serialize_game_state(game, player_id)

        # 2. Generate trade context if not provided
        if not trade_context:
            trade_context = _format_trade_candidates(
                suggest_valuable_trades(game, player_id),
            )

        # 3. Build prompt and call LLM
        user_prompt = build_propose_prompt(state_text, trade_context)
        response = self.client.complete_json(SYSTEM_PROMPT, user_prompt)

        # 4. Check for LLM parse errors or explicit no-trade
        if "error" in response:
            logger.warning("LLM returned error: %s", response.get("error"))
            return None
        if response.get("no_trade"):
            logger.info(
                "LLM decided no trade: %s", response.get("reasoning", ""),
            )
            return None

        # 5. Parse and validate
        proposal = self._parse_proposal(response, player_id)
        if proposal is None:
            return None

        valid, reason = self._validate_proposal(proposal, game)
        if not valid:
            logger.warning("LLM proposal failed validation: %s", reason)
            return None

        return proposal

    def _parse_proposal(
        self,
        response: dict[str, Any],
        player_id: int,
    ) -> ProposeTrade | None:
        """Parse LLM JSON response into a ProposeTrade action.

        Returns None if the response is missing required fields or has
        invalid types.
        """
        try:
            to_player = int(response["to_player"])
            give_properties = [int(p) for p in response.get("give_properties", [])]
            give_money = int(response.get("give_money", 0))
            want_properties = [int(p) for p in response.get("want_properties", [])]
            want_money = int(response.get("want_money", 0))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse LLM proposal: %s", exc)
            return None

        return ProposeTrade(
            player_id=player_id,
            to_player=to_player,
            give_properties=give_properties,
            give_money=give_money,
            want_properties=want_properties,
            want_money=want_money,
        )

    def _validate_proposal(
        self,
        proposal: ProposeTrade,
        game: MonopolyGame,
    ) -> tuple[bool, str]:
        """Validate a proposal against game rules.

        Uses the engine's ``ProposeTrade.validate()`` method, which checks
        ownership, money, building constraints, and player validity.
        """
        return proposal.validate(game)


def _format_trade_candidates(candidates: list[TradeCandidate]) -> str:
    """Format heuristic trade candidates as readable text for the LLM prompt."""
    if not candidates:
        return ""

    lines: list[str] = ["Suggested trade opportunities (from game analysis):"]
    for i, c in enumerate(candidates, 1):
        give = ", ".join(f"[{p}]" for p in c.give_properties) or "nothing"
        want = ", ".join(f"[{p}]" for p in c.want_properties) or "nothing"
        parts = [f"  {i}. Trade with Player {c.to_player}: offer {give}"]
        if c.give_money > 0:
            parts.append(f" + ${c.give_money}")
        parts.append(f" for {want}")
        if c.want_money > 0:
            parts.append(f" + ${c.want_money}")
        parts.append(f" (estimated value: {c.estimated_value:.0f})")
        lines.append("".join(parts))

    return "\n".join(lines)
