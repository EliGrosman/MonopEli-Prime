"""Trade responder: evaluates incoming trade proposals using an LLM.

The responder returns structured ``TradeDecision`` dicts that the
``NegotiationManager`` translates into engine actions (accept/reject/counter).
A fast path skips the LLM entirely when MCTS value estimates show a clear win.
"""

from __future__ import annotations

import logging
from typing import Any

from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import TradeOfferData

from ..trade_utils import trade_impact
from .client import LLMClient
from .prompts import SYSTEM_PROMPT, build_counter_prompt, build_evaluate_prompt
from .state_prompt import serialize_game_state, serialize_trade_proposal
from .trade_generator import _parse_int_list

logger = logging.getLogger(__name__)

# The responder returns a decision dict, NOT engine actions directly.
# The HybridAgent / NegotiationManager translates this into engine actions.
TradeDecision = dict[str, Any]


def _format_impact(impact: dict[str, Any]) -> str:
    """Format a trade_impact result as readable text for the LLM prompt."""
    from_id = impact["from_player_id"]
    to_id = impact["to_player_id"]
    lines: list[str] = []

    lines.append(f"Net worth change for Player {from_id}: {impact['from_player_net_change']:+d}")
    lines.append(f"Net worth change for Player {to_id}: {impact['to_player_net_change']:+d}")

    created = impact.get("monopolies_created", [])
    if created:
        for pid, color in created:
            lines.append(f"  CREATES monopoly: Player {pid} gets {color}")

    broken = impact.get("monopolies_broken", [])
    if broken:
        for pid, color in broken:
            lines.append(f"  BREAKS monopoly: Player {pid} loses {color}")

    return "\n".join(lines)


class TradeResponder:
    """Evaluates and responds to incoming trade proposals using an LLM.

    The responder supports a *fast path*: if MCTS value estimates are
    provided and the trade clearly improves the player's position
    (value delta > threshold), it accepts without an LLM call.

    Otherwise, it serializes the game state and trade details, calls the
    LLM, and returns a structured decision dict.
    """

    def __init__(
        self,
        client: LLMClient,
        mcts_value_threshold: float = 0.10,
    ) -> None:
        self.client = client
        self.mcts_value_threshold = mcts_value_threshold

    def respond_to_trade(
        self,
        game: MonopolyGame,
        player_id: int,
        trade: TradeOfferData,
        mcts_value_before: float | None = None,
        mcts_value_after: float | None = None,
    ) -> TradeDecision:
        """Decide whether to accept, reject, or counter-propose.

        Args:
            game: Current game instance.
            player_id: The player evaluating the trade (the receiver).
            trade: The incoming trade proposal.
            mcts_value_before: Optional MCTS win probability before trade.
            mcts_value_after: Optional MCTS win probability after trade.

        Returns:
            A dict with at minimum ``{"decision": "accept"|"reject"|"counter",
            "reasoning": "..."}`` and optionally ``"counter_proposal": {...}``.
        """
        # Fast path: MCTS says clearly beneficial
        fast = self._fast_path_check(mcts_value_before, mcts_value_after)
        if fast is not None:
            return {
                "decision": fast,
                "reasoning": (
                    f"MCTS value improvement: "
                    f"{(mcts_value_after or 0) - (mcts_value_before or 0):.2f}"
                ),
            }

        # Normal path: ask LLM
        state_text = serialize_game_state(game, player_id)
        trade_text = serialize_trade_proposal(trade, game)
        impact = trade_impact(game, trade)
        impact_text = _format_impact(impact)

        user_prompt = build_evaluate_prompt(state_text, trade_text, impact_text)
        response = self.client.complete_json(SYSTEM_PROMPT, user_prompt)

        return self._parse_response(response, player_id)

    def generate_counter(
        self,
        game: MonopolyGame,
        player_id: int,
        previous_trade: TradeOfferData,
        rejection_reason: str,
    ) -> TradeDecision:
        """Generate a counter-proposal after rejecting a trade.

        Args:
            game: Current game instance.
            player_id: The player generating the counter.
            previous_trade: The trade that was rejected.
            rejection_reason: Why the trade was rejected.

        Returns:
            A dict with ``{"action": "counter"|"stop", "reasoning": "...",
            "counter_proposal": {...}}`` (counter_proposal only if action
            is "counter").
        """
        state_text = serialize_game_state(game, player_id)
        trade_text = serialize_trade_proposal(previous_trade, game)

        user_prompt = build_counter_prompt(
            state_text,
            trade_text,
            rejection_reason,
        )
        response = self.client.complete_json(SYSTEM_PROMPT, user_prompt)

        return self._parse_counter_response(response)

    def _fast_path_check(
        self,
        value_before: float | None,
        value_after: float | None,
    ) -> str | None:
        """Check if fast path applies.

        Returns ``"accept"`` if the MCTS value improvement exceeds the
        threshold, or ``None`` to fall through to the LLM path.
        """
        if value_before is None or value_after is None:
            return None

        delta = value_after - value_before
        if delta > self.mcts_value_threshold:
            return "accept"

        return None

    def _parse_response(
        self,
        response: dict[str, Any],
        player_id: int,
    ) -> TradeDecision:
        """Parse LLM response into a TradeDecision.

        Normalizes the decision field and extracts counter_proposal if present.
        Falls back to reject on parse errors.
        """
        if "error" in response:
            logger.warning("LLM returned error, defaulting to reject")
            return {
                "decision": "reject",
                "reasoning": "LLM failed to produce valid response",
            }

        decision = str(response.get("decision", "reject")).lower().strip()
        if decision not in ("accept", "reject", "counter"):
            decision = "reject"

        reasoning = str(response.get("reasoning", ""))

        result: TradeDecision = {
            "decision": decision,
            "reasoning": reasoning,
        }

        if decision == "counter" and "counter_proposal" in response:
            cp = response["counter_proposal"]
            result["counter_proposal"] = self._normalize_counter(
                cp,
                player_id,
            )

        return result

    def _parse_counter_response(
        self,
        response: dict[str, Any],
    ) -> TradeDecision:
        """Parse LLM counter-proposal response."""
        if "error" in response:
            logger.warning("LLM returned error, defaulting to stop")
            return {
                "action": "stop",
                "reasoning": "LLM failed to produce valid response",
            }

        action = str(response.get("action", "stop")).lower().strip()
        if action not in ("counter", "stop"):
            action = "stop"

        reasoning = str(response.get("reasoning", ""))

        result: TradeDecision = {
            "action": action,
            "reasoning": reasoning,
        }

        if action == "counter" and "counter_proposal" in response:
            cp = response["counter_proposal"]
            result["counter_proposal"] = self._normalize_counter(cp)

        return result

    @staticmethod
    def _normalize_counter(
        cp: Any,
        player_id: int | None = None,
    ) -> dict[str, Any]:
        """Normalize a counter-proposal dict, coercing types."""
        if not isinstance(cp, dict):
            return {}

        result: dict[str, Any] = {}
        try:
            if "to_player" in cp:
                result["to_player"] = int(cp["to_player"])
            elif player_id is not None:
                result["to_player"] = player_id
            result["give_properties"] = _parse_int_list(
                cp.get("give_properties", []),
            )
            result["give_money"] = int(cp.get("give_money", 0))
            result["want_properties"] = _parse_int_list(
                cp.get("want_properties", []),
            )
            result["want_money"] = int(cp.get("want_money", 0))
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to normalize counter-proposal: %s", exc)
            return {}

        return result
