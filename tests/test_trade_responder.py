"""Tests for LLM trade responder."""

from __future__ import annotations

from typing import Any

from mcts.llm.client import LLMClient, LLMConfig
from mcts.llm.trade_responder import (
    TradeResponder,
    _format_impact,
)
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import TradeOfferData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLLMClient(LLMClient):
    """LLMClient that returns predetermined JSON dicts from complete_json."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(LLMConfig())
        self._responses = responses
        self._call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Use complete_json")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp

    def close(self) -> None:
        pass


def _make_game() -> MonopolyGame:
    """Create a game with property ownership for trading tests."""
    game = MonopolyGame(num_players=3, seed=42)
    pm = game.property_manager

    # Player 0 owns Mediterranean [1] and Connecticut [9]
    pm.properties[1].owner = 0
    pm.properties[9].owner = 0

    # Player 1 owns Oriental [6] and Vermont [8]
    pm.properties[6].owner = 1
    pm.properties[8].owner = 1

    # Player 2 owns St. Charles [11]
    pm.properties[11].owner = 2

    return game


def _make_trade() -> TradeOfferData:
    """A trade from Player 1 to Player 0: offer Oriental [6] for Connecticut [9]."""
    return TradeOfferData(
        from_player=1,
        to_player=0,
        give_properties=[6],
        give_money=0,
        want_properties=[9],
        want_money=0,
    )


# ---------------------------------------------------------------------------
# _format_impact
# ---------------------------------------------------------------------------


class TestFormatImpact:
    def test_basic_format(self) -> None:
        impact: dict[str, Any] = {
            "from_player_id": 0,
            "to_player_id": 1,
            "from_player_net_change": -50,
            "to_player_net_change": 80,
            "monopolies_created": [],
            "monopolies_broken": [],
        }
        result = _format_impact(impact)
        assert "Player 0" in result
        assert "-50" in result
        assert "+80" in result

    def test_monopolies_created(self) -> None:
        impact: dict[str, Any] = {
            "from_player_id": 0,
            "to_player_id": 1,
            "from_player_net_change": 0,
            "to_player_net_change": 0,
            "monopolies_created": [(1, "LIGHT_BLUE")],
            "monopolies_broken": [],
        }
        result = _format_impact(impact)
        assert "CREATES monopoly" in result
        assert "LIGHT_BLUE" in result

    def test_monopolies_broken(self) -> None:
        impact: dict[str, Any] = {
            "from_player_id": 0,
            "to_player_id": 1,
            "from_player_net_change": 0,
            "to_player_net_change": 0,
            "monopolies_created": [],
            "monopolies_broken": [(0, "BROWN")],
        }
        result = _format_impact(impact)
        assert "BREAKS monopoly" in result
        assert "BROWN" in result


# ---------------------------------------------------------------------------
# Fast path
# ---------------------------------------------------------------------------


class TestFastPath:
    def test_fast_accept(self) -> None:
        client = _FakeLLMClient([{}])
        responder = TradeResponder(client, mcts_value_threshold=0.10)
        result = responder._fast_path_check(0.30, 0.50)
        assert result == "accept"

    def test_no_fast_path_below_threshold(self) -> None:
        client = _FakeLLMClient([{}])
        responder = TradeResponder(client, mcts_value_threshold=0.10)
        result = responder._fast_path_check(0.30, 0.35)
        assert result is None

    def test_no_fast_path_when_none(self) -> None:
        client = _FakeLLMClient([{}])
        responder = TradeResponder(client, mcts_value_threshold=0.10)
        assert responder._fast_path_check(None, 0.50) is None
        assert responder._fast_path_check(0.30, None) is None
        assert responder._fast_path_check(None, None) is None

    def test_custom_threshold(self) -> None:
        client = _FakeLLMClient([{}])
        responder = TradeResponder(client, mcts_value_threshold=0.50)
        # 0.20 delta < 0.50 threshold
        assert responder._fast_path_check(0.30, 0.50) is None
        # 0.60 delta > 0.50 threshold
        assert responder._fast_path_check(0.10, 0.71) == "accept"


# ---------------------------------------------------------------------------
# respond_to_trade
# ---------------------------------------------------------------------------


class TestRespondToTrade:
    def test_fast_path_skips_llm(self) -> None:
        client = _FakeLLMClient([{"should": "not be called"}])
        responder = TradeResponder(client, mcts_value_threshold=0.10)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(
            game,
            player_id=0,
            trade=trade,
            mcts_value_before=0.30,
            mcts_value_after=0.50,
        )
        assert result["decision"] == "accept"
        assert "0.20" in result["reasoning"]
        assert client._call_count == 0  # LLM not called

    def test_llm_accept(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "decision": "accept",
                    "reasoning": "Good trade for me",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(game, player_id=0, trade=trade)
        assert result["decision"] == "accept"
        assert client._call_count == 1

    def test_llm_reject(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "decision": "reject",
                    "reasoning": "Bad deal",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(game, player_id=0, trade=trade)
        assert result["decision"] == "reject"
        assert result["reasoning"] == "Bad deal"

    def test_llm_counter_with_proposal(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "decision": "counter",
                    "reasoning": "I want more",
                    "counter_proposal": {
                        "give_properties": [9],
                        "give_money": 0,
                        "want_properties": [6],
                        "want_money": 100,
                    },
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(game, player_id=0, trade=trade)
        assert result["decision"] == "counter"
        assert "counter_proposal" in result
        cp = result["counter_proposal"]
        assert cp["want_money"] == 100
        assert cp["give_properties"] == [9]

    def test_llm_error_defaults_to_reject(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "error": "Failed to parse JSON",
                    "raw": "garbage",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(game, player_id=0, trade=trade)
        assert result["decision"] == "reject"

    def test_invalid_decision_defaults_to_reject(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "decision": "maybe",
                    "reasoning": "I'm not sure",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(game, player_id=0, trade=trade)
        assert result["decision"] == "reject"

    def test_counter_without_proposal_still_works(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "decision": "counter",
                    "reasoning": "I want more but didn't say what",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.respond_to_trade(game, player_id=0, trade=trade)
        assert result["decision"] == "counter"
        assert "counter_proposal" not in result


# ---------------------------------------------------------------------------
# generate_counter
# ---------------------------------------------------------------------------


class TestGenerateCounter:
    def test_counter_response(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "action": "counter",
                    "reasoning": "Modified offer",
                    "counter_proposal": {
                        "to_player": 1,
                        "give_properties": [9],
                        "give_money": 50,
                        "want_properties": [6, 8],
                        "want_money": 0,
                    },
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.generate_counter(
            game,
            player_id=0,
            previous_trade=trade,
            rejection_reason="Not enough value",
        )
        assert result["action"] == "counter"
        cp = result["counter_proposal"]
        assert cp["to_player"] == 1
        assert cp["give_money"] == 50
        assert cp["want_properties"] == [6, 8]

    def test_stop_response(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "action": "stop",
                    "reasoning": "No good counter available",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.generate_counter(
            game,
            player_id=0,
            previous_trade=trade,
            rejection_reason="Bad deal",
        )
        assert result["action"] == "stop"
        assert "counter_proposal" not in result

    def test_error_defaults_to_stop(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "error": "Failed to parse JSON",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.generate_counter(
            game,
            player_id=0,
            previous_trade=trade,
            rejection_reason="Nope",
        )
        assert result["action"] == "stop"

    def test_invalid_action_defaults_to_stop(self) -> None:
        client = _FakeLLMClient(
            [
                {
                    "action": "negotiate",
                    "reasoning": "Let's talk more",
                }
            ]
        )
        responder = TradeResponder(client)
        game = _make_game()
        trade = _make_trade()

        result = responder.generate_counter(
            game,
            player_id=0,
            previous_trade=trade,
            rejection_reason="Bad",
        )
        assert result["action"] == "stop"


# ---------------------------------------------------------------------------
# _normalize_counter
# ---------------------------------------------------------------------------


class TestNormalizeCounter:
    def test_basic_normalization(self) -> None:
        cp = {
            "to_player": "2",
            "give_properties": ["1", "3"],
            "give_money": "100",
            "want_properties": ["6"],
            "want_money": "0",
        }
        result = TradeResponder._normalize_counter(cp)
        assert result["to_player"] == 2
        assert result["give_properties"] == [1, 3]
        assert result["give_money"] == 100

    def test_defaults_for_missing_fields(self) -> None:
        cp: dict[str, Any] = {"to_player": 1}
        result = TradeResponder._normalize_counter(cp)
        assert result["give_properties"] == []
        assert result["give_money"] == 0
        assert result["want_properties"] == []
        assert result["want_money"] == 0

    def test_non_dict_returns_empty(self) -> None:
        assert TradeResponder._normalize_counter("not a dict") == {}
        assert TradeResponder._normalize_counter(42) == {}

    def test_invalid_types_return_empty(self) -> None:
        cp = {"to_player": "not_a_number"}
        result = TradeResponder._normalize_counter(cp)
        assert result == {}

    def test_player_id_fallback(self) -> None:
        cp: dict[str, Any] = {"give_properties": [9]}
        result = TradeResponder._normalize_counter(cp, player_id=0)
        assert result["to_player"] == 0
