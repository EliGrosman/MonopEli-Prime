"""Tests for LLM trade proposal generator."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from mcts.llm.client import LLMClient, LLMConfig
from mcts.llm.trade_generator import (
    TradeGenerator,
    _format_trade_candidates,
    _format_trade_options,
)
from mcts.trade_utils import TradeCandidate
from monopoly_engine.game import MonopolyGame

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


def _make_game(num_players: int = 3) -> MonopolyGame:
    """Create a game with some property ownership for testing."""
    game = MonopolyGame(num_players=num_players, seed=42)
    pm = game.property_manager

    # Player 0 owns Mediterranean [1] and Baltic [3] (Brown complete)
    pm.properties[1].owner = 0
    pm.properties[3].owner = 0

    # Player 1 owns Oriental [6] and Vermont [8] (Light Blue, missing CT [9])
    pm.properties[6].owner = 1
    pm.properties[8].owner = 1

    # Player 0 owns Connecticut [9] (blocks Player 1's light blue monopoly)
    pm.properties[9].owner = 0

    # Player 2 owns St. Charles [11] (Magenta, missing others)
    pm.properties[11].owner = 2

    return game


# ---------------------------------------------------------------------------
# _format_trade_candidates
# ---------------------------------------------------------------------------

class TestFormatTradeCandidates:
    def test_empty_list(self) -> None:
        assert _format_trade_candidates([]) == ""

    def test_single_candidate(self) -> None:
        candidates = [
            TradeCandidate(
                to_player=1,
                give_properties=[9],
                want_properties=[6],
                give_money=0,
                want_money=0,
                estimated_value=250.0,
            ),
        ]
        result = _format_trade_candidates(candidates)
        assert "Player 1" in result
        assert "[9]" in result
        assert "[6]" in result
        assert "250" in result

    def test_candidate_with_cash(self) -> None:
        candidates = [
            TradeCandidate(
                to_player=2,
                give_properties=[3],
                want_properties=[11],
                give_money=100,
                want_money=50,
                estimated_value=180.0,
            ),
        ]
        result = _format_trade_candidates(candidates)
        assert "$100" in result
        assert "$50" in result

    def test_multiple_candidates_numbered(self) -> None:
        candidates = [
            TradeCandidate(
                to_player=1, give_properties=[9], want_properties=[6],
                give_money=0, want_money=0, estimated_value=300.0,
            ),
            TradeCandidate(
                to_player=2, give_properties=[3], want_properties=[11],
                give_money=50, want_money=0, estimated_value=100.0,
            ),
        ]
        result = _format_trade_candidates(candidates)
        assert "1." in result
        assert "2." in result


# ---------------------------------------------------------------------------
# TradeGenerator._parse_proposal
# ---------------------------------------------------------------------------

class TestParseProposal:
    def setup_method(self) -> None:
        self.gen = TradeGenerator(_FakeLLMClient([{}]))

    def test_valid_proposal(self) -> None:
        response = {
            "to_player": 1,
            "give_properties": [9],
            "give_money": 50,
            "want_properties": [6],
            "want_money": 0,
            "reasoning": "Complete light blue",
        }
        result = self.gen._parse_proposal(response, player_id=0)
        assert result is not None
        assert result.player_id == 0
        assert result.to_player == 1
        assert result.give_properties == [9]
        assert result.give_money == 50
        assert result.want_properties == [6]
        assert result.want_money == 0

    def test_missing_to_player_returns_none(self) -> None:
        response = {"give_properties": [9], "want_properties": [6]}
        assert self.gen._parse_proposal(response, player_id=0) is None

    def test_invalid_type_returns_none(self) -> None:
        response = {"to_player": "not_a_number"}
        assert self.gen._parse_proposal(response, player_id=0) is None

    def test_defaults_for_optional_fields(self) -> None:
        response = {"to_player": 2}
        result = self.gen._parse_proposal(response, player_id=0)
        assert result is not None
        assert result.give_properties == []
        assert result.give_money == 0
        assert result.want_properties == []
        assert result.want_money == 0

    def test_string_numbers_coerced(self) -> None:
        response = {
            "to_player": "1",
            "give_properties": ["9"],
            "give_money": "50",
            "want_properties": ["6"],
            "want_money": "0",
        }
        result = self.gen._parse_proposal(response, player_id=0)
        assert result is not None
        assert result.to_player == 1
        assert result.give_properties == [9]
        assert result.give_money == 50


# ---------------------------------------------------------------------------
# TradeGenerator.generate_proposal
# ---------------------------------------------------------------------------

class TestGenerateProposal:
    def test_successful_proposal(self) -> None:
        game = _make_game()
        # Ensure it's player 0's turn
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "to_player": 1,
            "give_properties": [9],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
            "reasoning": "Trade Connecticut for Oriental",
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)

        assert result is not None
        assert result.to_player == 1
        assert result.give_properties == [9]
        assert result.want_properties == [6]
        assert client._call_count == 1

    def test_no_trade_response(self) -> None:
        game = _make_game()
        client = _FakeLLMClient([{
            "no_trade": True,
            "reasoning": "No beneficial trades available",
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None

    def test_llm_error_returns_none(self) -> None:
        game = _make_game()
        client = _FakeLLMClient([{
            "error": "Failed to parse JSON",
            "raw": "garbage",
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None

    def test_invalid_trade_rejected_by_validation(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        # Player 0 doesn't own position 11, so this trade should fail validation
        client = _FakeLLMClient([{
            "to_player": 1,
            "give_properties": [11],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None

    def test_trade_with_self_rejected(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "to_player": 0,
            "give_properties": [1],
            "want_properties": [3],
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None

    def test_custom_trade_context(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "to_player": 1,
            "give_properties": [9],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(
            game, player_id=0, trade_context="Custom context here",
        )
        assert result is not None

    def test_insufficient_funds_rejected(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        # Offer more money than player has
        client = _FakeLLMClient([{
            "to_player": 1,
            "give_properties": [],
            "give_money": 99999,
            "want_properties": [6],
            "want_money": 0,
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None

    def test_unparseable_response_returns_none(self) -> None:
        game = _make_game()
        # Missing to_player entirely
        client = _FakeLLMClient([{
            "reasoning": "I think we should trade",
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None

    def test_auto_generates_context_from_trade_utils(self) -> None:
        """Verify suggest_valuable_trades is called when no context provided."""
        game = _make_game()
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "no_trade": True,
            "reasoning": "Nothing good",
        }])
        gen = TradeGenerator(client)

        with patch("mcts.llm.trade_generator.suggest_valuable_trades") as mock_suggest:
            mock_suggest.return_value = []
            gen.generate_proposal(game, player_id=0)
            mock_suggest.assert_called_once_with(game, 0)

    def test_skips_auto_context_when_provided(self) -> None:
        """Verify suggest_valuable_trades is NOT called when context provided."""
        game = _make_game()
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "no_trade": True,
            "reasoning": "Nothing good",
        }])
        gen = TradeGenerator(client)

        with patch("mcts.llm.trade_generator.suggest_valuable_trades") as mock_suggest:
            gen.generate_proposal(
                game, player_id=0, trade_context="Pre-built context",
            )
            mock_suggest.assert_not_called()

    def test_trade_with_bankrupt_player_rejected(self) -> None:
        game = _make_game()
        game.state.current_player = 0
        game.players[2].bankrupt = True

        client = _FakeLLMClient([{
            "to_player": 2,
            "give_properties": [1],
            "give_money": 0,
            "want_properties": [],
            "want_money": 0,
        }])
        gen = TradeGenerator(client)
        result = gen.generate_proposal(game, player_id=0)
        assert result is None


# ---------------------------------------------------------------------------
# _format_trade_options
# ---------------------------------------------------------------------------

class TestFormatTradeOptions:
    def test_numbered_list(self) -> None:
        candidates = [
            TradeCandidate(
                to_player=1, give_properties=[9], want_properties=[6],
                give_money=0, want_money=0, estimated_value=300.0,
            ),
            TradeCandidate(
                to_player=2, give_properties=[3], want_properties=[11],
                give_money=50, want_money=0, estimated_value=100.0,
            ),
        ]
        result = _format_trade_options(candidates)
        assert result.startswith("1.")
        assert "2." in result
        assert "0. None" in result

    def test_includes_cash(self) -> None:
        candidates = [
            TradeCandidate(
                to_player=1, give_properties=[9], want_properties=[6],
                give_money=100, want_money=50, estimated_value=200.0,
            ),
        ]
        result = _format_trade_options(candidates)
        assert "$100" in result
        assert "$50" in result


# ---------------------------------------------------------------------------
# TradeGenerator.generate_proposal_from_candidates
# ---------------------------------------------------------------------------

class TestGenerateProposalFromCandidates:
    def test_selects_valid_candidate(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "choice": 1,
            "reasoning": "Completing light blue is valuable",
        }])
        gen = TradeGenerator(client)

        with patch("mcts.llm.trade_generator.suggest_valuable_trades") as mock_suggest:
            mock_suggest.return_value = [
                TradeCandidate(
                    to_player=1, give_properties=[9], want_properties=[6],
                    give_money=0, want_money=0, estimated_value=300.0,
                ),
            ]
            result = gen.generate_proposal_from_candidates(game, player_id=0)

        assert result is not None
        assert result.player_id == 0
        assert result.to_player == 1
        assert result.give_properties == [9]
        assert result.want_properties == [6]
        assert client._call_count == 1

    def test_choice_zero_returns_none(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        client = _FakeLLMClient([{
            "choice": 0,
            "reasoning": "None of these are good",
        }])
        gen = TradeGenerator(client)

        with patch("mcts.llm.trade_generator.suggest_valuable_trades") as mock_suggest:
            mock_suggest.return_value = [
                TradeCandidate(
                    to_player=1, give_properties=[9], want_properties=[6],
                    give_money=0, want_money=0, estimated_value=300.0,
                ),
            ]
            result = gen.generate_proposal_from_candidates(game, player_id=0)

        assert result is None

    def test_no_candidates_returns_none_without_llm_call(self) -> None:
        game = _make_game()

        client = _FakeLLMClient([{"choice": 1, "reasoning": "pick"}])
        gen = TradeGenerator(client)

        with patch("mcts.llm.trade_generator.suggest_valuable_trades") as mock_suggest:
            mock_suggest.return_value = []
            result = gen.generate_proposal_from_candidates(game, player_id=0)

        assert result is None
        assert client._call_count == 0

    def test_invalid_choice_returns_none(self) -> None:
        game = _make_game()
        game.state.current_player = 0

        # Choice 5 is out of range when there's only 1 candidate
        client = _FakeLLMClient([{
            "choice": 5,
            "reasoning": "I pick option 5",
        }])
        gen = TradeGenerator(client)

        with patch("mcts.llm.trade_generator.suggest_valuable_trades") as mock_suggest:
            mock_suggest.return_value = [
                TradeCandidate(
                    to_player=1, give_properties=[9], want_properties=[6],
                    give_money=0, want_money=0, estimated_value=300.0,
                ),
            ]
            result = gen.generate_proposal_from_candidates(game, player_id=0)

        assert result is None
