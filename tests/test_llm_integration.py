"""Optional integration test for LLM trading pipeline.

Requires a real LLM backend. Skipped unless the environment variable
``MONOPOLY_LLM_INTEGRATION_TEST=1`` is set.

Usage:
    MONOPOLY_LLM_INTEGRATION_TEST=1 MONOPOLY_LLM_PROVIDER=ollama \
        uv run python -m pytest tests/test_llm_integration.py -v

Set ``MONOPOLY_LLM_PROVIDER`` to choose the backend (default: ollama).
For Claude/OpenAI, the appropriate API key env var must also be set.
"""

from __future__ import annotations

import os

import pytest

from mcts.llm.budget import ResponseCache, TokenBudget
from mcts.llm.client import LLMConfig, create_client
from mcts.llm.trade_generator import TradeGenerator
from mcts.llm.trade_responder import TradeResponder
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import TradeOfferData

_SKIP_REASON = (
    "Set MONOPOLY_LLM_INTEGRATION_TEST=1 to run live LLM tests"
)
_ENABLED = os.environ.get("MONOPOLY_LLM_INTEGRATION_TEST") == "1"

pytestmark = pytest.mark.skipif(not _ENABLED, reason=_SKIP_REASON)


def _make_game() -> MonopolyGame:
    """Create a game with property ownership suitable for trade proposals."""
    game = MonopolyGame(num_players=3, seed=42)
    pm = game.property_manager

    # Player 0: owns Brown monopoly + Connecticut (blocks P1's light blue)
    pm.properties[1].owner = 0
    pm.properties[3].owner = 0
    pm.properties[9].owner = 0

    # Player 1: owns Oriental + Vermont (needs Connecticut for light blue)
    pm.properties[6].owner = 1
    pm.properties[8].owner = 1

    # Player 1 also owns Reading Railroad (non-critical, can offer in trades)
    pm.properties[5].owner = 1

    # Player 2: owns St. Charles + States (needs Virginia for magenta)
    pm.properties[11].owner = 2
    pm.properties[13].owner = 2

    return game


class TestLLMIntegration:
    """Live integration tests against a real LLM backend."""

    def setup_method(self) -> None:
        provider = os.environ.get("MONOPOLY_LLM_PROVIDER", "ollama")
        self.config = LLMConfig(
            provider=provider,
            timeout_seconds=30.0,
            max_retries=1,
        )
        self.client = create_client(self.config)

    def teardown_method(self) -> None:
        self.client.close()

    def test_generate_trade_proposal(self) -> None:
        """LLM can generate a valid trade proposal or no_trade."""
        game = _make_game()
        game.state.current_player = 0

        gen = TradeGenerator(self.client)
        result = gen.generate_proposal(game, player_id=0)

        # Result is either a valid ProposeTrade or None (no_trade / validation fail)
        # We just verify the pipeline doesn't crash
        if result is not None:
            assert result.player_id == 0
            assert result.to_player in (1, 2)

    def test_respond_to_trade(self) -> None:
        """LLM can evaluate and respond to an incoming trade."""
        game = _make_game()

        trade = TradeOfferData(
            from_player=1,
            to_player=0,
            give_properties=[6],
            give_money=50,
            want_properties=[9],
            want_money=0,
        )

        responder = TradeResponder(self.client)
        result = responder.respond_to_trade(
            game, player_id=0, trade=trade,
        )

        assert "decision" in result
        assert result["decision"] in ("accept", "reject", "counter")
        assert "reasoning" in result

    def test_budget_tracking(self) -> None:
        """Budget tracks calls correctly during live usage."""
        budget = TokenBudget(max_calls_per_game=5)

        game = _make_game()
        game.state.current_player = 0

        gen = TradeGenerator(self.client)

        # Make a call and track it
        assert budget.budget_remaining is True
        gen.generate_proposal(game, player_id=0)
        budget.record_usage(tokens=200)

        assert budget.calls_used == 1
        assert budget.tokens_used == 200

    def test_cache_hit_avoids_redundant_call(self) -> None:
        """Response cache returns cached result for identical state."""
        cache = ResponseCache()
        game = _make_game()

        state_hash = ResponseCache.hash_game_state(game, player_id=0)
        prompt_hash = ResponseCache.hash_prompt("test prompt")

        # Miss on first lookup
        assert cache.get(state_hash, prompt_hash) is None

        # Store a response
        response = {"decision": "accept", "reasoning": "cached"}
        cache.put(state_hash, prompt_hash, response)

        # Hit on second lookup
        assert cache.get(state_hash, prompt_hash) == response

    def test_end_to_end_negotiation_pipeline(self) -> None:
        """Full pipeline: generate proposal -> evaluate -> respond."""
        game = _make_game()
        game.state.current_player = 0

        gen = TradeGenerator(self.client)
        proposal = gen.generate_proposal(game, player_id=0)

        if proposal is None:
            pytest.skip("LLM returned no_trade, can't test response flow")

        # Now have player 1 or 2 respond
        trade = TradeOfferData(
            from_player=0,
            to_player=proposal.to_player,
            give_properties=proposal.give_properties,
            give_money=proposal.give_money,
            want_properties=proposal.want_properties,
            want_money=proposal.want_money,
        )

        responder = TradeResponder(self.client)
        result = responder.respond_to_trade(
            game, player_id=proposal.to_player, trade=trade,
        )

        assert result["decision"] in ("accept", "reject", "counter")
