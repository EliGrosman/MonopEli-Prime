"""Tests for HybridAgent (MCTS + LLM trading)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from agents.hybrid_agent import HybridAgent, HybridAgentConfig
from mcts.llm.budget import TokenBudget
from mcts.llm.client import LLMConfig
from mcts.negotiation import NegotiationManager, NegotiationRecord, NegotiationStatus
from mcts.trade_verifier import TradeEvaluation
from monopoly_engine.actions import ProposeTrade
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import TradeOfferData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game() -> MonopolyGame:
    """Create a game with property ownership for trade tests."""
    game = MonopolyGame(num_players=3, seed=42)
    pm = game.property_manager

    # Player 0: owns Mediterranean [1], Baltic [3] (Brown monopoly)
    pm.properties[1].owner = 0
    pm.properties[3].owner = 0

    # Player 1: owns Oriental [6], Vermont [8] (missing CT [9])
    pm.properties[6].owner = 1
    pm.properties[8].owner = 1

    # Player 0 also owns Connecticut [9]
    pm.properties[9].owner = 0

    return game


def _make_action_mask() -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Create a dummy action mask with at least one valid action."""
    mask = np.zeros(149, dtype=np.bool_)
    mask[0] = True  # EndTurn always valid
    return mask


def _make_observation() -> dict[str, Any]:
    """Create a dummy observation dict."""
    return {"player_id": 0}


class _FakeLLMClient:
    """Minimal LLM client stub for tests."""

    def __init__(self) -> None:
        self.config = LLMConfig()
        self.responses: list[dict[str, Any]] = []
        self._call_idx = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "{}"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return {"error": "no more responses"}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HybridAgentConfig
# ---------------------------------------------------------------------------

class TestHybridAgentConfig:
    def test_default_config(self) -> None:
        config = HybridAgentConfig()
        assert config.mcts_simulations == 200
        assert config.trade_eval_simulations == 50
        assert config.trade_acceptance_threshold == 0.02
        assert config.max_negotiation_rounds == 3
        assert config.max_trades_per_game == 10
        assert config.trade_check_interval == 5
        assert config.fast_accept_threshold == 0.10

    def test_custom_config(self) -> None:
        config = HybridAgentConfig(
            mcts_simulations=50,
            trade_eval_simulations=10,
            max_trades_per_game=5,
        )
        assert config.mcts_simulations == 50
        assert config.trade_eval_simulations == 10
        assert config.max_trades_per_game == 5


# ---------------------------------------------------------------------------
# HybridAgent.__init__
# ---------------------------------------------------------------------------

class TestHybridAgentInit:
    def test_default_init(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        assert agent.player_id == 0
        assert agent.name == "HybridAgent(p0)"
        assert agent.trades_proposed == 0

    def test_custom_config(self) -> None:
        config = HybridAgentConfig(mcts_simulations=50)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=1, config=config, llm_client=client)
        assert agent._mcts_agent.num_simulations == 50

    def test_shared_negotiation_manager(self) -> None:
        mgr = NegotiationManager(max_rounds=5)
        client = _FakeLLMClient()
        agent = HybridAgent(
            player_id=0, negotiation_manager=mgr, llm_client=client,
        )
        assert agent.negotiation_manager is mgr

    def test_creates_negotiation_manager_if_none(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        assert agent.negotiation_manager is not None


# ---------------------------------------------------------------------------
# choose_action (delegates to MCTSAgent)
# ---------------------------------------------------------------------------

class TestChooseAction:
    def test_delegates_to_mcts_agent(self) -> None:
        """When no trade conditions, delegates to MCTSAgent."""
        client = _FakeLLMClient()
        config = HybridAgentConfig(trade_check_interval=999)
        agent = HybridAgent(player_id=0, config=config, llm_client=client)

        game = _make_game()
        obs = _make_observation()
        mask = _make_action_mask()

        # Mock MCTSAgent to return a known action
        agent._mcts_agent.choose_action = MagicMock(return_value=42)

        action = agent.choose_action(obs, mask, game)
        assert action == 42
        agent._mcts_agent.choose_action.assert_called_once_with(obs, mask, game)

    def test_handles_trade_before_mcts(self) -> None:
        """Trade phase runs before MCTS delegation."""
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)

        game = _make_game()
        obs = _make_observation()
        mask = _make_action_mask()

        trade_phase_called = False

        def mock_handle(g: Any) -> None:
            nonlocal trade_phase_called
            trade_phase_called = True

        agent._handle_trade_phase = mock_handle  # type: ignore[assignment]
        agent._mcts_agent.choose_action = MagicMock(return_value=0)

        agent.choose_action(obs, mask, game)
        assert trade_phase_called


# ---------------------------------------------------------------------------
# _should_attempt_trade
# ---------------------------------------------------------------------------

class TestShouldAttemptTrade:
    def test_respects_budget(self) -> None:
        budget = TokenBudget(max_calls_per_game=0)
        config = HybridAgentConfig(token_budget=budget)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)

        game = _make_game()
        assert agent._should_attempt_trade(game) is False

    def test_respects_max_trades(self) -> None:
        config = HybridAgentConfig(max_trades_per_game=0)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)

        game = _make_game()
        assert agent._should_attempt_trade(game) is False

    def test_respects_interval(self) -> None:
        config = HybridAgentConfig(trade_check_interval=10)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)

        game = _make_game()
        # Default last_trade_turn is -999, and turn_number starts at 0
        # So 0 - (-999) = 999 >= 10, should be True
        assert agent._should_attempt_trade(game) is True

        # Simulate recent trade
        agent._last_trade_turn = game.state.turn_number
        assert agent._should_attempt_trade(game) is False

    def test_allowed_when_conditions_met(self) -> None:
        config = HybridAgentConfig(trade_check_interval=1)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)

        game = _make_game()
        assert agent._should_attempt_trade(game) is True


# ---------------------------------------------------------------------------
# _generate_and_verify_trade
# ---------------------------------------------------------------------------

class TestGenerateAndVerifyTrade:
    def test_successful_trade_proposal(self) -> None:
        """LLM generates, MCTS approves, NegotiationManager executes."""
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()

        # Mock the generator to return a proposal
        proposal = ProposeTrade(
            player_id=0, to_player=1,
            give_properties=[9], want_money=100,
        )
        agent._generator.generate_proposal = MagicMock(return_value=proposal)

        # Mock the verifier to approve
        eval_result = TradeEvaluation(
            trade=TradeOfferData(
                from_player=0, to_player=1,
                give_properties=[9], give_money=0,
                want_properties=[], want_money=100,
            ),
            value_before=0.3, value_after=0.4,
            value_delta=0.1, recommended=True, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)

        result = agent._generate_and_verify_trade(game)
        assert result is True
        assert agent.trades_proposed == 1

    def test_llm_returns_none(self) -> None:
        """LLM decides no trade → no proposal."""
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()

        agent._generator.generate_proposal = MagicMock(return_value=None)

        result = agent._generate_and_verify_trade(game)
        assert result is False
        assert agent.trades_proposed == 0

    def test_mcts_rejects(self) -> None:
        """MCTS says trade is bad → no execution."""
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()

        proposal = ProposeTrade(
            player_id=0, to_player=1,
            give_properties=[1, 3], want_money=0,
        )
        agent._generator.generate_proposal = MagicMock(return_value=proposal)

        eval_result = TradeEvaluation(
            trade=TradeOfferData(
                from_player=0, to_player=1,
                give_properties=[1, 3], give_money=0,
                want_properties=[], want_money=0,
            ),
            value_before=0.4, value_after=0.2,
            value_delta=-0.2, recommended=False, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)

        result = agent._generate_and_verify_trade(game)
        assert result is False
        assert agent.trades_proposed == 0

    def test_engine_validation_failure(self) -> None:
        """NegotiationManager raises ValueError → graceful failure."""
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()

        proposal = ProposeTrade(
            player_id=0, to_player=1,
            give_properties=[9], want_money=100,
        )
        agent._generator.generate_proposal = MagicMock(return_value=proposal)

        eval_result = TradeEvaluation(
            trade=TradeOfferData(
                from_player=0, to_player=1,
                give_properties=[9], give_money=0,
                want_properties=[], want_money=100,
            ),
            value_before=0.3, value_after=0.4,
            value_delta=0.1, recommended=True, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)

        # Mock NegotiationManager to raise
        agent._negotiation_mgr.start_negotiation = MagicMock(
            side_effect=ValueError("Invalid trade"),
        )

        result = agent._generate_and_verify_trade(game)
        assert result is False
        assert agent.trades_proposed == 0

    def test_records_budget_usage(self) -> None:
        """Budget usage is recorded even when LLM returns None."""
        budget = TokenBudget(max_calls_per_game=10)
        config = HybridAgentConfig(token_budget=budget)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)
        game = _make_game()

        agent._generator.generate_proposal = MagicMock(return_value=None)

        agent._generate_and_verify_trade(game)
        assert budget.calls_used == 1


# ---------------------------------------------------------------------------
# _respond_to_negotiation
# ---------------------------------------------------------------------------

class TestRespondToNegotiation:
    def _make_pending_negotiation(
        self,
    ) -> NegotiationRecord:
        """Create a pending negotiation for player 0 to respond to."""
        trade: TradeOfferData = {
            "from_player": 1,
            "to_player": 0,
            "give_properties": [6],
            "give_money": 0,
            "want_properties": [9],
            "want_money": 0,
        }
        return NegotiationRecord(
            negotiation_id=0,
            original_proposer=1,
            current_proposer=1,
            current_responder=0,
            round_number=1,
            max_rounds=3,
            status=NegotiationStatus.PENDING,
            history=[trade],
            current_trade_id=1,
        )

    def test_accept_decision(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()
        neg = self._make_pending_negotiation()

        # Mock MCTS evaluation
        eval_result = TradeEvaluation(
            trade=neg.history[-1],
            value_before=0.3, value_after=0.45,
            value_delta=0.15, recommended=True, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)

        # Mock LLM to accept (fast path: delta > 0.10)
        agent._responder.respond_to_trade = MagicMock(
            return_value={"decision": "accept", "reasoning": "good trade"},
        )

        # Mock NegotiationManager.accept
        agent._negotiation_mgr.accept = MagicMock(return_value=True)

        agent._respond_to_negotiation(game, neg)
        agent._negotiation_mgr.accept.assert_called_once_with(
            game, 0, 0,
        )

    def test_reject_decision(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()
        neg = self._make_pending_negotiation()

        eval_result = TradeEvaluation(
            trade=neg.history[-1],
            value_before=0.4, value_after=0.3,
            value_delta=-0.1, recommended=False, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)
        agent._responder.respond_to_trade = MagicMock(
            return_value={"decision": "reject", "reasoning": "bad deal"},
        )
        agent._negotiation_mgr.reject = MagicMock(return_value=True)

        agent._respond_to_negotiation(game, neg)
        agent._negotiation_mgr.reject.assert_called_once_with(
            game, 0, 0,
        )

    def test_counter_decision(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()
        neg = self._make_pending_negotiation()

        eval_result = TradeEvaluation(
            trade=neg.history[-1],
            value_before=0.3, value_after=0.32,
            value_delta=0.02, recommended=True, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)
        agent._responder.respond_to_trade = MagicMock(
            return_value={
                "decision": "counter",
                "reasoning": "want more",
                "counter_proposal": {
                    "give_properties": [9],
                    "want_properties": [6],
                    "want_money": 50,
                },
            },
        )
        agent._negotiation_mgr.counter_propose = MagicMock(return_value=True)

        agent._respond_to_negotiation(game, neg)
        agent._negotiation_mgr.counter_propose.assert_called_once_with(
            game, 0, 0,
            give_properties=[9],
            give_money=0,
            want_properties=[6],
            want_money=50,
        )

    def test_empty_counter_becomes_reject(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()
        neg = self._make_pending_negotiation()

        eval_result = TradeEvaluation(
            trade=neg.history[-1],
            value_before=0.3, value_after=0.3,
            value_delta=0.0, recommended=False, simulations_used=50,
        )
        agent._verifier.evaluate_trade = MagicMock(return_value=eval_result)
        agent._responder.respond_to_trade = MagicMock(
            return_value={
                "decision": "counter",
                "reasoning": "want more",
                "counter_proposal": {},
            },
        )
        agent._negotiation_mgr.reject = MagicMock(return_value=True)

        agent._respond_to_negotiation(game, neg)
        agent._negotiation_mgr.reject.assert_called_once()

    def test_empty_history_rejects(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()

        neg = NegotiationRecord(
            negotiation_id=0,
            original_proposer=1,
            current_proposer=1,
            current_responder=0,
            status=NegotiationStatus.PENDING,
            history=[],
        )
        agent._negotiation_mgr.reject = MagicMock(return_value=True)

        agent._respond_to_negotiation(game, neg)
        agent._negotiation_mgr.reject.assert_called_once()


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_resets_all_state(self) -> None:
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)

        # Simulate some usage
        agent._trades_proposed = 5
        agent._last_trade_turn = 20
        agent._budget.record_usage(100)

        agent.reset()

        assert agent.trades_proposed == 0
        assert agent._last_trade_turn == -999
        assert agent._budget.calls_used == 0


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

class TestClose:
    def test_closes_owned_client(self) -> None:
        client = _FakeLLMClient()
        client.close = MagicMock()  # type: ignore[method-assign]
        # Pass client but force _owns_client=True to test
        agent = HybridAgent(player_id=0, llm_client=client)
        agent._owns_client = True

        agent.close()
        client.close.assert_called_once()

    def test_does_not_close_external_client(self) -> None:
        client = _FakeLLMClient()
        client.close = MagicMock()  # type: ignore[method-assign]
        agent = HybridAgent(player_id=0, llm_client=client)
        # _owns_client should be False since we passed it in
        assert not agent._owns_client

        agent.close()
        client.close.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: _handle_trade_phase routing
# ---------------------------------------------------------------------------

class TestHandleTradePhase:
    def test_responds_to_pending_first(self) -> None:
        """Pending negotiations take priority over proposing new trades."""
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, llm_client=client)
        game = _make_game()

        trade: TradeOfferData = {
            "from_player": 1, "to_player": 0,
            "give_properties": [6], "give_money": 0,
            "want_properties": [9], "want_money": 0,
        }
        pending = NegotiationRecord(
            negotiation_id=0,
            original_proposer=1,
            current_proposer=1,
            current_responder=0,
            status=NegotiationStatus.PENDING,
            history=[trade],
            current_trade_id=1,
        )

        agent._negotiation_mgr.get_pending_for_player = MagicMock(
            return_value=[pending],
        )
        agent._respond_to_negotiation = MagicMock()  # type: ignore[assignment]
        agent._generate_and_verify_trade = MagicMock()  # type: ignore[assignment]

        agent._handle_trade_phase(game)

        agent._respond_to_negotiation.assert_called_once_with(game, pending)
        agent._generate_and_verify_trade.assert_not_called()

    def test_proposes_when_no_pending(self) -> None:
        """When no pending negotiations, may propose a new trade."""
        config = HybridAgentConfig(trade_check_interval=1)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)
        game = _make_game()

        agent._negotiation_mgr.get_pending_for_player = MagicMock(
            return_value=[],
        )
        agent._generate_and_verify_trade = MagicMock()  # type: ignore[assignment]

        agent._handle_trade_phase(game)
        agent._generate_and_verify_trade.assert_called_once_with(game)

    def test_skips_trade_when_interval_not_met(self) -> None:
        """No proposal when trade interval hasn't elapsed."""
        config = HybridAgentConfig(trade_check_interval=999)
        client = _FakeLLMClient()
        agent = HybridAgent(player_id=0, config=config, llm_client=client)
        game = _make_game()

        agent._negotiation_mgr.get_pending_for_player = MagicMock(
            return_value=[],
        )
        agent._generate_and_verify_trade = MagicMock()  # type: ignore[assignment]

        # Set last trade turn to current turn
        agent._last_trade_turn = game.state.turn_number

        agent._handle_trade_phase(game)
        agent._generate_and_verify_trade.assert_not_called()
