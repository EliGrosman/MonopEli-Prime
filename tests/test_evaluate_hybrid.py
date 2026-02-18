"""Tests for the hybrid agent evaluation script."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_hybrid import (
    EvalResults,
    GameResult,
    _create_opponent,
    _EvalLLMClient,
    play_eval_game,
    run_evaluation,
)

from agents.hybrid_agent import HybridAgent, HybridAgentConfig
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from mcts.llm.budget import TokenBudget
from mcts.negotiation import NegotiationManager

# ---------------------------------------------------------------------------
# _EvalLLMClient
# ---------------------------------------------------------------------------

class TestEvalLLMClient:
    def test_complete_json_returns_no_trade(self) -> None:
        client = _EvalLLMClient()
        result = client.complete_json("sys", "propose a trade")
        assert result.get("no_trade") is True

    def test_complete_json_returns_reject_for_eval(self) -> None:
        client = _EvalLLMClient()
        result = client.complete_json("sys", "evaluate this trade")
        assert result.get("decision") == "reject"

    def test_tracks_calls(self) -> None:
        client = _EvalLLMClient()
        client.complete_json("sys", "test")
        client.complete_json("sys", "test2")
        assert client.call_count == 2
        assert client.total_tokens == 100


# ---------------------------------------------------------------------------
# _create_opponent
# ---------------------------------------------------------------------------

class TestCreateOpponent:
    def test_creates_random(self) -> None:
        agent = _create_opponent("random", 1)
        assert isinstance(agent, RandomAgent)
        assert agent.player_id == 1

    def test_creates_rule_based(self) -> None:
        agent = _create_opponent("rule_based", 2)
        assert isinstance(agent, RuleBasedAgent)
        assert agent.player_id == 2

    def test_unknown_raises(self) -> None:
        try:
            _create_opponent("unknown_type", 1)
            assert False, "Expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# EvalResults
# ---------------------------------------------------------------------------

class TestEvalResults:
    def test_empty_results(self) -> None:
        r = EvalResults(agent_name="test", opponent_name="random")
        assert r.win_rate == 0.0
        assert r.avg_trades_proposed == 0.0
        assert r.avg_game_length == 0.0
        assert r.avg_tokens_used == 0.0
        assert r.trade_acceptance_rate == 0.0

    def test_computed_properties(self) -> None:
        r = EvalResults(
            agent_name="test",
            opponent_name="random",
            games_played=10,
            wins=7,
            losses=2,
            draws=1,
            total_trades_proposed=20,
            total_trades_accepted=5,
            total_actions=500,
            total_tokens=1000,
        )
        assert r.win_rate == 0.7
        assert r.avg_trades_proposed == 2.0
        assert r.avg_trades_accepted == 0.5
        assert r.trade_acceptance_rate == 0.25
        assert r.avg_game_length == 50.0
        assert r.avg_tokens_used == 100.0


# ---------------------------------------------------------------------------
# play_eval_game
# ---------------------------------------------------------------------------

class TestPlayEvalGame:
    def test_runs_single_game(self) -> None:
        """Verify a single game runs to completion."""
        llm_client = _EvalLLMClient()
        config = HybridAgentConfig(
            mcts_simulations=10,
            trade_eval_simulations=0,
            trade_check_interval=999,  # Disable trading for speed
            token_budget=TokenBudget(max_calls_per_game=0),
        )
        mgr = NegotiationManager()
        agent = HybridAgent(
            player_id=0, config=config,
            negotiation_manager=mgr, llm_client=llm_client,
        )
        opponents = [RandomAgent(player_id=1)]

        result = play_eval_game(
            agent, opponents, llm_client,
            max_turns=200, seed=42,
        )

        assert isinstance(result, GameResult)
        assert result.action_count > 0
        assert result.action_count <= 200
        assert result.elapsed_sec >= 0

    def test_tracks_trade_metrics(self) -> None:
        """Verify trade metrics are captured."""
        llm_client = _EvalLLMClient()
        config = HybridAgentConfig(
            mcts_simulations=10,
            trade_eval_simulations=0,
            trade_check_interval=999,
            token_budget=TokenBudget(max_calls_per_game=0),
        )
        mgr = NegotiationManager()
        agent = HybridAgent(
            player_id=0, config=config,
            negotiation_manager=mgr, llm_client=llm_client,
        )
        opponents = [RandomAgent(player_id=1)]

        result = play_eval_game(
            agent, opponents, llm_client,
            max_turns=100, seed=42,
        )

        # With trading disabled, should have 0 trades
        assert result.trades_proposed == 0
        assert result.trades_accepted == 0


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_runs_multiple_games(self) -> None:
        """Verify evaluation runs multiple games and aggregates."""
        results = run_evaluation(
            opponent_type="random",
            num_games=3,
            num_players=2,
            mcts_simulations=10,
            max_turns=100,
            seed=42,
            verbose=False,
        )

        assert results.games_played == 3
        assert results.wins + results.losses + results.draws == 3
        assert len(results.game_results) == 3
        assert results.agent_name == "HybridAgent"
        assert results.opponent_name == "random"
        assert results.total_time_sec > 0
