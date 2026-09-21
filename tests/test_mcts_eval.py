"""Tests for MCTS evaluation harness (C2)."""

from __future__ import annotations

import pytest

from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from agents.rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent
from mcts.eval import (
    MCTSEvalResult,
    _make_opponent,
    evaluate_mcts_agent,
    play_evaluation_game,
)


# ===========================================================================
# C2-1: MCTSEvalResult
# ===========================================================================


class TestMCTSEvalResult:
    """Test MCTSEvalResult dataclass."""

    def _make_result(self, wins: int = 3, losses: int = 5, draws: int = 2) -> MCTSEvalResult:
        total = wins + losses + draws
        return MCTSEvalResult(
            opponent_type="random",
            num_games=total,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / total,
            avg_game_length=150.0,
            avg_time_sec=5.0,
            total_time_sec=50.0,
            avg_mcts_time_per_move=0.02,
        )

    def test_str_contains_win_rate(self) -> None:
        """__str__ should include opponent type and win rate."""
        result = self._make_result()
        s = str(result)
        assert "random" in s
        assert "W" in s  # wins/losses/draws summary

    def test_to_dict_has_expected_keys(self) -> None:
        """to_dict should return all required metric keys."""
        result = self._make_result()
        d = result.to_dict()

        expected_keys = {
            "win_rate", "avg_game_length", "avg_time_sec",
            "avg_mcts_time_per_move", "wins", "losses", "draws", "num_games",
        }
        assert expected_keys <= set(d.keys())

    def test_to_dict_win_rate_value(self) -> None:
        """to_dict win_rate should match the dataclass field."""
        result = self._make_result(wins=4, losses=6, draws=0)
        assert result.to_dict()["win_rate"] == pytest.approx(0.4)


# ===========================================================================
# C2-2: _make_opponent factory
# ===========================================================================


class TestMakeOpponent:
    """Test opponent agent factory."""

    def test_random_opponent(self) -> None:
        opp = _make_opponent("random", player_id=1)
        assert isinstance(opp, RandomAgent)
        assert opp.player_id == 1

    def test_rule_based_opponent(self) -> None:
        opp = _make_opponent("rule_based", player_id=2)
        assert isinstance(opp, RuleBasedAgent)

    def test_aggressive_opponent(self) -> None:
        opp = _make_opponent("aggressive", player_id=3)
        assert isinstance(opp, AggressiveAgent)

    def test_conservative_opponent(self) -> None:
        opp = _make_opponent("conservative", player_id=1)
        assert isinstance(opp, ConservativeAgent)

    def test_unknown_opponent_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown opponent"):
            _make_opponent("gpt4", player_id=1)


# ===========================================================================
# C2-3: play_evaluation_game
# ===========================================================================


class TestPlayEvaluationGame:
    """Test single evaluation game runner."""

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_returns_three_tuple(self) -> None:
        """play_evaluation_game should return (winner, action_count, elapsed)."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RandomAgent(player_id=1), RandomAgent(player_id=2),
                     RandomAgent(player_id=3)]

        result = play_evaluation_game(mcts, opponents, max_turns=20, seed=42)

        assert len(result) == 3

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_action_count_bounded(self) -> None:
        """Action count should not exceed max_turns."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RandomAgent(player_id=1), RandomAgent(player_id=2),
                     RandomAgent(player_id=3)]
        max_turns = 15

        _, action_count, _ = play_evaluation_game(
            mcts, opponents, max_turns=max_turns, seed=7
        )

        assert action_count <= max_turns

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_winner_is_valid_player_or_none(self) -> None:
        """Winner should be a player ID (0-3) or None."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RandomAgent(player_id=i + 1) for i in range(3)]

        winner, _, _ = play_evaluation_game(
            mcts, opponents, max_turns=30, seed=99
        )

        assert winner is None or 0 <= winner <= 3

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_elapsed_is_positive(self) -> None:
        """Elapsed time should be positive."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RandomAgent(player_id=1), RandomAgent(player_id=2),
                     RandomAgent(player_id=3)]

        _, _, elapsed = play_evaluation_game(
            mcts, opponents, max_turns=20, seed=42
        )

        assert elapsed > 0.0

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_resets_agent_between_calls(self) -> None:
        """Each game call should reset the MCTS agent stats."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RandomAgent(player_id=i + 1) for i in range(3)]

        play_evaluation_game(mcts, opponents, max_turns=10, seed=1)
        stats_after_first = mcts.search_stats.total_searches

        play_evaluation_game(mcts, opponents, max_turns=10, seed=2)
        stats_after_second = mcts.search_stats.total_searches

        # Second game resets stats, so second game's count should equal first
        assert stats_after_second == stats_after_first

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_two_player_game(self) -> None:
        """Should work with 2-player game (MCTS vs 1 opponent)."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RandomAgent(player_id=1)]

        winner, _, _ = play_evaluation_game(
            mcts, opponents, max_turns=20, seed=42
        )

        assert winner is None or winner in (0, 1)


# ===========================================================================
# C2-4: evaluate_mcts_agent
# ===========================================================================


class TestEvaluateMCTSAgent:
    """Test the full evaluation harness."""

    def _fast_kwargs(self) -> dict:
        return dict(
            network_path=None,
            num_simulations=5,
            num_games=2,
            num_players=4,
            max_turns=20,
            seed=42,
            verbose=False,
        )

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_returns_dict(self) -> None:
        """evaluate_mcts_agent should return a dict."""
        result = evaluate_mcts_agent(
            opponents=["random"],
            **self._fast_kwargs(),
        )
        assert isinstance(result, dict)

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_result_has_opponent_keys(self) -> None:
        """Result dict should have one key per opponent type."""
        result = evaluate_mcts_agent(
            opponents=["random", "rule_based"],
            **self._fast_kwargs(),
        )
        assert "random" in result
        assert "rule_based" in result

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_default_opponents(self) -> None:
        """Default opponents should be random and rule_based."""
        result = evaluate_mcts_agent(**self._fast_kwargs())
        assert "random" in result
        assert "rule_based" in result

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_each_result_has_metric_keys(self) -> None:
        """Each opponent dict should have expected numeric keys."""
        result = evaluate_mcts_agent(
            opponents=["random"],
            **self._fast_kwargs(),
        )
        metrics = result["random"]
        for key in ("win_rate", "avg_game_length", "avg_time_sec", "num_games"):
            assert key in metrics, f"Missing key: {key}"

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_win_rate_in_valid_range(self) -> None:
        """win_rate should be in [0, 1]."""
        result = evaluate_mcts_agent(
            opponents=["random"],
            **self._fast_kwargs(),
        )
        wr = result["random"]["win_rate"]
        assert 0.0 <= wr <= 1.0

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_all_four_opponent_types(self) -> None:
        """Should work with all four supported opponent types."""
        result = evaluate_mcts_agent(
            opponents=["random", "rule_based", "aggressive", "conservative"],
            **self._fast_kwargs(),
        )
        assert len(result) == 4

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_evaluation_runs_to_completion(self) -> None:
        """Should run without errors for a complete eval."""
        # No assertion beyond "doesn't raise"
        evaluate_mcts_agent(
            opponents=["random"],
            **self._fast_kwargs(),
        )

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_tensorboard_logging(self, tmp_path: pytest.TempPathFactory) -> None:
        """When tensorboard_dir is given, event files should be created."""
        tb_dir = tmp_path / "tb"  # type: ignore[operator]
        evaluate_mcts_agent(
            opponents=["random"],
            tensorboard_dir=tb_dir,
            **self._fast_kwargs(),
        )
        assert tb_dir.exists()  # type: ignore[union-attr]
        events = list(tb_dir.glob("events.out.*"))  # type: ignore[union-attr]
        assert len(events) > 0
