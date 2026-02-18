"""Tests for MCTSAgent (C1)."""

from __future__ import annotations

import numpy as np
import pytest

from agents.base import Agent
from agents.mcts_agent import MCTSAgent, SearchStats
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder


def _make_game(num_players: int = 4, seed: int = 42) -> MonopolyGame:
    """Create a fresh game and auto-roll dice so the first player can act."""
    from mcts.search import _auto_roll_dice

    game = MonopolyGame(num_players=num_players, seed=seed)
    _auto_roll_dice(game)
    return game


def _get_obs_and_mask(game: MonopolyGame, player_id: int) -> tuple[dict, np.ndarray]:
    """Return a dummy observation dict and the real action mask."""
    encoder = ActionEncoder(enable_trades=False)
    mask = encoder.get_action_mask(game, player_id)
    obs: dict = {}
    return obs, mask


# ===========================================================================
# C1-1: SearchStats
# ===========================================================================


class TestSearchStats:
    """Test SearchStats dataclass and computed properties."""

    def test_default_values(self) -> None:
        """Fresh stats should have zero counts."""
        stats = SearchStats()
        assert stats.total_searches == 0
        assert stats.total_time_sec == 0.0
        assert stats.total_simulations == 0

    def test_avg_time_zero_searches(self) -> None:
        """avg_time_per_move should be 0 when no searches done."""
        assert SearchStats().avg_time_per_move == 0.0

    def test_avg_simulations_zero_searches(self) -> None:
        """avg_simulations_per_move should be 0 when no searches done."""
        assert SearchStats().avg_simulations_per_move == 0.0

    def test_avg_time_computation(self) -> None:
        """avg_time_per_move = total_time / total_searches."""
        stats = SearchStats(total_searches=4, total_time_sec=2.0, total_simulations=400)
        assert stats.avg_time_per_move == pytest.approx(0.5)

    def test_avg_simulations_computation(self) -> None:
        """avg_simulations_per_move = total_simulations / total_searches."""
        stats = SearchStats(total_searches=4, total_time_sec=2.0, total_simulations=400)
        assert stats.avg_simulations_per_move == pytest.approx(100.0)


# ===========================================================================
# C1-2: MCTSAgent construction
# ===========================================================================


class TestMCTSAgentInit:
    """Test MCTSAgent initialization."""

    def test_is_agent_subclass(self) -> None:
        """MCTSAgent should implement the Agent ABC."""
        agent = MCTSAgent(player_id=0)
        assert isinstance(agent, Agent)

    def test_default_attributes(self) -> None:
        """Verify default parameter values."""
        agent = MCTSAgent(player_id=2)
        assert agent.player_id == 2
        assert agent.num_simulations == 200
        assert agent.temperature == 0.0
        assert agent.network is None

    def test_custom_simulations(self) -> None:
        """Custom num_simulations should be stored."""
        agent = MCTSAgent(player_id=0, num_simulations=50)
        assert agent.num_simulations == 50

    def test_network_none_without_path(self) -> None:
        """Without network_path, network property should be None."""
        agent = MCTSAgent(player_id=0)
        assert agent.network is None

    def test_search_stats_initialized(self) -> None:
        """search_stats should start at zero."""
        agent = MCTSAgent(player_id=0)
        assert agent.search_stats.total_searches == 0

    def test_name_includes_player_id(self) -> None:
        """Agent name should reference the player ID."""
        agent = MCTSAgent(player_id=3)
        assert "3" in agent.name


# ===========================================================================
# C1-3: choose_action
# ===========================================================================


class TestMCTSAgentChooseAction:
    """Test MCTSAgent.choose_action returns valid actions."""

    def test_returns_valid_action(self) -> None:
        """Action returned should be in the valid action mask."""
        game = _make_game(seed=42)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(player_id=pid, num_simulations=10)

        action = agent.choose_action(obs, mask, game)

        assert isinstance(action, int)
        assert 0 <= action < 149
        assert mask[action], f"Action {action} should be valid per mask"

    def test_works_without_network(self) -> None:
        """Pure MCTS (no value network) should still return a valid action."""
        game = _make_game(seed=7)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(player_id=pid, num_simulations=8, network_path=None)

        action = agent.choose_action(obs, mask, game)

        assert mask[action], f"Action {action} is not valid"

    def test_does_not_modify_game(self) -> None:
        """choose_action must not change the live game state."""
        game = _make_game(seed=10)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        state_before = game.to_dict()
        agent = MCTSAgent(player_id=pid, num_simulations=8)

        agent.choose_action(obs, mask, game)

        assert game.to_dict() == state_before

    def test_updates_search_stats(self) -> None:
        """After choose_action, stats should reflect one search."""
        game = _make_game(seed=15)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(player_id=pid, num_simulations=5)

        agent.choose_action(obs, mask, game)

        assert agent.search_stats.total_searches == 1
        assert agent.search_stats.total_simulations == 5
        assert agent.search_stats.total_time_sec > 0.0

    def test_multiple_calls_accumulate_stats(self) -> None:
        """Repeated calls should accumulate stats."""
        game = _make_game(seed=20)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(player_id=pid, num_simulations=5)

        agent.choose_action(obs, mask, game)
        agent.choose_action(obs, mask, game)

        assert agent.search_stats.total_searches == 2
        assert agent.search_stats.total_simulations == 10

    def test_deterministic_at_temperature_zero(self) -> None:
        """temperature=0 should select the same action repeatedly."""
        game = _make_game(seed=99)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(player_id=pid, num_simulations=20, temperature=0.0)

        actions = [agent.choose_action(obs, mask, game) for _ in range(3)]

        # All calls on the same state with temperature=0 should agree
        assert len(set(actions)) == 1

    def test_with_network(self, tmp_path: pytest.TempPathFactory) -> None:
        """Agent loaded with a network checkpoint should return valid action."""
        from mcts.features import get_feature_size
        from mcts.network import ValueNetwork

        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size, num_players=4)
        net_path = tmp_path / "net.pt"  # type: ignore[operator]
        net.save(net_path)

        game = _make_game(seed=5)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(
            player_id=pid,
            num_simulations=5,
            network_path=net_path,
        )

        assert agent.network is not None
        action = agent.choose_action(obs, mask, game)
        assert mask[action]


# ===========================================================================
# C1-4: reset
# ===========================================================================


class TestMCTSAgentReset:
    """Test MCTSAgent.reset clears per-game state."""

    def test_reset_clears_stats(self) -> None:
        """reset() should zero out search_stats."""
        game = _make_game(seed=1)
        pid = game.current_player
        obs, mask = _get_obs_and_mask(game, pid)
        agent = MCTSAgent(player_id=pid, num_simulations=5)

        agent.choose_action(obs, mask, game)
        assert agent.search_stats.total_searches == 1

        agent.reset()

        assert agent.search_stats.total_searches == 0
        assert agent.search_stats.total_time_sec == 0.0
        assert agent.search_stats.total_simulations == 0
