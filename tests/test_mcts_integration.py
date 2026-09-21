"""End-to-end integration tests for the MCTS engine (C5).

These tests exercise multiple components working together rather than
isolated units. All tests use minimal configs (few simulations, short games)
to keep runtime fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from mcts.data import ReplayBuffer, generate_training_data
from mcts.eval import evaluate_mcts_agent, play_evaluation_game
from mcts.features import extract_features, get_feature_size
from mcts.network import ValueNetwork
from mcts.search import _auto_roll_dice
from mcts.training import (
    SelfPlayConfig,
    TrainingConfig,
    load_checkpoint,
    self_play_loop,
    train_value_network,
)
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_game(num_players: int = 2, seed: int = 42) -> MonopolyGame:
    """Create a game with dice already rolled so the first player can act."""
    game = MonopolyGame(num_players=num_players, seed=seed)
    _auto_roll_dice(game)
    return game


def _fast_mcts(player_id: int = 0) -> MCTSAgent:
    """Return an MCTSAgent with very few simulations for speed."""
    return MCTSAgent(player_id=player_id, num_simulations=5)


def _small_selfplay_config() -> SelfPlayConfig:
    """Return a minimal SelfPlayConfig for smoke-testing the self-play loop."""
    return SelfPlayConfig(
        num_iterations=1,
        games_per_iteration=2,
        mcts_simulations=5,
        num_players=2,
        temperature=1.0,
        eval_games=2,
        eval_frequency=999,   # skip head-to-head eval
        max_turns_per_game=30,
        training_config=TrainingConfig(num_epochs=1, batch_size=8),
    )


# ===========================================================================
# I1: Full gameplay with MCTSAgent
# ===========================================================================


class TestMCTSGameplay:
    """Integration tests for MCTSAgent playing real games."""

    def test_mcts_agent_completes_two_player_game(self) -> None:
        """MCTSAgent vs RandomAgent game runs to completion without error."""
        mcts = _fast_mcts(player_id=0)
        opponent = RandomAgent(player_id=1)
        game = MonopolyGame(num_players=2, seed=7)
        encoder = ActionEncoder(enable_trades=False)
        agents = {0: mcts, 1: opponent}

        for _ in range(30):
            if game.game_over:
                break
            pid = game.current_player
            mask = encoder.get_action_mask(game, pid)
            action_idx = agents[pid].choose_action({}, mask, game)
            action = encoder.decode(action_idx, pid, game)
            valid, _ = action.validate(game)
            if valid:
                action.execute(game)

        # Should complete without exception (winner or truncated — both OK)
        assert game.winner is None or 0 <= game.winner <= 1

    def test_mcts_agent_actions_always_valid(self) -> None:
        """All actions chosen by MCTSAgent must pass action validation."""
        mcts = _fast_mcts(player_id=0)
        game = MonopolyGame(num_players=2, seed=13)
        encoder = ActionEncoder(enable_trades=False)
        opponent = RandomAgent(player_id=1)
        agents = {0: mcts, 1: opponent}

        for _ in range(20):
            if game.game_over:
                break
            pid = game.current_player
            mask = encoder.get_action_mask(game, pid)
            action_idx = agents[pid].choose_action({}, mask, game)
            action = encoder.decode(action_idx, pid, game)

            if pid == 0:
                # MCTS action must be valid per the mask
                assert mask[action_idx], (
                    f"MCTSAgent chose invalid action {action_idx} "
                    f"(mask sum={mask.sum()})"
                )

            valid, _ = action.validate(game)
            if valid:
                action.execute(game)

    def test_game_state_preserved_during_search(self) -> None:
        """choose_action must not modify the live game state."""
        mcts = _fast_mcts(player_id=0)
        game = _make_game(num_players=2, seed=99)
        encoder = ActionEncoder(enable_trades=False)
        mask = encoder.get_action_mask(game, 0)

        state_before = game.to_dict()
        mcts.choose_action({}, mask, game)
        state_after = game.to_dict()

        assert state_before == state_after, "Game state was mutated during MCTS search"

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_mcts_with_network_completes_game(self, tmp_path: pytest.TempPathFactory) -> None:
        """MCTSAgent loaded with a value network plays a game without error."""
        feature_size = get_feature_size(2)
        net = ValueNetwork(input_size=feature_size, num_players=2)
        net_path = tmp_path / "net.pt"  # type: ignore[operator]
        net.save(net_path)

        mcts = MCTSAgent(
            player_id=0,
            num_simulations=5,
            network_path=net_path,
        )
        opponent = RandomAgent(player_id=1)
        game = MonopolyGame(num_players=2, seed=55)
        encoder = ActionEncoder(enable_trades=False)
        agents = {0: mcts, 1: opponent}

        for _ in range(20):
            if game.game_over:
                break
            pid = game.current_player
            mask = encoder.get_action_mask(game, pid)
            action_idx = agents[pid].choose_action({}, mask, game)
            action = encoder.decode(action_idx, pid, game)
            valid, _ = action.validate(game)
            if valid:
                action.execute(game)

        assert mcts.network is not None
        assert mcts.search_stats.total_searches > 0

    def test_search_stats_accumulate_across_turns(self) -> None:
        """Stats should grow monotonically as MCTSAgent takes more turns."""
        mcts = _fast_mcts(player_id=0)
        game = MonopolyGame(num_players=2, seed=3)
        encoder = ActionEncoder(enable_trades=False)

        prev_searches = 0
        for _ in range(15):
            if game.game_over:
                break
            pid = game.current_player
            mask = encoder.get_action_mask(game, pid)
            action_idx = mcts.choose_action({}, mask, game)
            action = encoder.decode(action_idx, pid, game)
            valid, _ = action.validate(game)
            if valid:
                action.execute(game)

            if pid == 0:
                assert mcts.search_stats.total_searches > prev_searches
                prev_searches = mcts.search_stats.total_searches


# ===========================================================================
# I2: Training data pipeline
# ===========================================================================


class TestDataPipeline:
    """Integration tests for the data generation → training pipeline."""

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_generate_data_feeds_buffer(self) -> None:
        """generate_training_data output can be added to a ReplayBuffer."""
        buf = ReplayBuffer(capacity=500)
        new_data = generate_training_data(
            num_games=2,
            num_players=2,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=0,
        )
        for example in new_data.sample(len(new_data)):
            buf.add(example)

        assert len(buf) > 0

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_training_data_trains_network(self) -> None:
        """End-to-end: generate data → buffer → train_value_network runs."""
        buf = generate_training_data(
            num_games=3,
            num_players=2,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=25,
            seed=1,
        )
        if len(buf) == 0:
            pytest.skip("No training examples generated")

        feature_size = get_feature_size(2)
        net = ValueNetwork(input_size=feature_size, num_players=2)
        config = TrainingConfig(num_epochs=2, batch_size=4)

        stats = train_value_network(net, buf, config)

        assert len(stats.epoch_losses) == 2
        for vloss, ploss, total in stats.epoch_losses:
            assert total >= 0.0

    def test_features_from_real_game_match_expected_size(self) -> None:
        """Features extracted from a live game state have the correct dimension."""
        game = MonopolyGame(num_players=4, seed=42)
        _auto_roll_dice(game)
        expected_size = get_feature_size(4)

        feats = extract_features(game, player_id=0)

        assert feats.shape == (expected_size,)
        assert np.isfinite(feats).all(), "Feature vector contains NaN or Inf"

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_buffer_round_trip_preserves_data(self, tmp_path: pytest.TempPathFactory) -> None:
        """ReplayBuffer saved and reloaded preserves all examples."""
        buf = generate_training_data(
            num_games=2,
            num_players=2,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=2,
        )
        if len(buf) == 0:
            pytest.skip("No training examples generated")

        buf_path = tmp_path / "buf.npz"  # type: ignore[operator]
        buf.save(buf_path)
        loaded = ReplayBuffer.load(buf_path)

        assert len(loaded) == len(buf)


# ===========================================================================
# I3: Self-play training loop
# ===========================================================================


class TestSelfPlayPipeline:
    """Integration tests for the AlphaZero-style self-play loop."""

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_one_iteration_creates_checkpoint(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """One self-play iteration should create mcts_iter_0/ directory."""
        config = _small_selfplay_config()
        save_dir = tmp_path / "sp"  # type: ignore[operator]

        self_play_loop(config, save_dir=save_dir)

        ckpt = save_dir / "mcts_iter_0"  # type: ignore[operator]
        assert ckpt.exists(), "Iteration checkpoint not created"
        assert (ckpt / "network.pt").exists()  # type: ignore[operator]
        assert (ckpt / "buffer.npz").exists()  # type: ignore[operator]
        assert (ckpt / "metadata.json").exists()  # type: ignore[operator]

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_checkpoint_loads_valid_network(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Network saved in checkpoint is loadable and produces valid output."""
        config = _small_selfplay_config()
        save_dir = tmp_path / "sp"  # type: ignore[operator]

        self_play_loop(config, save_dir=save_dir)
        ckpt = save_dir / "mcts_iter_0"  # type: ignore[operator]

        network, buffer, iteration, stats = load_checkpoint(ckpt)

        assert isinstance(network, ValueNetwork)
        assert iteration == 0

        # Network should produce valid output on a real game state
        game = MonopolyGame(num_players=2, seed=42)
        _auto_roll_dice(game)
        feats = extract_features(game, player_id=0, num_players=2)

        import torch
        t = torch.from_numpy(feats).unsqueeze(0)
        values, policy = network(t)
        assert values.shape == (1, 2)
        assert policy.shape == (1, 149)

    @pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_resume_continues_from_previous_iteration(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Resuming from a checkpoint produces the next iteration index."""
        config = _small_selfplay_config()
        save_dir = tmp_path / "sp"  # type: ignore[operator]

        self_play_loop(config, save_dir=save_dir)
        ckpt_0 = save_dir / "mcts_iter_0"  # type: ignore[operator]
        assert ckpt_0.exists()

        self_play_loop(config, save_dir=save_dir, resume_from=ckpt_0)
        ckpt_1 = save_dir / "mcts_iter_1"  # type: ignore[operator]
        assert ckpt_1.exists(), "Resume did not create iteration 1"


# ===========================================================================
# I4: Evaluation pipeline
# ===========================================================================


class TestEvaluationPipeline:
    """Integration tests for the evaluation harness."""

    def _fast_eval_kwargs(self) -> dict:  # type: ignore[type-arg]
        return dict(
            network_path=None,
            num_simulations=5,
            num_games=2,
            num_players=2,
            max_turns=20,
            seed=42,
            verbose=False,
        )

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_evaluate_no_network_returns_dict(self) -> None:
        """evaluate_mcts_agent with no network returns a valid results dict."""
        result = evaluate_mcts_agent(
            opponents=["random"],
            **self._fast_eval_kwargs(),
        )
        assert isinstance(result, dict)
        assert "random" in result

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_evaluation_metrics_in_valid_ranges(self) -> None:
        """win_rate ∈ [0,1], avg_game_length > 0, avg_time_sec > 0."""
        result = evaluate_mcts_agent(
            opponents=["random"],
            **self._fast_eval_kwargs(),
        )
        m = result["random"]
        assert 0.0 <= m["win_rate"] <= 1.0
        assert m["avg_game_length"] > 0
        assert m["avg_time_sec"] > 0

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_evaluate_with_network_runs(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """evaluate_mcts_agent with a trained network does not crash."""
        feature_size = get_feature_size(2)
        net = ValueNetwork(input_size=feature_size, num_players=2)
        net_path = tmp_path / "net.pt"  # type: ignore[operator]
        net.save(net_path)

        result = evaluate_mcts_agent(
            network_path=net_path,
            opponents=["random"],
            **{k: v for k, v in self._fast_eval_kwargs().items()
               if k != "network_path"},
        )
        assert "random" in result
        assert 0.0 <= result["random"]["win_rate"] <= 1.0

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_full_pipeline_smoke(self, tmp_path: pytest.TempPathFactory) -> None:
        """Generate data → train network → evaluate: full pipeline completes."""
        # Step 1: generate training data
        buf = generate_training_data(
            num_games=2,
            num_players=2,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=10,
        )

        # Step 2: train network
        feature_size = get_feature_size(2)
        net = ValueNetwork(input_size=feature_size, num_players=2)
        if len(buf) > 0:
            train_value_network(
                net,
                buf,
                TrainingConfig(num_epochs=1, batch_size=4),
            )

        # Step 3: save and evaluate
        net_path = tmp_path / "net.pt"  # type: ignore[operator]
        net.save(net_path)

        result = evaluate_mcts_agent(
            network_path=net_path,
            opponents=["random"],
            num_simulations=5,
            num_games=2,
            num_players=2,
            max_turns=20,
            seed=99,
            verbose=False,
        )

        assert isinstance(result, dict)
        assert "random" in result

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Deferred search/trading positive path; foundation rejects this mode")
    def test_play_evaluation_game_uses_both_agent_types(self) -> None:
        """play_evaluation_game works with MCTSAgent vs RuleBasedAgent."""
        mcts = MCTSAgent(player_id=0, num_simulations=5)
        opponents = [RuleBasedAgent(player_id=1)]

        winner, action_count, elapsed = play_evaluation_game(
            mcts,
            opponents,
            max_turns=25,
            seed=7,
        )

        assert action_count <= 25
        assert elapsed > 0.0
        assert winner is None or winner in (0, 1)
