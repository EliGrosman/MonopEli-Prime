"""Tests for self-play training module."""

import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.self_play import (
    SelfPlayConfig,
    SelfPlayEnv,
    SelfPlayOpponent,
    SelfPlayStats,
    SelfPlayTrainer,
)


class TestSelfPlayConfig:
    """Tests for SelfPlayConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SelfPlayConfig()

        assert config.total_timesteps == 1_000_000
        assert config.num_envs == 8
        assert config.checkpoint_freq == 50_000
        assert config.past_version_prob == 0.5
        assert config.max_past_versions == 10
        assert config.eval_freq == 25_000
        assert config.eval_episodes == 20
        assert config.learning_rate == 3e-4
        assert config.n_steps == 512
        assert config.batch_size == 128
        assert config.seed == 42
        assert config.save_dir == "models/self_play"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SelfPlayConfig(
            total_timesteps=500_000,
            num_envs=4,
            checkpoint_freq=25_000,
            past_version_prob=0.7,
            max_past_versions=5,
            seed=123,
            save_dir="custom/path",
        )

        assert config.total_timesteps == 500_000
        assert config.num_envs == 4
        assert config.checkpoint_freq == 25_000
        assert config.past_version_prob == 0.7
        assert config.max_past_versions == 5
        assert config.seed == 123
        assert config.save_dir == "custom/path"


class TestSelfPlayStats:
    """Tests for SelfPlayStats."""

    def test_default_values(self) -> None:
        """Test default statistics values."""
        stats = SelfPlayStats()

        assert stats.total_steps == 0
        assert stats.num_checkpoints == 0
        assert stats.win_rates == []
        assert stats.training_time == 0.0

    def test_mutable_win_rates(self) -> None:
        """Test that win_rates list is mutable."""
        stats = SelfPlayStats()
        stats.win_rates.append({"step": 1000, "vs_random": 0.5})

        assert len(stats.win_rates) == 1
        assert stats.win_rates[0]["step"] == 1000


class TestSelfPlayOpponent:
    """Tests for SelfPlayOpponent."""

    def test_init(self) -> None:
        """Test opponent initialization."""
        mock_model = MagicMock()
        opponent = SelfPlayOpponent(mock_model, player_id=1)

        assert opponent.model is mock_model
        assert opponent.player_id == 1
        assert opponent.deterministic is True
        assert opponent.name == "SelfPlay"

    def test_init_non_deterministic(self) -> None:
        """Test opponent with non-deterministic actions."""
        mock_model = MagicMock()
        opponent = SelfPlayOpponent(mock_model, player_id=2, deterministic=False)

        assert opponent.deterministic is False

    def test_choose_action_with_observation(self) -> None:
        """Test action selection with provided observation."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(42), None)

        opponent = SelfPlayOpponent(mock_model, player_id=1)

        # Create mock observation and mask
        obs = np.zeros(280, dtype=np.float32)
        action_mask = np.ones(149, dtype=np.bool_)
        game = MagicMock()

        action = opponent.choose_action(obs, action_mask, game)

        assert action == 42
        mock_model.predict.assert_called_once()

    def test_reset_does_nothing(self) -> None:
        """Test that reset doesn't raise."""
        mock_model = MagicMock()
        opponent = SelfPlayOpponent(mock_model, player_id=1)

        # Should not raise
        opponent.reset()

    def test_notify_result_does_nothing(self) -> None:
        """Test that notify_result doesn't raise."""
        mock_model = MagicMock()
        opponent = SelfPlayOpponent(mock_model, player_id=1)

        # Should not raise
        opponent.notify_result(1, 0.5, {}, False, False)


class TestSelfPlayEnv:
    """Tests for SelfPlayEnv."""

    def test_init(self) -> None:
        """Test environment initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SelfPlayEnv(
                checkpoint_dir=tmpdir,
                past_version_prob=0.5,
                num_players=2,
                max_turns=500,
                seed=42,
            )

            assert env.checkpoint_dir == Path(tmpdir)
            assert env.past_version_prob == 0.5
            assert env.num_players == 2
            assert env.max_turns == 500
            assert env.observation_space is not None
            assert env.action_space is not None

            env.close()

    def test_refresh_checkpoints_empty_dir(self) -> None:
        """Test refreshing checkpoints in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SelfPlayEnv(checkpoint_dir=tmpdir)
            env.refresh_checkpoints()

            assert env._checkpoints == []

            env.close()

    def test_select_opponent_no_checkpoints(self) -> None:
        """Test opponent selection with no checkpoints (falls back to random)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SelfPlayEnv(checkpoint_dir=tmpdir, past_version_prob=1.0)

            opponent = env._select_opponent()

            # Should be RandomAgent since no checkpoints exist
            assert opponent.name == "Random"

            env.close()

    def test_reset_returns_valid_observation(self) -> None:
        """Test reset returns valid observation and info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SelfPlayEnv(checkpoint_dir=tmpdir, seed=42)

            obs, info = env.reset()

            assert obs is not None
            assert isinstance(info, dict)
            assert "action_mask" in info

            env.close()

    def test_step_returns_valid_result(self) -> None:
        """Test step returns valid result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SelfPlayEnv(checkpoint_dir=tmpdir, seed=42)
            env.reset()

            # Get valid action from mask
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = int(valid_actions[0]) if len(valid_actions) > 0 else 146  # end turn

            obs, reward, term, trunc, info = env.step(action)

            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)

            env.close()

    def test_action_masks_returns_valid_mask(self) -> None:
        """Test action_masks returns valid mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SelfPlayEnv(checkpoint_dir=tmpdir, seed=42)
            env.reset()

            mask = env.action_masks()

            assert isinstance(mask, np.ndarray)
            assert mask.shape == (149,)
            assert mask.dtype == np.bool_

            env.close()


class TestSelfPlayTrainer:
    """Tests for SelfPlayTrainer."""

    def test_init_default_config(self) -> None:
        """Test trainer initialization with default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelfPlayConfig(save_dir=tmpdir)
            trainer = SelfPlayTrainer(config)

            assert trainer.config is config
            assert trainer.save_dir == Path(tmpdir)
            assert trainer.checkpoint_dir.exists()

    def test_init_creates_directories(self) -> None:
        """Test trainer creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "nested" / "self_play"
            config = SelfPlayConfig(save_dir=str(save_dir))
            trainer = SelfPlayTrainer(config)

            assert trainer.checkpoint_dir.exists()

    def test_stats_initialized(self) -> None:
        """Test statistics are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelfPlayConfig(save_dir=tmpdir)
            trainer = SelfPlayTrainer(config)

            assert trainer.stats.total_steps == 0
            assert trainer.stats.num_checkpoints == 0


# Tests that require sb3-contrib
class TestSelfPlayTrainerWithSB3:
    """Tests for SelfPlayTrainer that require sb3-contrib."""

    @pytest.fixture(autouse=True)
    def check_sb3(self) -> None:
        """Skip tests if sb3-contrib is not available."""
        pytest.importorskip("sb3_contrib")

    def test_save_checkpoint(self) -> None:
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            from monopoly_gym import SingleAgentMonopolyEnv

            config = SelfPlayConfig(save_dir=tmpdir, max_past_versions=3)
            trainer = SelfPlayTrainer(config)

            # Create a simple model
            env = DummyVecEnv([lambda: SingleAgentMonopolyEnv(num_players=2)])
            model = MaskablePPO("MlpPolicy", env, verbose=0)

            # Save checkpoint
            path = trainer._save_checkpoint(model, 1000)

            assert path.exists()
            assert trainer.stats.num_checkpoints == 1

            env.close()

    def test_checkpoint_pruning(self) -> None:
        """Test that old checkpoints are pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            from monopoly_gym import SingleAgentMonopolyEnv

            config = SelfPlayConfig(save_dir=tmpdir, max_past_versions=2)
            trainer = SelfPlayTrainer(config)

            # Create a simple model
            env = DummyVecEnv([lambda: SingleAgentMonopolyEnv(num_players=2)])
            model = MaskablePPO("MlpPolicy", env, verbose=0)

            # Save multiple checkpoints
            trainer._save_checkpoint(model, 1000)
            trainer._save_checkpoint(model, 2000)
            trainer._save_checkpoint(model, 3000)

            # Should have pruned oldest
            checkpoints = list(trainer.checkpoint_dir.glob("checkpoint_*.zip"))
            assert len(checkpoints) <= 2

            env.close()

    def test_quick_training(self) -> None:
        """Test that training runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SelfPlayConfig(
                total_timesteps=1000,
                num_envs=1,
                checkpoint_freq=500,
                eval_freq=1000,
                eval_episodes=2,
                n_steps=64,
                batch_size=32,
                save_dir=tmpdir,
            )
            trainer = SelfPlayTrainer(config)

            model = trainer.train(verbose=False)

            assert model is not None
            assert trainer.stats.total_steps == 1000
            assert trainer.stats.num_checkpoints >= 2  # Initial + at least one more


class TestSelfPlayIntegration:
    """Integration tests for self-play training."""

    @pytest.fixture(autouse=True)
    def check_sb3(self) -> None:
        """Skip tests if sb3-contrib is not available."""
        pytest.importorskip("sb3_contrib")

    def test_self_play_env_with_checkpoint(self) -> None:
        """Test self-play env can load and use a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            from monopoly_gym import SingleAgentMonopolyEnv

            # Create and save a model
            env = DummyVecEnv([lambda: SingleAgentMonopolyEnv(num_players=2)])
            model = MaskablePPO("MlpPolicy", env, verbose=0)
            model.save(f"{tmpdir}/checkpoint_1000.zip")
            env.close()

            # Create self-play env
            sp_env = SelfPlayEnv(
                checkpoint_dir=tmpdir,
                past_version_prob=1.0,  # Always use past version
                seed=42,
            )
            sp_env.refresh_checkpoints()

            # Opponent should be SelfPlayOpponent
            opponent = sp_env._select_opponent()
            assert opponent.name == "SelfPlay"

            sp_env.close()

    def test_train_self_play_function(self) -> None:
        """Test convenience function for self-play training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from training.self_play import train_self_play

            model = train_self_play(
                total_timesteps=500,
                checkpoint_freq=250,
                past_version_prob=0.5,
                save_dir=tmpdir,
                verbose=False,
            )

            assert model is not None
            assert (Path(tmpdir) / "final_model.zip").exists()


class TestSelfPlayOpponentWithGame:
    """Tests for SelfPlayOpponent with actual game state."""

    @pytest.fixture(autouse=True)
    def check_sb3(self) -> None:
        """Skip tests if sb3-contrib is not available."""
        pytest.importorskip("sb3_contrib")

    def test_choose_action_with_none_observation(self) -> None:
        """Test action selection when observation is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            from monopoly_engine import MonopolyGame
            from monopoly_gym import SingleAgentMonopolyEnv

            # Create and train a minimal model
            env = DummyVecEnv([lambda: SingleAgentMonopolyEnv(num_players=2)])
            model = MaskablePPO("MlpPolicy", env, verbose=0)
            env.close()

            # Create opponent
            opponent = SelfPlayOpponent(model, player_id=1)

            # Create game state
            game = MonopolyGame(num_players=2, seed=42)
            action_mask = np.ones(149, dtype=np.bool_)

            # Should work with None observation
            action = opponent.choose_action(None, action_mask, game)

            assert isinstance(action, int)
            assert 0 <= action < 149
