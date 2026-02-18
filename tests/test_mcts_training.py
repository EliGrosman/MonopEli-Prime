"""Tests for MCTS value network training loop (B4) and self-play loop (B5+B6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mcts.data import ReplayBuffer, TrainingExample
from mcts.features import get_feature_size
from mcts.network import ValueNetwork
from mcts.training import (
    SelfPlayConfig,
    TrainingConfig,
    TrainingStats,
    load_checkpoint,
    save_checkpoint,
    self_play_loop,
    train_value_network,
)


def _make_buffer(n: int = 50, num_players: int = 4) -> ReplayBuffer:
    """Create a replay buffer with n synthetic training examples."""
    feature_size = get_feature_size(num_players)
    buf = ReplayBuffer(capacity=n * 2)
    rng = np.random.default_rng(42)

    for i in range(n):
        features = rng.random(feature_size).astype(np.float32)
        policy = rng.random(149).astype(np.float32)
        policy /= policy.sum()  # normalize to distribution
        outcome = np.zeros(num_players, dtype=np.float32)
        outcome[i % num_players] = 1.0
        for j in range(num_players):
            if j != i % num_players:
                outcome[j] = -1.0

        buf.add(TrainingExample(
            features=features,
            mcts_policy=policy,
            outcome=outcome,
            player_id=i % num_players,
        ))

    return buf


# ===========================================================================
# B4-1: TrainingConfig
# ===========================================================================


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify default configuration values."""
        config = TrainingConfig()

        assert config.learning_rate == 1e-3
        assert config.lr_decay == 0.99
        assert config.batch_size == 256
        assert config.num_epochs == 10
        assert config.value_loss_weight == 1.0
        assert config.policy_loss_weight == 1.0
        assert config.weight_decay == 1e-4
        assert config.validation_fraction == 0.1
        assert config.device == "cpu"

    def test_custom_values(self) -> None:
        """Can override all fields."""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            num_epochs=5,
            device="cpu",
        )

        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.num_epochs == 5


# ===========================================================================
# B4-2: TrainingStats
# ===========================================================================


class TestTrainingStats:
    """Test TrainingStats dataclass."""

    def test_empty_stats(self) -> None:
        """New stats should have empty lists."""
        stats = TrainingStats()

        assert stats.epoch_losses == []
        assert stats.val_losses == []


# ===========================================================================
# B4-3: Loss computation
# ===========================================================================


class TestLossComputation:
    """Test value and policy loss computation."""

    def test_value_loss_is_mse(self) -> None:
        """Value loss should be MSE between predictions and targets."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        net.eval()

        features = torch.randn(8, feature_size)
        targets = torch.zeros(8, 4)
        targets[:, 0] = 1.0
        targets[:, 1:] = -1.0

        values, policy_logits = net(features)
        value_loss = torch.nn.functional.mse_loss(values, targets)

        assert value_loss.item() > 0

    def test_policy_loss_is_cross_entropy(self) -> None:
        """Policy loss should be cross-entropy with MCTS distribution."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        net.eval()

        features = torch.randn(8, feature_size)
        # Create a target distribution (sums to 1)
        target_policy = torch.zeros(8, 149)
        target_policy[:, 0] = 1.0  # all probability on action 0

        _, policy_logits = net(features)
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

        assert policy_loss.item() > 0

    def test_zero_value_loss_for_perfect_prediction(self) -> None:
        """MSE loss should be zero when predictions match targets exactly."""
        target = torch.tensor([[1.0, -1.0, -1.0, -1.0]])
        predicted = torch.tensor([[1.0, -1.0, -1.0, -1.0]])
        loss = torch.nn.functional.mse_loss(predicted, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# B4-4: train_value_network
# ===========================================================================


class TestTrainValueNetwork:
    """Test the full training loop."""

    def test_training_returns_stats(self) -> None:
        """Training should return TrainingStats with correct number of epochs."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(30)
        config = TrainingConfig(
            num_epochs=3,
            batch_size=16,
            device="cpu",
            validation_fraction=0.0,  # no validation for simplicity
        )

        stats = train_value_network(net, buf, config)

        assert isinstance(stats, TrainingStats)
        assert len(stats.epoch_losses) == 3

    def test_training_reduces_loss(self) -> None:
        """Loss should decrease over multiple epochs of training."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(100)
        config = TrainingConfig(
            num_epochs=20,
            batch_size=32,
            learning_rate=1e-3,
            device="cpu",
            validation_fraction=0.0,
        )

        stats = train_value_network(net, buf, config)

        first_loss = stats.epoch_losses[0][2]  # total loss
        last_loss = stats.epoch_losses[-1][2]
        assert last_loss < first_loss

    def test_training_with_validation(self) -> None:
        """Training with validation split should record val losses."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(50)
        config = TrainingConfig(
            num_epochs=3,
            batch_size=16,
            device="cpu",
            validation_fraction=0.2,
        )

        stats = train_value_network(net, buf, config)

        assert len(stats.val_losses) == 3
        # Validation losses should be finite
        for vloss, ploss, total in stats.val_losses:
            assert np.isfinite(vloss)
            assert np.isfinite(ploss)
            assert np.isfinite(total)

    def test_training_empty_buffer_raises(self) -> None:
        """Training on an empty buffer should raise ValueError."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = ReplayBuffer(capacity=100)

        with pytest.raises(ValueError, match="empty"):
            train_value_network(net, buf)

    def test_training_default_config(self) -> None:
        """Training should work with default config (None)."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(20)

        stats = train_value_network(net, buf, config=None)

        assert len(stats.epoch_losses) == 10  # default num_epochs

    def test_training_modifies_network_weights(self) -> None:
        """Network weights should change after training."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(30)
        config = TrainingConfig(
            num_epochs=5,
            batch_size=16,
            device="cpu",
            validation_fraction=0.0,
        )

        # Capture initial weights
        initial_weights = {
            name: param.clone() for name, param in net.named_parameters()
        }

        train_value_network(net, buf, config)

        # At least some weights should have changed
        changed = False
        for name, param in net.named_parameters():
            if not torch.equal(param, initial_weights[name]):
                changed = True
                break
        assert changed

    def test_loss_components_are_positive(self) -> None:
        """Both value and policy losses should be positive."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(30)
        config = TrainingConfig(
            num_epochs=1,
            batch_size=16,
            device="cpu",
            validation_fraction=0.0,
        )

        stats = train_value_network(net, buf, config)

        vloss, ploss, total = stats.epoch_losses[0]
        assert vloss > 0
        assert ploss > 0
        assert total > 0

    def test_small_buffer_no_validation(self) -> None:
        """Very small buffer should skip validation even if fraction > 0."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(3)
        config = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            device="cpu",
            validation_fraction=0.2,  # would be < 1 example
        )

        stats = train_value_network(net, buf, config)

        # With only 3 examples and 20% val, val_size = 0 -> no val
        assert len(stats.epoch_losses) == 2
        assert len(stats.val_losses) == 0


# ===========================================================================
# B4-5: TensorBoard logging
# ===========================================================================


class TestTensorBoardLogging:
    """Test TensorBoard integration."""

    def test_tensorboard_creates_log_files(self, tmp_path: pytest.TempPathFactory) -> None:
        """TensorBoard logs should be created when dir is specified."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(30)
        config = TrainingConfig(
            num_epochs=2,
            batch_size=16,
            device="cpu",
            validation_fraction=0.2,
        )

        log_dir = tmp_path / "tb_logs"  # type: ignore[operator]
        train_value_network(net, buf, config, tensorboard_dir=log_dir)

        # TensorBoard should have created event files
        assert log_dir.exists()  # type: ignore[union-attr]
        event_files = list(log_dir.glob("events.out.*"))  # type: ignore[union-attr]
        assert len(event_files) > 0

    def test_training_without_tensorboard(self) -> None:
        """Training should work fine without TensorBoard dir."""
        feature_size = get_feature_size(4)
        net = ValueNetwork(input_size=feature_size)
        buf = _make_buffer(20)
        config = TrainingConfig(
            num_epochs=2,
            batch_size=16,
            device="cpu",
        )

        stats = train_value_network(net, buf, config, tensorboard_dir=None)
        assert len(stats.epoch_losses) == 2


# ===========================================================================
# B6: Checkpoint Management
# ===========================================================================


class TestCheckpoint:
    """Test save_checkpoint and load_checkpoint."""

    def _make_network(self) -> ValueNetwork:
        return ValueNetwork(input_size=get_feature_size(4), num_players=4)

    def _make_buffer(self) -> ReplayBuffer:
        buf = ReplayBuffer(capacity=100)
        rng = np.random.default_rng(0)
        feature_size = get_feature_size(4)
        for i in range(5):
            buf.add(TrainingExample(
                features=rng.random(feature_size).astype(np.float32),
                mcts_policy=rng.random(149).astype(np.float32),
                outcome=np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                player_id=i % 4,
            ))
        return buf

    def test_checkpoint_directory_structure(self, tmp_path: pytest.TempPathFactory) -> None:
        """save_checkpoint creates the expected files."""
        net = self._make_network()
        buf = self._make_buffer()
        ckpt_dir = tmp_path / "ckpt"  # type: ignore[operator]

        save_checkpoint(ckpt_dir, net, buf, iteration=3, stats={"loss": 0.5})

        assert (ckpt_dir / "network.pt").exists()  # type: ignore[operator]
        assert (ckpt_dir / "buffer.npz").exists()  # type: ignore[operator]
        assert (ckpt_dir / "metadata.json").exists()  # type: ignore[operator]

    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loaded checkpoint should match what was saved."""
        net = self._make_network()
        buf = self._make_buffer()
        original_stats = {"win_rate": 0.62, "iteration": 7}
        ckpt_dir = tmp_path / "ckpt"  # type: ignore[operator]

        save_checkpoint(ckpt_dir, net, buf, iteration=7, stats=original_stats)
        loaded_net, loaded_buf, loaded_iter, loaded_stats = load_checkpoint(ckpt_dir)

        assert loaded_iter == 7
        assert loaded_stats["win_rate"] == pytest.approx(0.62)
        assert len(loaded_buf) == len(buf)
        assert loaded_net.input_size == net.input_size
        assert loaded_net.num_players == net.num_players

    def test_checkpoint_network_weights_preserved(self, tmp_path: pytest.TempPathFactory) -> None:
        """Network weights should be identical after save/load."""
        net = self._make_network()
        ckpt_dir = tmp_path / "ckpt"  # type: ignore[operator]

        save_checkpoint(ckpt_dir, net, self._make_buffer(), iteration=0, stats={})
        loaded_net, _, _, _ = load_checkpoint(ckpt_dir)

        for (name, p1), (_, p2) in zip(
            net.named_parameters(), loaded_net.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter {name} differs after load"

    def test_checkpoint_creates_parent_dirs(self, tmp_path: pytest.TempPathFactory) -> None:
        """save_checkpoint should create nested directories."""
        net = self._make_network()
        deep_path = tmp_path / "a" / "b" / "c"  # type: ignore[operator]

        save_checkpoint(deep_path, net, self._make_buffer(), iteration=0, stats={})

        assert (deep_path / "network.pt").exists()  # type: ignore[operator]

    def test_metadata_iteration_field(self, tmp_path: pytest.TempPathFactory) -> None:
        """metadata.json should contain the iteration number."""
        import json as json_mod

        net = self._make_network()
        ckpt_dir = tmp_path / "ckpt"  # type: ignore[operator]
        save_checkpoint(ckpt_dir, net, self._make_buffer(), iteration=42, stats={})

        with open(ckpt_dir / "metadata.json") as f:  # type: ignore[operator]
            meta = json_mod.load(f)

        assert meta["iteration"] == 42
        assert "network_config" in meta


# ===========================================================================
# B5: SelfPlayConfig and self_play_loop
# ===========================================================================


class TestSelfPlayConfig:
    """Test SelfPlayConfig defaults."""

    def test_defaults(self) -> None:
        """Verify all default configuration values."""
        config = SelfPlayConfig()

        assert config.num_iterations == 50
        assert config.games_per_iteration == 100
        assert config.mcts_simulations == 100
        assert config.num_players == 4
        assert config.temperature == 1.0
        assert config.eval_games == 50
        assert config.eval_frequency == 5
        assert config.acceptance_threshold == 0.55
        assert config.max_turns_per_game == 500
        assert isinstance(config.training_config, TrainingConfig)

    def test_training_config_is_independent(self) -> None:
        """Each SelfPlayConfig instance gets its own TrainingConfig."""
        c1 = SelfPlayConfig()
        c2 = SelfPlayConfig()
        assert c1.training_config is not c2.training_config

    def test_custom_training_config(self) -> None:
        """Can pass a custom TrainingConfig."""
        tc = TrainingConfig(learning_rate=0.01, num_epochs=5)
        config = SelfPlayConfig(training_config=tc)
        assert config.training_config.learning_rate == 0.01
        assert config.training_config.num_epochs == 5


class TestSelfPlayLoop:
    """Test the self_play_loop function."""

    def _small_config(self) -> SelfPlayConfig:
        """Return a fast self-play config for testing."""
        return SelfPlayConfig(
            num_iterations=1,
            games_per_iteration=1,
            mcts_simulations=5,
            num_players=4,
            temperature=1.0,
            eval_games=2,
            eval_frequency=1,
            max_turns_per_game=30,
            training_config=TrainingConfig(
                num_epochs=2,
                batch_size=8,
                device="cpu",
                validation_fraction=0.0,
            ),
        )

    def test_self_play_one_iteration_runs(self, tmp_path: pytest.TempPathFactory) -> None:
        """self_play_loop should complete one iteration without error."""
        config = self._small_config()
        save_dir = tmp_path / "selfplay"  # type: ignore[operator]

        self_play_loop(config, save_dir=save_dir)

        # Should have created an iteration checkpoint
        assert (save_dir / "mcts_iter_0").exists()  # type: ignore[operator]

    def test_self_play_creates_best_checkpoint(self, tmp_path: pytest.TempPathFactory) -> None:
        """First eval should create mcts_best checkpoint."""
        config = self._small_config()
        save_dir = tmp_path / "selfplay"  # type: ignore[operator]

        self_play_loop(config, save_dir=save_dir)

        # eval_frequency=1 means iteration 0 triggers eval, first eval always accepts
        assert (save_dir / "mcts_best").exists()  # type: ignore[operator]
        assert (save_dir / "mcts_best" / "network.pt").exists()  # type: ignore[operator]

    def test_self_play_iter_checkpoint_structure(self, tmp_path: pytest.TempPathFactory) -> None:
        """Iteration checkpoint should have all required files."""
        config = self._small_config()
        save_dir = tmp_path / "selfplay"  # type: ignore[operator]

        self_play_loop(config, save_dir=save_dir)

        ckpt = save_dir / "mcts_iter_0"  # type: ignore[operator]
        assert (ckpt / "network.pt").exists()  # type: ignore[operator]
        assert (ckpt / "buffer.npz").exists()  # type: ignore[operator]
        assert (ckpt / "metadata.json").exists()  # type: ignore[operator]

    def test_self_play_resume(self, tmp_path: pytest.TempPathFactory) -> None:
        """self_play_loop should resume from checkpoint and run more iterations."""
        config = self._small_config()
        save_dir = tmp_path / "selfplay"  # type: ignore[operator]

        # Run iteration 0
        self_play_loop(config, save_dir=save_dir)
        assert (save_dir / "mcts_iter_0").exists()  # type: ignore[operator]

        # Resume: should create iteration 1
        self_play_loop(
            config,
            save_dir=save_dir,
            resume_from=save_dir / "mcts_iter_0",  # type: ignore[operator]
        )
        assert (save_dir / "mcts_iter_1").exists()  # type: ignore[operator]
