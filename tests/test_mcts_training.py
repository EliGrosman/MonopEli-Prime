"""Tests for MCTS value network training loop (B4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mcts.data import ReplayBuffer, TrainingExample
from mcts.features import get_feature_size
from mcts.network import ValueNetwork
from mcts.training import TrainingConfig, TrainingStats, train_value_network


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
