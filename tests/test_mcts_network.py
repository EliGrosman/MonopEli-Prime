"""Tests for MCTS value/policy network (B1)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn

from mcts.network import ValueNetwork

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE = 279  # 4-player game features (428 - 149 action mask)
NUM_PLAYERS = 4
ACTION_SIZE = 149


# ===========================================================================
# B1: ValueNetwork tests
# ===========================================================================


class TestValueNetworkInit:
    """Test network initialization and structure."""

    def test_default_construction(self) -> None:
        """Network should construct with default parameters."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        assert net.input_size == INPUT_SIZE
        assert net.num_players == 4
        assert net.action_size == 149
        assert net.hidden_size == 256
        assert net.trunk_layers == 3

    def test_custom_construction(self) -> None:
        """Network should accept custom parameters."""
        net = ValueNetwork(
            input_size=100,
            num_players=2,
            action_size=50,
            hidden_size=128,
            trunk_layers=2,
        )
        assert net.input_size == 100
        assert net.num_players == 2
        assert net.action_size == 50
        assert net.hidden_size == 128
        assert net.trunk_layers == 2

    def test_has_trunk_value_policy(self) -> None:
        """Network should have trunk, value_head, and policy_head."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        assert hasattr(net, "trunk")
        assert hasattr(net, "value_head")
        assert hasattr(net, "policy_head")

    def test_parameter_count_reasonable(self) -> None:
        """Network should have a reasonable number of parameters."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        total_params = sum(p.numel() for p in net.parameters())
        # Should be in the range of ~100k-500k for this size
        assert 10_000 < total_params < 1_000_000


class TestValueNetworkForward:
    """Test forward pass shapes and value ranges."""

    def test_forward_shapes(self) -> None:
        """Forward pass should produce correct output shapes."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.eval()

        batch_size = 8
        x = torch.randn(batch_size, INPUT_SIZE)

        values, policy_logits = net(x)

        assert values.shape == (batch_size, NUM_PLAYERS)
        assert policy_logits.shape == (batch_size, ACTION_SIZE)

    def test_forward_single_sample(self) -> None:
        """Forward pass should work with batch_size=1."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.eval()

        x = torch.randn(1, INPUT_SIZE)
        values, policy_logits = net(x)

        assert values.shape == (1, NUM_PLAYERS)
        assert policy_logits.shape == (1, ACTION_SIZE)

    def test_value_range(self) -> None:
        """Values should be in [-1, 1] due to tanh activation."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.eval()

        x = torch.randn(16, INPUT_SIZE)
        values, _ = net(x)

        assert (values >= -1.0).all()
        assert (values <= 1.0).all()

    def test_policy_masked(self) -> None:
        """Invalid actions should get -inf logits when mask is provided."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.eval()

        x = torch.randn(4, INPUT_SIZE)
        mask = torch.zeros(4, ACTION_SIZE, dtype=torch.bool)
        # Only actions 0, 5, 10 are valid
        mask[:, 0] = True
        mask[:, 5] = True
        mask[:, 10] = True

        _, policy_logits = net(x, action_mask=mask)

        # Invalid actions should be -inf
        for i in range(ACTION_SIZE):
            if i not in (0, 5, 10):
                assert (policy_logits[:, i] == float("-inf")).all()

    def test_policy_unmasked(self) -> None:
        """Without mask, all logits should be finite."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.eval()

        x = torch.randn(4, INPUT_SIZE)
        _, policy_logits = net(x)

        assert torch.isfinite(policy_logits).all()

    def test_masked_softmax_valid_distribution(self) -> None:
        """Softmax of masked logits should give valid probs for valid actions."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.eval()

        x = torch.randn(2, INPUT_SIZE)
        mask = torch.zeros(2, ACTION_SIZE, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 1] = True
        mask[:, 2] = True

        _, policy_logits = net(x, action_mask=mask)
        probs = torch.softmax(policy_logits, dim=-1)

        # Valid actions should have non-zero probability
        for i in range(3):
            assert (probs[:, i] > 0).all()

        # Invalid actions should have ~0 probability
        assert (probs[:, 3:] < 1e-6).all()

        # Should sum to ~1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


class TestValueNetworkPredict:
    """Test numpy-based predict() interface."""

    def test_predict_single_sample(self) -> None:
        """predict() should work with a single feature vector."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        features = np.random.randn(INPUT_SIZE).astype(np.float32)
        values, policy = net.predict(features)

        assert values.shape == (NUM_PLAYERS,)
        assert policy.shape == (ACTION_SIZE,)
        assert values.dtype == np.float32
        assert policy.dtype == np.float32

    def test_predict_batch(self) -> None:
        """predict() should work with a batch of features."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        features = np.random.randn(8, INPUT_SIZE).astype(np.float32)
        values, policy = net.predict(features)

        assert values.shape == (8, NUM_PLAYERS)
        assert policy.shape == (8, ACTION_SIZE)

    def test_predict_values_bounded(self) -> None:
        """Predicted values should be in [-1, 1]."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        features = np.random.randn(INPUT_SIZE).astype(np.float32)
        values, _ = net.predict(features)

        assert np.all(values >= -1.0)
        assert np.all(values <= 1.0)

    def test_predict_policy_is_distribution(self) -> None:
        """Predicted policy should be a valid probability distribution."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        features = np.random.randn(INPUT_SIZE).astype(np.float32)
        _, policy = net.predict(features)

        assert np.all(policy >= 0)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_predict_with_mask(self) -> None:
        """predict() with action mask should zero out invalid actions."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        features = np.random.randn(INPUT_SIZE).astype(np.float32)
        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        mask[0] = True
        mask[10] = True

        _, policy = net.predict(features, action_mask=mask)

        # Invalid actions should be near zero
        for i in range(ACTION_SIZE):
            if i not in (0, 10):
                assert policy[i] < 1e-6

        # Valid actions should have non-zero probability
        assert policy[0] > 0
        assert policy[10] > 0

    def test_predict_no_gradients(self) -> None:
        """predict() should not accumulate gradients."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        features = np.random.randn(INPUT_SIZE).astype(np.float32)
        net.predict(features)

        # No parameter should have gradients
        for param in net.parameters():
            assert param.grad is None


class TestValueNetworkSaveLoad:
    """Test save/load checkpoint round-trip."""

    def test_save_load_roundtrip(self) -> None:
        """Saving and loading should preserve weights and config."""
        net = ValueNetwork(input_size=INPUT_SIZE, num_players=4)

        features = np.random.randn(INPUT_SIZE).astype(np.float32)
        original_values, original_policy = net.predict(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            net.save(path)

            loaded = ValueNetwork.load(path)

        loaded_values, loaded_policy = loaded.predict(features)

        np.testing.assert_allclose(loaded_values, original_values, atol=1e-6)
        np.testing.assert_allclose(loaded_policy, original_policy, atol=1e-6)

    def test_save_load_preserves_config(self) -> None:
        """Loaded network should have same config as original."""
        net = ValueNetwork(
            input_size=100,
            num_players=2,
            action_size=50,
            hidden_size=128,
            trunk_layers=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            net.save(path)

            loaded = ValueNetwork.load(path)

        assert loaded.input_size == 100
        assert loaded.num_players == 2
        assert loaded.action_size == 50
        assert loaded.hidden_size == 128
        assert loaded.trunk_layers == 2

    def test_save_creates_file(self) -> None:
        """save() should create a file at the specified path."""
        net = ValueNetwork(input_size=INPUT_SIZE)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            assert not path.exists()
            net.save(path)
            assert path.exists()


class TestValueNetworkTraining:
    """Test that the network can be trained (gradient flow)."""

    def test_gradients_flow(self) -> None:
        """Loss.backward() should produce gradients on all parameters."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.train()

        x = torch.randn(4, INPUT_SIZE)
        target_values = torch.randn(4, NUM_PLAYERS)
        target_policy = torch.softmax(torch.randn(4, ACTION_SIZE), dim=-1)

        values, policy_logits = net(x)

        value_loss = torch.nn.functional.mse_loss(values, target_values)
        policy_loss = -(target_policy * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        loss = value_loss + policy_loss
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_one_step_reduces_loss(self) -> None:
        """A single optimizer step should reduce training loss."""
        net = ValueNetwork(input_size=INPUT_SIZE)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        x = torch.randn(16, INPUT_SIZE)
        target_values = torch.zeros(16, NUM_PLAYERS)
        target_policy = torch.ones(16, ACTION_SIZE) / ACTION_SIZE

        # Initial loss
        values, policy_logits = net(x)
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        policy_loss = -(target_policy * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        initial_loss = (value_loss + policy_loss).item()

        # Training step
        optimizer.zero_grad()
        values, policy_logits = net(x)
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        policy_loss = -(target_policy * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()

        # Post-step loss
        with torch.no_grad():
            values, policy_logits = net(x)
            value_loss = torch.nn.functional.mse_loss(values, target_values)
            policy_loss = (
                -(target_policy * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
            )
            final_loss = (value_loss + policy_loss).item()

        assert final_loss < initial_loss
