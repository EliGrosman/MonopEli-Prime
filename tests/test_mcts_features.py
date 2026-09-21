"""Tests for MCTS feature extraction (B2)."""

from __future__ import annotations

import numpy as np

from monopoly_engine.actions import RollDice
from monopoly_engine.game import MonopolyGame

from mcts.features import extract_features, get_feature_size


# ===========================================================================
# B2: Feature Extraction tests
# ===========================================================================


class TestGetFeatureSize:
    """Test get_feature_size() function."""

    def test_feature_size_4_players(self) -> None:
        """4-player game should produce 279 features."""
        assert get_feature_size(4) == 279

    def test_feature_size_2_players(self) -> None:
        """2-player game should have fewer features (fewer opponents)."""
        size_2 = get_feature_size(2)
        size_4 = get_feature_size(4)
        assert size_2 < size_4
        # 2-player: 33 player + 1*34 opponent + 140 board + 4 game = 211
        assert size_2 == 211

    def test_feature_size_excludes_action_mask(self) -> None:
        """Feature size should be flat obs size minus 149 (action mask)."""
        from monopoly_gym.observation import get_flat_observation_size

        for n in (2, 3, 4):
            assert get_feature_size(n) == get_flat_observation_size(n) - 158 - 13


class TestExtractFeatures:
    """Test extract_features() function."""

    def test_feature_shape(self) -> None:
        """Features should have shape (feature_size,)."""
        game = MonopolyGame(num_players=4, seed=42)
        features = extract_features(game, player_id=0)

        assert features.shape == (279,)

    def test_feature_dtype(self) -> None:
        """Features should be float32."""
        game = MonopolyGame(num_players=4, seed=42)
        features = extract_features(game, player_id=0)

        assert features.dtype == np.float32

    def test_features_normalized(self) -> None:
        """All feature values should be in [0, 1]."""
        game = MonopolyGame(num_players=4, seed=42)

        for pid in range(4):
            features = extract_features(game, pid)
            assert features.min() >= 0.0, f"Player {pid}: min={features.min()}"
            assert features.max() <= 1.0, f"Player {pid}: max={features.max()}"

    def test_features_normalized_after_play(self) -> None:
        """Features should remain normalized after game actions."""
        game = MonopolyGame(num_players=4, seed=42)
        roll = RollDice(player_id=0)
        roll.execute(game)

        features = extract_features(game, player_id=0)
        assert features.min() >= 0.0
        assert features.max() <= 1.0

    def test_features_different_players(self) -> None:
        """Different player perspectives should produce different features."""
        game = MonopolyGame(num_players=4, seed=42)
        # Give players different money to ensure different perspectives
        game.players[0].money = 1500
        game.players[1].money = 500

        f0 = extract_features(game, player_id=0)
        f1 = extract_features(game, player_id=1)

        # Money feature (index 0) should differ
        assert f0[0] != f1[0]
        # Overall features should not be identical
        assert not np.allclose(f0, f1)

    def test_features_deterministic(self) -> None:
        """Same state should always produce same features."""
        game = MonopolyGame(num_players=4, seed=42)

        f1 = extract_features(game, player_id=0)
        f2 = extract_features(game, player_id=0)

        np.testing.assert_array_equal(f1, f2)

    def test_features_two_player_game(self) -> None:
        """extract_features should work for 2-player games."""
        game = MonopolyGame(num_players=2, seed=42)
        features = extract_features(game, player_id=0, num_players=2)

        assert features.shape == (get_feature_size(2),)
        assert features.dtype == np.float32
        assert features.min() >= 0.0
        assert features.max() <= 1.0

    def test_features_match_network_input_size(self) -> None:
        """Feature size should match what ValueNetwork expects."""
        from mcts.network import ValueNetwork

        feature_size = get_feature_size(4)
        game = MonopolyGame(num_players=4, seed=42)
        features = extract_features(game, player_id=0)

        # Features should be usable by the network
        net = ValueNetwork(input_size=feature_size)
        values, policy = net.predict(features)

        assert values.shape == (4,)
        assert policy.shape == (149,)

    def test_features_reflect_money_change(self) -> None:
        """Changing a player's money should change the features."""
        game = MonopolyGame(num_players=4, seed=42)

        f_before = extract_features(game, player_id=0).copy()
        game.players[0].money = 5000
        f_after = extract_features(game, player_id=0)

        # Money feature (index 0) should differ
        assert f_before[0] != f_after[0]

    def test_features_reflect_position_change(self) -> None:
        """Moving a player should change the position feature."""
        game = MonopolyGame(num_players=4, seed=42)

        f_before = extract_features(game, player_id=0).copy()
        game.players[0].position = 20
        f_after = extract_features(game, player_id=0)

        # Position feature (index 1) should differ
        assert f_before[1] != f_after[1]
