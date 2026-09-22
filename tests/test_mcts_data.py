"""Tests for MCTS training data generation (B3)."""

from __future__ import annotations

import numpy as np
import pytest

from mcts.data import ReplayBuffer, TrainingExample, compute_outcomes, generate_training_data
from mcts.features import get_feature_size
from monopoly_engine.game import MonopolyGame

# ===========================================================================
# B3-1: TrainingExample
# ===========================================================================


class TestTrainingExample:
    """Test the TrainingExample dataclass."""

    def test_create_example(self) -> None:
        """Can create a TrainingExample with correct attributes."""
        features = np.zeros(279, dtype=np.float32)
        policy = np.ones(149, dtype=np.float32) / 149
        outcome = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        ex = TrainingExample(
            features=features,
            mcts_policy=policy,
            outcome=outcome,
            player_id=0,
        )

        assert ex.features.shape == (279,)
        assert ex.mcts_policy.shape == (149,)
        assert ex.outcome.shape == (4,)
        assert ex.player_id == 0

    def test_example_dtypes(self) -> None:
        """All arrays should be float32."""
        ex = TrainingExample(
            features=np.zeros(279, dtype=np.float32),
            mcts_policy=np.zeros(149, dtype=np.float32),
            outcome=np.zeros(4, dtype=np.float32),
            player_id=2,
        )

        assert ex.features.dtype == np.float32
        assert ex.mcts_policy.dtype == np.float32
        assert ex.outcome.dtype == np.float32


# ===========================================================================
# B3-2: ReplayBuffer
# ===========================================================================


class TestReplayBuffer:
    """Test ReplayBuffer storage and sampling."""

    def _make_example(self, player_id: int = 0) -> TrainingExample:
        """Helper to create a dummy training example."""
        return TrainingExample(
            features=np.random.default_rng(player_id).random(279).astype(np.float32),
            mcts_policy=np.random.default_rng(player_id).random(149).astype(np.float32),
            outcome=np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            player_id=player_id,
        )

    def test_add_and_len(self) -> None:
        """Buffer should track length correctly."""
        buf = ReplayBuffer(capacity=100)

        assert len(buf) == 0
        buf.add(self._make_example(0))
        assert len(buf) == 1
        buf.add(self._make_example(1))
        assert len(buf) == 2

    def test_capacity_eviction(self) -> None:
        """Buffer should evict oldest examples at capacity."""
        buf = ReplayBuffer(capacity=3)

        for i in range(5):
            buf.add(self._make_example(i))

        assert len(buf) == 3
        # The first two should have been evicted; last three remain
        examples = buf.sample(3)
        player_ids = sorted(e.player_id for e in examples)
        assert player_ids == [2, 3, 4]

    def test_sample_returns_correct_count(self) -> None:
        """Sample should return requested number of examples."""
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.add(self._make_example(i))

        sampled = buf.sample(5)
        assert len(sampled) == 5

    def test_sample_clamps_to_buffer_size(self) -> None:
        """Sample requesting more than available should return all."""
        buf = ReplayBuffer(capacity=100)
        for i in range(3):
            buf.add(self._make_example(i))

        sampled = buf.sample(100)
        assert len(sampled) == 3

    def test_sample_returns_training_examples(self) -> None:
        """Sample should return TrainingExample objects."""
        buf = ReplayBuffer(capacity=100)
        buf.add(self._make_example(0))

        sampled = buf.sample(1)
        assert isinstance(sampled[0], TrainingExample)


# ===========================================================================
# B3-3: ReplayBuffer Save/Load
# ===========================================================================


class TestReplayBufferSaveLoad:
    """Test ReplayBuffer serialization."""

    def _make_example(self, idx: int) -> TrainingExample:
        """Helper to create a deterministic training example."""
        rng = np.random.default_rng(idx)
        return TrainingExample(
            features=rng.random(279).astype(np.float32),
            mcts_policy=rng.random(149).astype(np.float32),
            outcome=np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            player_id=idx % 4,
        )

    def test_save_load_round_trip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Save and load should preserve all examples."""
        buf = ReplayBuffer(capacity=1000)
        for i in range(10):
            buf.add(self._make_example(i))

        path = tmp_path / "buffer.npz"  # type: ignore[operator]
        buf.save(path)

        loaded = ReplayBuffer.load(path)
        assert len(loaded) == 10
        assert loaded.capacity == 1000

    def test_save_load_preserves_data(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loaded examples should have identical data."""
        buf = ReplayBuffer(capacity=100)
        original = self._make_example(42)
        buf.add(original)

        path = tmp_path / "buffer.npz"  # type: ignore[operator]
        buf.save(path)

        loaded = ReplayBuffer.load(path)
        sampled = loaded.sample(1)
        restored = sampled[0]

        np.testing.assert_array_almost_equal(restored.features, original.features)
        np.testing.assert_array_almost_equal(restored.mcts_policy, original.mcts_policy)
        np.testing.assert_array_almost_equal(restored.outcome, original.outcome)
        assert restored.player_id == original.player_id

    def test_save_load_empty_buffer(self, tmp_path: pytest.TempPathFactory) -> None:
        """Save/load of empty buffer should work."""
        buf = ReplayBuffer(capacity=500)

        path = tmp_path / "empty.npz"  # type: ignore[operator]
        buf.save(path)

        loaded = ReplayBuffer.load(path)
        assert len(loaded) == 0
        assert loaded.capacity == 500

    def test_save_load_preserves_capacity(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loaded buffer should have the same capacity."""
        buf = ReplayBuffer(capacity=42)
        buf.add(self._make_example(0))

        path = tmp_path / "buffer.npz"  # type: ignore[operator]
        buf.save(path)

        loaded = ReplayBuffer.load(path)
        assert loaded.capacity == 42


# ===========================================================================
# B3-4: compute_outcomes
# ===========================================================================


class TestComputeOutcomes:
    """Test outcome computation for finished games."""

    def test_winner_gets_plus_one(self) -> None:
        """Winner should get +1.0, losers -1.0."""
        game = MonopolyGame(num_players=4, seed=42)
        game.game_over = True
        game.winner = 2

        outcomes = compute_outcomes(game)

        assert outcomes.shape == (4,)
        assert outcomes[2] == pytest.approx(1.0)
        for i in [0, 1, 3]:
            assert outcomes[i] == pytest.approx(-1.0)

    def test_winner_two_player(self) -> None:
        """Two-player: winner +1, loser -1."""
        game = MonopolyGame(num_players=2, seed=42)
        game.game_over = True
        game.winner = 0

        outcomes = compute_outcomes(game)

        assert outcomes.shape == (2,)
        assert outcomes[0] == pytest.approx(1.0)
        assert outcomes[1] == pytest.approx(-1.0)

    def test_truncated_game_ranking(self) -> None:
        """Truncated game: outcomes based on net worth ranking."""
        game = MonopolyGame(num_players=4, seed=42)
        # Give different amounts of money to create a clear ranking
        game.players[0].money = 100
        game.players[1].money = 500
        game.players[2].money = 1500
        game.players[3].money = 3000

        outcomes = compute_outcomes(game)

        assert outcomes.shape == (4,)
        # Lowest net worth gets -1, highest gets +1
        assert outcomes[0] == pytest.approx(-1.0)
        assert outcomes[3] == pytest.approx(1.0)
        # Middle players get intermediate values
        assert -1.0 < outcomes[1] < 1.0
        assert -1.0 < outcomes[2] < 1.0
        # Ranking should be monotonic with net worth
        assert outcomes[0] < outcomes[1] < outcomes[2] < outcomes[3]

    def test_truncated_values_in_range(self) -> None:
        """All truncated outcomes should be in [-1, 1]."""
        game = MonopolyGame(num_players=4, seed=42)

        outcomes = compute_outcomes(game)

        assert outcomes.min() >= -1.0
        assert outcomes.max() <= 1.0

    def test_outcome_dtype(self) -> None:
        """Outcomes should be float32."""
        game = MonopolyGame(num_players=4, seed=42)
        game.game_over = True
        game.winner = 0

        outcomes = compute_outcomes(game)
        assert outcomes.dtype == np.float32


# ===========================================================================
# B3-5: generate_training_data (integration)
# ===========================================================================


class TestGenerateTrainingData:
    """Test the full data generation pipeline."""

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_generates_nonempty_buffer(self) -> None:
        """Should produce at least some training examples."""
        buffer = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )

        assert len(buffer) > 0

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_example_shapes(self) -> None:
        """Training examples should have correct array shapes."""
        buffer = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )

        example = buffer.sample(1)[0]
        feature_size = get_feature_size(4)

        assert example.features.shape == (feature_size,)
        assert example.mcts_policy.shape == (149,)
        assert example.outcome.shape == (4,)
        assert example.player_id == 0

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_policy_is_distribution(self) -> None:
        """MCTS policy should sum to ~1.0 (probability distribution)."""
        buffer = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )

        example = buffer.sample(1)[0]
        policy_sum = example.mcts_policy.sum()

        # Should be approximately 1.0 (visit count distribution)
        assert policy_sum == pytest.approx(1.0, abs=0.01)

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_outcomes_filled_in(self) -> None:
        """Outcomes should be non-zero (filled in after game)."""
        buffer = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )

        example = buffer.sample(1)[0]
        # At least some outcome values should be non-zero
        assert np.any(example.outcome != 0)

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_multiple_games(self) -> None:
        """Running multiple games should produce more examples."""
        buf1 = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )
        buf2 = generate_training_data(
            num_games=2,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )

        assert len(buf2) >= len(buf1)

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should generally produce different data."""
        buf1 = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )
        buf2 = generate_training_data(
            num_games=1,
            num_players=4,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=99,
            opponent_policy="random",
        )

        # Different seeds create different games, so at minimum
        # the examples should differ (features come from different game states)
        if len(buf1) > 0 and len(buf2) > 0:
            ex1 = buf1.sample(1)[0]
            ex2 = buf2.sample(1)[0]
            assert not np.allclose(ex1.features, ex2.features)

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="Deferred search/trading positive path; foundation rejects this mode",
    )
    def test_two_player_game(self) -> None:
        """Should work with 2-player games."""
        buffer = generate_training_data(
            num_games=1,
            num_players=2,
            mcts_simulations=5,
            temperature=1.0,
            max_turns=20,
            seed=42,
            opponent_policy="random",
        )

        if len(buffer) > 0:
            example = buffer.sample(1)[0]
            feature_size = get_feature_size(2)
            assert example.features.shape == (feature_size,)
            assert example.outcome.shape == (2,)
