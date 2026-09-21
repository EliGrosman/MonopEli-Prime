"""Training data generation and replay buffer for MCTS self-play.

Provides TrainingExample, ReplayBuffer, compute_outcomes, and
generate_training_data for the AlphaZero-style training loop.
"""

from __future__ import annotations

import random as stdlib_random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from monopoly_engine.game import MonopolyGame
from monopoly_engine.rules import calculate_net_worth
from monopoly_gym.action_space import ActionEncoder

from .features import extract_features
from .search import MCTSConfig, MCTSSearch, _auto_roll_dice


@dataclass
class TrainingExample:
    """A single training example from MCTS self-play.

    Attributes:
        features: Game state features (from extract_features).
        mcts_policy: Visit count distribution from MCTS search (149-dim).
        outcome: Actual game outcome for each player. +1 for winner,
                -1 for losers (or fractional based on net worth ranking).
        player_id: Which player's perspective this example is from.
    """

    features: NDArray[np.float32]       # shape: (feature_size,)
    mcts_policy: NDArray[np.float32]    # shape: (149,)
    outcome: NDArray[np.float32]        # shape: (num_players,)
    player_id: int


class ReplayBuffer:
    """Stores training examples from self-play games.

    Supports a maximum capacity with FIFO eviction using a deque.

    Attributes:
        capacity: Maximum number of examples.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._examples: deque[TrainingExample] = deque(maxlen=capacity)

    def add(self, example: TrainingExample) -> None:
        """Add a training example (FIFO eviction if at capacity)."""
        self._examples.append(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample. Clamped to buffer size.

        Returns:
            List of sampled TrainingExample objects.
        """
        n = min(batch_size, len(self._examples))
        indices = stdlib_random.sample(range(len(self._examples)), n)
        return [self._examples[i] for i in indices]

    def __len__(self) -> int:
        return len(self._examples)

    def save(self, path: str | Path) -> None:
        """Save buffer to disk as .npz file.

        Stores features, policies, outcomes, and player_ids as separate
        arrays for efficient loading.

        Args:
            path: File path to save to.
        """
        if len(self._examples) == 0:
            np.savez(
                path,
                features=np.empty((0,), dtype=np.float32),
                policies=np.empty((0,), dtype=np.float32),
                outcomes=np.empty((0,), dtype=np.float32),
                player_ids=np.empty((0,), dtype=np.int32),
                capacity=np.array([self.capacity]),
            )
            return

        features = np.array([e.features for e in self._examples], dtype=np.float32)
        policies = np.array([e.mcts_policy for e in self._examples], dtype=np.float32)
        outcomes = np.array([e.outcome for e in self._examples], dtype=np.float32)
        player_ids = np.array([e.player_id for e in self._examples], dtype=np.int32)

        np.savez(
            path,
            features=features,
            policies=policies,
            outcomes=outcomes,
            player_ids=player_ids,
            capacity=np.array([self.capacity]),
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        """Load buffer from disk.

        Args:
            path: File path to load from.

        Returns:
            Reconstructed ReplayBuffer with all examples.
        """
        data = np.load(path)
        capacity = int(data["capacity"][0])
        buffer = cls(capacity=capacity)

        features = data["features"]
        if features.ndim < 2 or features.shape[0] == 0:
            return buffer

        policies = data["policies"]
        outcomes = data["outcomes"]
        player_ids = data["player_ids"]

        for i in range(len(features)):
            example = TrainingExample(
                features=features[i].astype(np.float32),
                mcts_policy=policies[i].astype(np.float32),
                outcome=outcomes[i].astype(np.float32),
                player_id=int(player_ids[i]),
            )
            buffer.add(example)

        return buffer


def compute_outcomes(game: MonopolyGame) -> NDArray[np.float32]:
    """Compute per-player outcome values for a completed game.

    If the game has a winner: winner gets +1.0, all others get -1.0.
    If truncated (no winner): rank by net worth with linear interpolation.

    Args:
        game: The completed (or truncated) game.

    Returns:
        Array of shape (num_players,) with values in [-1, 1].
    """
    num_players = len(game.players)
    outcomes = np.zeros(num_players, dtype=np.float32)

    if game.game_over and game.winner is not None:
        outcomes[:] = -1.0
        outcomes[game.winner] = 1.0
    else:
        # Truncated -- rank by net worth
        net_worths = [
            calculate_net_worth(game.players[i], game.property_manager)
            for i in range(num_players)
        ]
        ranked = list(np.argsort(net_worths))  # ascending: index 0 = worst
        for rank, pid in enumerate(ranked):
            if num_players > 1:
                outcomes[pid] = 2.0 * rank / (num_players - 1) - 1.0
            else:
                outcomes[pid] = 0.0

    return outcomes


def generate_training_data(
    num_games: int,
    num_players: int = 4,
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    max_turns: int = 500,
    value_network: Any | None = None,
    opponent_policy: str = "rule_based",
    seed: int | None = None,
) -> ReplayBuffer:
    """Play games with MCTS and collect training data.

    For each game:
    1. Create a MonopolyGame with num_players
    2. At each of player 0's turns, run MCTS search and record:
       - Game features (from player 0's perspective)
       - MCTS visit count distribution (the search policy)
    3. All players use their respective policies for actions
    4. After game ends, fill in actual outcomes for all examples

    Args:
        num_games: Number of games to play.
        num_players: Players per game.
        mcts_simulations: Simulations per MCTS search.
        temperature: Temperature for MCTS action selection during data gen.
        max_turns: Maximum turns per game.
        value_network: Optional network for MCTS leaf evaluation.
        opponent_policy: How opponents play ("rule_based", "random").
        seed: Random seed for game creation.

    Returns:
        ReplayBuffer with collected training examples.
    """
    raise NotImplementedError("Search data generation is deferred pending chance-aware search certification")
    buffer = ReplayBuffer()
    encoder = ActionEncoder(enable_trades=False)

    mcts_config = MCTSConfig(
        num_simulations=mcts_simulations,
        max_rollout_depth=max(50, max_turns // 10),
        use_value_network=value_network is not None,
        temperature=temperature,
        opponent_policy=opponent_policy,
    )
    mcts = MCTSSearch(mcts_config, value_network=value_network)

    # Set up opponent agents (lazy, created on demand by _get_opponent_action)
    for game_idx in range(num_games):
        game_seed = seed + game_idx if seed is not None else None
        game = MonopolyGame(num_players=num_players, seed=game_seed)

        # Roll dice for the first player
        _auto_roll_dice(game)

        # Collect examples for this game (outcomes filled in later)
        game_examples: list[TrainingExample] = []

        turn = 0
        while not game.game_over and turn < max_turns:
            pid = game.decision_player
            player = game.players[pid]

            if player.bankrupt:
                # Skip bankrupt players
                from monopoly_engine.actions import EndTurn

                end = EndTurn(player_id=pid)
                valid, _ = end.validate(game)
                if valid:
                    game.apply_action(end.player_id, end)
                    _auto_roll_dice(game)
                turn += 1
                continue

            if pid == 0:
                # MCTS player: search and record training data
                visit_counts = mcts.search(game, player_id=0)

                if not visit_counts:
                    turn += 1
                    continue

                # Record training example (outcome is placeholder)
                features = extract_features(game, player_id=0, num_players=num_players)
                policy = mcts.get_policy_distribution(visit_counts)
                example = TrainingExample(
                    features=features,
                    mcts_policy=policy,
                    outcome=np.zeros(num_players, dtype=np.float32),  # filled later
                    player_id=0,
                )
                game_examples.append(example)

                # Select and execute action
                action_idx = mcts.select_action(visit_counts, temperature)
                action = encoder.decode(action_idx, pid, game)
                game.apply_action(action.player_id, action)
            else:
                # Opponent player: use configured policy
                action_idx = mcts._get_opponent_action(game, pid)
                action = encoder.decode(action_idx, pid, game)
                game.apply_action(action.player_id, action)

            # Auto-roll dice if turn changed
            prev_player = pid
            if game.decision_player != prev_player:
                _auto_roll_dice(game)

            turn += 1

        # Fill in actual outcomes for all examples from this game
        outcomes = compute_outcomes(game)
        for example in game_examples:
            example.outcome = outcomes
            buffer.add(example)

    return buffer
