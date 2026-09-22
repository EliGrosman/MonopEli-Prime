"""Reward function implementations for Monopoly RL environment.

This module provides configurable reward functions for training RL agents.
Rewards can be sparse (win/loss only) or dense (incremental progress).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame


@dataclass
class RewardConfig:
    """Configuration for reward calculation.

    Attributes:
        win_reward: Reward for winning the game
        loss_reward: Penalty for losing (bankruptcy or game over)
        net_worth_scale: Scale factor for net worth changes (dense rewards)
        monopoly_bonus: Bonus for completing a monopoly
        property_bonus: Bonus for acquiring a property
        building_bonus: Bonus for building a house/hotel
        bankruptcy_penalty: Extra penalty when going bankrupt
        invalid_action_penalty: Penalty for attempting invalid actions
    """

    win_reward: float = 1.0
    loss_reward: float = -1.0
    net_worth_scale: float = 0.001
    monopoly_bonus: float = 0.5
    property_bonus: float = 0.1
    building_bonus: float = 0.05
    bankruptcy_penalty: float = -2.0
    invalid_action_penalty: float = -0.01


def calculate_sparse_reward(
    game: "MonopolyGame",
    player_id: int,
    config: RewardConfig | None = None,
) -> float:
    """Calculate sparse reward (only on game end).

    Args:
        game: Current game state
        player_id: Player to calculate reward for
        config: Reward configuration (uses defaults if None)

    Returns:
        Reward value (+1 for win, -1 for loss, 0 otherwise)
    """
    if config is None:
        config = RewardConfig()

    if not game.game_over:
        return 0.0

    if game.winner == player_id:
        return config.win_reward
    else:
        return config.loss_reward


def calculate_dense_reward(
    game: "MonopolyGame",
    player_id: int,
    prev_net_worth: int,
    config: RewardConfig | None = None,
) -> tuple[float, int]:
    """Calculate dense reward (incremental progress).

    Args:
        game: Current game state
        player_id: Player to calculate reward for
        prev_net_worth: Player's net worth before this step
        config: Reward configuration

    Returns:
        Tuple of (reward, current_net_worth) for tracking
    """
    from monopoly_engine import calculate_net_worth

    if config is None:
        config = RewardConfig()

    player = game.players[player_id]
    current_worth = calculate_net_worth(player, game.property_manager)

    # Base reward from net worth change
    reward = (current_worth - prev_net_worth) * config.net_worth_scale

    # Win bonus
    if game.game_over and game.winner == player_id:
        reward += config.win_reward * 10  # Large bonus for winning
    elif game.game_over:
        reward += config.loss_reward

    # Bankruptcy penalty
    if player.bankrupt:
        reward += config.bankruptcy_penalty

    return reward, current_worth


def count_monopolies(game: "MonopolyGame", player_id: int) -> int:
    """Count the number of monopolies a player has.

    Args:
        game: Current game state
        player_id: Player to count monopolies for

    Returns:
        Number of complete color group monopolies (excluding railroad/utility)
    """
    from monopoly_engine.types import PropertyColor

    count = 0
    for color in PropertyColor:
        if color in (PropertyColor.RAILROAD, PropertyColor.UTILITY):
            continue
        if game.property_manager.has_monopoly(player_id, color):
            count += 1
    return count


@dataclass
class RewardTracker:
    """Tracks reward state across steps for dense rewards.

    Usage:
        tracker = RewardTracker(num_players=4)
        tracker.reset(game)

        # After each step:
        reward = tracker.calculate_reward(game, player_id, reward_type="dense")

    Attributes:
        num_players: Number of players in the game
        config: Reward configuration settings
    """

    num_players: int
    config: RewardConfig = field(default_factory=RewardConfig)
    _prev_net_worth: dict[int, int] = field(default_factory=dict)
    _prev_monopolies: dict[int, int] = field(default_factory=dict)

    def reset(self, game: "MonopolyGame") -> None:
        """Reset tracking state for a new game.

        Args:
            game: The game state to initialize tracking from
        """
        from monopoly_engine import calculate_net_worth

        self._prev_net_worth = {}
        self._prev_monopolies = {}

        for player_id in range(self.num_players):
            player = game.players[player_id]
            self._prev_net_worth[player_id] = calculate_net_worth(player, game.property_manager)
            self._prev_monopolies[player_id] = count_monopolies(game, player_id)

    def calculate_reward(
        self,
        game: "MonopolyGame",
        player_id: int,
        reward_type: str = "sparse",
    ) -> float:
        """Calculate reward for a player.

        Args:
            game: Current game state
            player_id: Player to calculate reward for
            reward_type: "sparse" or "dense"

        Returns:
            Reward value

        Raises:
            ValueError: If reward_type is not "sparse" or "dense"
        """
        if reward_type == "sparse":
            return calculate_sparse_reward(game, player_id, self.config)
        elif reward_type == "dense":
            reward, new_worth = calculate_dense_reward(
                game,
                player_id,
                self._prev_net_worth.get(player_id, 1500),
                self.config,
            )

            # Monopoly bonus
            new_monopolies = count_monopolies(game, player_id)
            prev_monopolies = self._prev_monopolies.get(player_id, 0)
            if new_monopolies > prev_monopolies:
                reward += self.config.monopoly_bonus * (new_monopolies - prev_monopolies)

            # Update tracking
            self._prev_net_worth[player_id] = new_worth
            self._prev_monopolies[player_id] = new_monopolies

            return reward
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
