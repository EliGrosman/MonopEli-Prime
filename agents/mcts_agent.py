"""MCTS-based agent for Monopoly.

Implements the Agent ABC using Monte Carlo Tree Search with an optional
learned value network (AlphaZero-style). Without a network, uses random
rollouts for leaf evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mcts.search import MCTSConfig, MCTSSearch

from .base import Agent

if TYPE_CHECKING:
    from monopoly_engine.game import MonopolyGame


@dataclass
class SearchStats:
    """Per-game search statistics tracked by MCTSAgent.

    Attributes:
        total_searches: Number of MCTS searches performed this game.
        total_time_sec: Total wall-clock seconds spent searching.
        total_simulations: Total simulation count across all searches.
    """

    total_searches: int = 0
    total_time_sec: float = 0.0
    total_simulations: int = 0

    @property
    def avg_time_per_move(self) -> float:
        """Average seconds per MCTS search."""
        if self.total_searches == 0:
            return 0.0
        return self.total_time_sec / self.total_searches

    @property
    def avg_simulations_per_move(self) -> float:
        """Average simulations per MCTS search."""
        if self.total_searches == 0:
            return 0.0
        return self.total_simulations / self.total_searches


class MCTSAgent(Agent):
    """Agent that uses Monte Carlo Tree Search to select actions.

    Can operate in two modes:
    1. Pure MCTS with random rollouts (no value network)
    2. MCTS with learned value network (AlphaZero-style)

    The agent clones the game state at each call to choose_action so
    the live game is never modified during search.

    Attributes:
        num_simulations: Number of MCTS simulations per move.
        temperature: Action selection temperature (0 = deterministic).
        search_stats: Accumulated statistics for the current game.
    """

    def __init__(
        self,
        player_id: int,
        num_simulations: int = 200,
        temperature: float = 0.0,
        network_path: str | Path | None = None,
        opponent_policy: str = "rule_based",
        exploration_constant: float = 1.41,
        max_rollout_depth: int = 500,
    ) -> None:
        """Initialize MCTS agent.

        Args:
            player_id: The player ID this agent controls.
            num_simulations: MCTS simulations per move.
            temperature: 0 = deterministic (argmax), >0 = proportional to
                visit counts raised to 1/temperature.
            network_path: Path to ValueNetwork checkpoint. None = pure MCTS
                with random rollouts (no learned evaluation).
            opponent_policy: Policy for simulating opponent moves during search
                ("rule_based" or "random").
            exploration_constant: UCB1 c constant (higher = more exploration).
            max_rollout_depth: Maximum depth for random rollout simulations.
        """
        super().__init__(player_id=player_id, name=f"MCTSAgent(p{player_id})")

        self.num_simulations = num_simulations
        self.temperature = temperature
        self.search_stats = SearchStats()

        # Load network if a path is provided (requires torch)
        network: Any = None
        if network_path is not None:
            from mcts.network import ValueNetwork

            network = ValueNetwork.load(Path(network_path))

        config = MCTSConfig(
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            max_rollout_depth=max_rollout_depth,
            use_value_network=network is not None,
            temperature=temperature,
            opponent_policy=opponent_policy,
        )
        self._mcts = MCTSSearch(config, value_network=network)

    def choose_action(
        self,
        observation: dict[str, Any],
        action_mask: NDArray[np.bool_],
        game: MonopolyGame,
    ) -> int:
        """Select action using MCTS search.

        Runs num_simulations MCTS simulations from the current game state,
        then selects an action according to the visit count distribution and
        the configured temperature.

        Args:
            observation: Dict observation (unused — MCTS reads from game directly).
            action_mask: Binary mask of valid actions (used as fallback if MCTS
                returns no visit counts).
            game: The live game instance (cloned internally; never modified).

        Returns:
            Action index in [0, 148].
        """
        t0 = time.monotonic()
        visit_counts = self._mcts.search(game, player_id=self.player_id)
        elapsed = time.monotonic() - t0

        # Update stats
        self.search_stats.total_searches += 1
        self.search_stats.total_time_sec += elapsed
        self.search_stats.total_simulations += self.num_simulations

        if not visit_counts:
            # Fallback: pick uniformly from valid actions
            valid_actions = np.where(action_mask)[0]
            return int(np.random.choice(valid_actions))

        return self._mcts.select_action(visit_counts, self.temperature)

    def reset(self) -> None:
        """Reset per-game search statistics."""
        self.search_stats = SearchStats()

    @property
    def network(self) -> Any:
        """The loaded value network, or None for pure-rollout mode."""
        return self._mcts.value_network
