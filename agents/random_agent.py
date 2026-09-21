"""Random agent that selects uniformly from valid actions."""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .base import Agent

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame


class RandomAgent(Agent):
    """Agent that selects uniformly random valid actions.

    This serves as a baseline for evaluating trained agents.
    """

    def __init__(
        self,
        player_id: int,
        seed: int | None = None,
    ) -> None:
        """Initialize random agent.

        Args:
            player_id: The player ID this agent controls
            seed: Random seed for reproducibility
        """
        super().__init__(player_id, name="Random")
        self.rng = np.random.default_rng(seed)

    def choose_action(
        self,
        observation: dict[str, Any],
        action_mask: NDArray[np.bool_],
        game: "MonopolyGame",
    ) -> int:
        """Select a random valid action.

        Args:
            observation: Dict observation (unused)
            action_mask: Binary mask of valid actions
            game: Game instance (unused)

        Returns:
            Random valid action index
        """
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            raise RuntimeError("Live decision has no legal actions")
        return int(self.rng.choice(valid_actions))
