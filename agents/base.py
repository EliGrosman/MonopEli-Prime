"""Base agent interface for Monopoly RL environment."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame


class Agent(ABC):
    """Abstract base class for Monopoly agents.

    All agent types (random, rule-based, RL) implement this interface.
    The interface is designed to be forward-compatible with Phase 2.5
    trade actions - trades are just additional actions in the mask.
    """

    def __init__(self, player_id: int, name: str = "Agent") -> None:
        """Initialize agent.

        Args:
            player_id: The player ID this agent controls
            name: Human-readable name for the agent
        """
        self.player_id = player_id
        self.name = name

    @abstractmethod
    def choose_action(
        self,
        observation: dict[str, Any],
        action_mask: NDArray[np.bool_],
        game: "MonopolyGame",
    ) -> int:
        """Select an action given the current observation.

        Args:
            observation: Dict observation from the environment
            action_mask: Binary mask of valid actions (True = valid)
            game: The game instance (for rule-based agents that need full state)

        Returns:
            Action index (0-148 in Phase 2, will expand in Phase 2.5)
        """
        pass

    def reset(self) -> None:
        """Reset agent state at the start of a new game.

        Override in subclasses that maintain state across turns.
        """
        pass

    def notify_result(
        self,
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Receive feedback on the result of an action.

        Override in learning agents to store experiences.

        Args:
            action: The action that was taken
            reward: The reward received
            next_observation: The resulting observation
            terminated: Whether the episode ended (win/loss/bankruptcy)
            truncated: Whether the episode was cut short (max turns)
        """
        pass
