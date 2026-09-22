"""Base agent interface for Monopoly RL environment."""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from monopoly_engine import Action, DecisionView, MonopolyGame


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

    def choose_decision(self, view: "DecisionView") -> "Action":
        """Choose from detached public state using the indexed compatibility policy."""
        from monopoly_engine import RejectTrade
        from monopoly_gym.action_space import ActionEncoder

        if view.pending_offer is not None:
            return RejectTrade(self.player_id, view.pending_offer.trade_id)
        properties = {
            item.position: SimpleNamespace(
                position=item.position,
                owner=item.owner,
                houses=item.houses,
                mortgaged=item.mortgaged,
            )
            for item in view.properties
        }
        property_manager = SimpleNamespace(
            properties=properties,
            get=lambda position: properties.get(position),
        )
        facade = SimpleNamespace(
            state=SimpleNamespace(phase=view.phase),
            players=[SimpleNamespace(**vars(player)) for player in view.players],
            property_manager=property_manager,
            current_player=view.turn_owner,
        )
        encoder = ActionEncoder()
        mask = np.zeros(encoder.action_space_size, dtype=np.bool_)
        for item in view.legal_actions:
            action = _public_action(vars(item))
            mask[encoder.encode(action)] = True
        index = int(self.choose_action(view.to_dict(), mask, facade))
        if not 0 <= index < len(mask) or not mask[index]:
            raise ValueError(f"Policy selected illegal public action index {index}")
        return encoder.decode(index, self.player_id, facade)

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


def _public_action(data: dict[str, Any]) -> "Action":
    from monopoly_engine.actions import (
        BuildHotel,
        BuildHouse,
        BuyProperty,
        DeclareBankruptcy,
        EndTurn,
        MortgageProperty,
        PassBuy,
        PayJailFine,
        RollDice,
        SellBuildingGroup,
        SellHotel,
        SellHouse,
        UnmortgageProperty,
        UseJailCard,
    )

    classes = {
        cls.__name__: cls
        for cls in (
            RollDice,
            BuyProperty,
            PassBuy,
            BuildHouse,
            BuildHotel,
            SellHouse,
            SellHotel,
            MortgageProperty,
            UnmortgageProperty,
            EndTurn,
            UseJailCard,
            PayJailFine,
            DeclareBankruptcy,
            SellBuildingGroup,
        )
    }
    cls = classes[data["type"]]
    kwargs = {key: value for key, value in data.items() if key != "type" and value is not None}
    return cls(**kwargs)
