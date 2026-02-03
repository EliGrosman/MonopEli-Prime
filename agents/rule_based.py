"""Rule-based heuristic agents for Monopoly."""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .base import Agent

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame


class RuleBasedAgent(Agent):
    """Heuristic agent using simple strategy rules.

    Configurable thresholds control aggressiveness.
    """

    def __init__(
        self,
        player_id: int,
        buy_threshold: float = 0.5,
        build_threshold: float = 0.3,
    ) -> None:
        """Initialize rule-based agent.

        Args:
            player_id: The player ID this agent controls
            buy_threshold: Buy if cost < this fraction of money
            build_threshold: Build if cost < this fraction of money
        """
        super().__init__(player_id, name="RuleBased")
        self.buy_threshold = buy_threshold
        self.build_threshold = build_threshold

    def choose_action(
        self,
        observation: dict[str, Any],
        action_mask: NDArray[np.bool_],
        game: "MonopolyGame",
    ) -> int:
        """Select action based on heuristic rules."""
        from monopoly_engine.rules import get_building_cost, get_unmortgage_cost
        from monopoly_gym.action_space import (
            BUYABLE_POSITIONS,
            DEVELOPABLE_POSITIONS,
            OFFSET_BUILD_HOTEL,
            OFFSET_BUILD_HOUSE,
            OFFSET_BUY_PROPERTY,
            OFFSET_END_TURN,
            OFFSET_PASS_BUY,
            OFFSET_PAY_JAIL_FINE,
            OFFSET_UNMORTGAGE,
            OFFSET_USE_JAIL_CARD,
        )

        player = game.players[self.player_id]

        # Priority 0: Buy property if affordable
        if action_mask[OFFSET_BUY_PROPERTY]:
            prop = game.property_manager.get(player.position)
            if prop is not None:
                from monopoly_engine.rules import get_property_cost

                prop_cost = get_property_cost(player.position)
                if prop_cost < player.money * self.buy_threshold:
                    return OFFSET_BUY_PROPERTY
            # If can't afford threshold, pass
            return OFFSET_PASS_BUY

        # Priority 1: Get out of jail
        if player.in_jail:
            if action_mask[OFFSET_USE_JAIL_CARD]:
                return OFFSET_USE_JAIL_CARD
            if action_mask[OFFSET_PAY_JAIL_FINE]:
                return OFFSET_PAY_JAIL_FINE

        # Priority 2: Build houses/hotels on monopolies
        if player.money > 200:  # Keep cash reserve
            build_house_dim = len(DEVELOPABLE_POSITIONS)
            for i in range(build_house_dim):
                if action_mask[OFFSET_BUILD_HOUSE + i]:
                    prop_pos = DEVELOPABLE_POSITIONS[i]
                    cost = get_building_cost(prop_pos)
                    if cost < player.money * self.build_threshold:
                        return OFFSET_BUILD_HOUSE + i

            # Build hotels
            for i in range(build_house_dim):
                if action_mask[OFFSET_BUILD_HOTEL + i]:
                    prop_pos = DEVELOPABLE_POSITIONS[i]
                    cost = get_building_cost(prop_pos)
                    if cost < player.money * self.build_threshold:
                        return OFFSET_BUILD_HOTEL + i

        # Priority 3: Unmortgage properties if affordable
        mortgage_dim = len(BUYABLE_POSITIONS)
        for i in range(mortgage_dim):
            if action_mask[OFFSET_UNMORTGAGE + i]:
                prop_pos = BUYABLE_POSITIONS[i]
                cost = get_unmortgage_cost(prop_pos)
                if cost < player.money * 0.3:
                    return OFFSET_UNMORTGAGE + i

        # Default: End turn
        return OFFSET_END_TURN


class AggressiveAgent(RuleBasedAgent):
    """More aggressive - builds and buys more eagerly."""

    def __init__(self, player_id: int) -> None:
        """Initialize aggressive agent.

        Args:
            player_id: The player ID this agent controls
        """
        super().__init__(
            player_id,
            buy_threshold=0.8,
            build_threshold=0.5,
        )
        self.name = "Aggressive"


class ConservativeAgent(RuleBasedAgent):
    """More conservative - keeps larger cash reserve."""

    def __init__(self, player_id: int) -> None:
        """Initialize conservative agent.

        Args:
            player_id: The player ID this agent controls
        """
        super().__init__(
            player_id,
            buy_threshold=0.3,
            build_threshold=0.2,
        )
        self.name = "Conservative"
