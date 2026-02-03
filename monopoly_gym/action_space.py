"""Action space encoding for the Monopoly RL environment.

This module provides the ActionEncoder class that handles conversion between
the game engine's Action objects and integer action indices used by RL agents.

Action Space Structure (149 total dimensions):
- Buy Property: 1 (index 0) - buy the property you landed on
- Pass Buy: 1 (index 1) - decline to buy, triggers auction
- Build House: 22 (indices 2-23) - on developable properties
- Build Hotel: 22 (indices 24-45) - on developable properties
- Sell House: 22 (indices 46-67) - from developable properties
- Sell Hotel: 22 (indices 68-89) - from developable properties
- Mortgage: 28 (indices 90-117) - any purchasable property
- Unmortgage: 28 (indices 118-145) - any purchasable property
- End Turn: 1 (index 146)
- Use Jail Card: 1 (index 147)
- Pay Jail Fine: 1 (index 148)

Note: Trades are not included in Phase 2 action space.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from monopoly_engine import (
    Action,
    Board,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    EndTurn,
    MortgageProperty,
    PayJailFine,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
    can_build_house,
    can_mortgage_property,
    can_sell_house,
    can_unmortgage_property,
    get_property_cost,
)

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame

# All 28 buyable property positions (colored properties, railroads, utilities)
BUYABLE_POSITIONS: tuple[int, ...] = (
    1, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19,
    21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 37, 39
)

# 22 developable positions (colored properties only - no railroads/utilities)
DEVELOPABLE_POSITIONS: tuple[int, ...] = (
    1, 3, 6, 8, 9, 11, 13, 14, 16, 18, 19,
    21, 23, 24, 26, 27, 29, 31, 32, 34, 37, 39
)

# Position to index mappings for fast lookup
_BUYABLE_TO_INDEX: dict[int, int] = {pos: idx for idx, pos in enumerate(BUYABLE_POSITIONS)}
_DEVELOPABLE_TO_INDEX: dict[int, int] = {pos: idx for idx, pos in enumerate(DEVELOPABLE_POSITIONS)}

# Action space offsets
OFFSET_BUY_PROPERTY = 0
OFFSET_PASS_BUY = 1
OFFSET_BUILD_HOUSE = 2
OFFSET_BUILD_HOTEL = 24
OFFSET_SELL_HOUSE = 46
OFFSET_SELL_HOTEL = 68
OFFSET_MORTGAGE = 90
OFFSET_UNMORTGAGE = 118
OFFSET_END_TURN = 146
OFFSET_USE_JAIL_CARD = 147
OFFSET_PAY_JAIL_FINE = 148

# Total action space size
ACTION_SPACE_SIZE = 149


class ActionEncoder:
    """Encodes and decodes actions between Action objects and integer indices.

    This class provides bidirectional conversion between the game engine's
    Action objects and the flat integer action space used by RL agents.
    It also provides action masking to identify valid actions in any game state.

    Attributes:
        action_space_size: The total number of possible actions (149).
    """

    def __init__(self) -> None:
        """Initialize the ActionEncoder."""
        self.action_space_size = ACTION_SPACE_SIZE

    def encode(self, action: Action) -> int:
        """Convert an Action object to its corresponding action index.

        Args:
            action: The Action object to encode.

        Returns:
            The integer action index (0-148).

        Raises:
            ValueError: If the action type is not supported or property is invalid.
        """
        if isinstance(action, BuyProperty):
            return OFFSET_BUY_PROPERTY

        if isinstance(action, EndTurn):
            # EndTurn at offset 1 represents "Pass Buy" when landing on property
            # EndTurn at offset 146 represents regular end turn
            # We use OFFSET_END_TURN for the general case
            return OFFSET_END_TURN

        if isinstance(action, BuildHouse):
            if action.property_id not in _DEVELOPABLE_TO_INDEX:
                raise ValueError(f"Cannot build on property {action.property_id}")
            return OFFSET_BUILD_HOUSE + _DEVELOPABLE_TO_INDEX[action.property_id]

        if isinstance(action, BuildHotel):
            if action.property_id not in _DEVELOPABLE_TO_INDEX:
                raise ValueError(f"Cannot build hotel on property {action.property_id}")
            return OFFSET_BUILD_HOTEL + _DEVELOPABLE_TO_INDEX[action.property_id]

        if isinstance(action, SellHouse):
            if action.property_id not in _DEVELOPABLE_TO_INDEX:
                raise ValueError(f"Cannot sell house from property {action.property_id}")
            return OFFSET_SELL_HOUSE + _DEVELOPABLE_TO_INDEX[action.property_id]

        if isinstance(action, SellHotel):
            if action.property_id not in _DEVELOPABLE_TO_INDEX:
                raise ValueError(f"Cannot sell hotel from property {action.property_id}")
            return OFFSET_SELL_HOTEL + _DEVELOPABLE_TO_INDEX[action.property_id]

        if isinstance(action, MortgageProperty):
            if action.property_id not in _BUYABLE_TO_INDEX:
                raise ValueError(f"Cannot mortgage property {action.property_id}")
            return OFFSET_MORTGAGE + _BUYABLE_TO_INDEX[action.property_id]

        if isinstance(action, UnmortgageProperty):
            if action.property_id not in _BUYABLE_TO_INDEX:
                raise ValueError(f"Cannot unmortgage property {action.property_id}")
            return OFFSET_UNMORTGAGE + _BUYABLE_TO_INDEX[action.property_id]

        if isinstance(action, UseJailCard):
            return OFFSET_USE_JAIL_CARD

        if isinstance(action, PayJailFine):
            return OFFSET_PAY_JAIL_FINE

        raise ValueError(f"Unsupported action type: {type(action).__name__}")

    def decode(self, action_idx: int, player_id: int, game: "MonopolyGame") -> Action:
        """Convert an action index to the corresponding Action object.

        Args:
            action_idx: The action index (0-148).
            player_id: The ID of the player taking the action.
            game: The current game state (used to determine property positions).

        Returns:
            The corresponding Action object.

        Raises:
            ValueError: If the action index is out of range or invalid.
        """
        if action_idx < 0 or action_idx >= ACTION_SPACE_SIZE:
            raise ValueError(f"Action index {action_idx} out of range [0, {ACTION_SPACE_SIZE})")

        # Buy Property (index 0)
        if action_idx == OFFSET_BUY_PROPERTY:
            player = game.players[player_id]
            return BuyProperty(player_id=player_id, property_id=player.position)

        # Pass Buy (index 1) - represented as EndTurn
        if action_idx == OFFSET_PASS_BUY:
            return EndTurn(player_id=player_id)

        # Build House (indices 2-23)
        if OFFSET_BUILD_HOUSE <= action_idx < OFFSET_BUILD_HOTEL:
            prop_idx = action_idx - OFFSET_BUILD_HOUSE
            property_id = DEVELOPABLE_POSITIONS[prop_idx]
            return BuildHouse(player_id=player_id, property_id=property_id)

        # Build Hotel (indices 24-45)
        if OFFSET_BUILD_HOTEL <= action_idx < OFFSET_SELL_HOUSE:
            prop_idx = action_idx - OFFSET_BUILD_HOTEL
            property_id = DEVELOPABLE_POSITIONS[prop_idx]
            return BuildHotel(player_id=player_id, property_id=property_id)

        # Sell House (indices 46-67)
        if OFFSET_SELL_HOUSE <= action_idx < OFFSET_SELL_HOTEL:
            prop_idx = action_idx - OFFSET_SELL_HOUSE
            property_id = DEVELOPABLE_POSITIONS[prop_idx]
            return SellHouse(player_id=player_id, property_id=property_id)

        # Sell Hotel (indices 68-89)
        if OFFSET_SELL_HOTEL <= action_idx < OFFSET_MORTGAGE:
            prop_idx = action_idx - OFFSET_SELL_HOTEL
            property_id = DEVELOPABLE_POSITIONS[prop_idx]
            return SellHotel(player_id=player_id, property_id=property_id)

        # Mortgage (indices 90-117)
        if OFFSET_MORTGAGE <= action_idx < OFFSET_UNMORTGAGE:
            prop_idx = action_idx - OFFSET_MORTGAGE
            property_id = BUYABLE_POSITIONS[prop_idx]
            return MortgageProperty(player_id=player_id, property_id=property_id)

        # Unmortgage (indices 118-145)
        if OFFSET_UNMORTGAGE <= action_idx < OFFSET_END_TURN:
            prop_idx = action_idx - OFFSET_UNMORTGAGE
            property_id = BUYABLE_POSITIONS[prop_idx]
            return UnmortgageProperty(player_id=player_id, property_id=property_id)

        # End Turn (index 146)
        if action_idx == OFFSET_END_TURN:
            return EndTurn(player_id=player_id)

        # Use Jail Card (index 147)
        if action_idx == OFFSET_USE_JAIL_CARD:
            return UseJailCard(player_id=player_id)

        # Pay Jail Fine (index 148)
        if action_idx == OFFSET_PAY_JAIL_FINE:
            return PayJailFine(player_id=player_id)

        raise ValueError(f"Invalid action index: {action_idx}")

    def get_action_mask(self, game: "MonopolyGame", player_id: int) -> NDArray[np.bool_]:
        """Generate a boolean mask indicating which actions are valid.

        This method performs all validation upfront. If an action is marked
        as valid in the mask, it is guaranteed to execute successfully.

        Args:
            game: The current game state.
            player_id: The ID of the player for whom to generate the mask.

        Returns:
            A numpy boolean array of shape (149,) where True indicates a valid action.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)

        player = game.players[player_id]

        # Cannot take actions if bankrupt
        if player.bankrupt:
            return mask

        # Cannot take actions if not current player (except some sell/mortgage during debt)
        is_current_player = game.current_player == player_id

        # Check Buy Property (index 0) and Pass Buy (index 1)
        if is_current_player:
            position = player.position

            if Board.is_buyable(position):
                prop = game.property_manager.get(position)
                if prop is not None and not prop.is_owned:
                    # Check if player can afford it
                    cost = get_property_cost(position)
                    if player.can_afford(cost):
                        mask[OFFSET_BUY_PROPERTY] = True
                    # Pass buy is always available when on unowned property
                    mask[OFFSET_PASS_BUY] = True

        # Check Build House (indices 2-23)
        for idx, prop_pos in enumerate(DEVELOPABLE_POSITIONS):
            if is_current_player:
                can_build, _ = can_build_house(
                    player,
                    game.property_manager,
                    prop_pos,
                    game.houses_remaining,
                    game.hotels_remaining,
                )
                if can_build:
                    prop = game.property_manager.get(prop_pos)
                    if prop is not None and prop.houses < 4:
                        # Building a house (not a hotel)
                        mask[OFFSET_BUILD_HOUSE + idx] = True

        # Check Build Hotel (indices 24-45)
        for idx, prop_pos in enumerate(DEVELOPABLE_POSITIONS):
            if is_current_player:
                prop = game.property_manager.get(prop_pos)
                if prop is not None and prop.houses == 4:
                    can_build, _ = can_build_house(
                        player,
                        game.property_manager,
                        prop_pos,
                        game.houses_remaining,
                        game.hotels_remaining,
                    )
                    if can_build:
                        mask[OFFSET_BUILD_HOTEL + idx] = True

        # Check Sell House (indices 46-67) - can be done anytime by owner
        for idx, prop_pos in enumerate(DEVELOPABLE_POSITIONS):
            prop = game.property_manager.get(prop_pos)
            if prop is not None and prop.owner == player_id:
                can_sell, _ = can_sell_house(player, game.property_manager, prop_pos)
                if can_sell:
                    if prop.houses > 0 and prop.houses < 5:
                        # Selling a house (not a hotel)
                        mask[OFFSET_SELL_HOUSE + idx] = True

        # Check Sell Hotel (indices 68-89) - can be done anytime by owner
        for idx, prop_pos in enumerate(DEVELOPABLE_POSITIONS):
            prop = game.property_manager.get(prop_pos)
            if prop is not None and prop.owner == player_id and prop.houses == 5:
                can_sell, _ = can_sell_house(player, game.property_manager, prop_pos)
                if can_sell:
                    mask[OFFSET_SELL_HOTEL + idx] = True

        # Check Mortgage (indices 90-117) - can be done anytime by owner
        for idx, prop_pos in enumerate(BUYABLE_POSITIONS):
            prop = game.property_manager.get(prop_pos)
            if prop is not None and prop.owner == player_id:
                can_mortgage, _ = can_mortgage_property(
                    player, game.property_manager, prop_pos
                )
                if can_mortgage:
                    mask[OFFSET_MORTGAGE + idx] = True

        # Check Unmortgage (indices 118-145)
        if is_current_player:
            for idx, prop_pos in enumerate(BUYABLE_POSITIONS):
                prop = game.property_manager.get(prop_pos)
                if prop is not None and prop.owner == player_id and prop.mortgaged:
                    can_unmortgage, _ = can_unmortgage_property(
                        player, game.property_manager, prop_pos
                    )
                    if can_unmortgage:
                        mask[OFFSET_UNMORTGAGE + idx] = True

        # Check End Turn (index 146)
        if is_current_player:
            # End turn is generally always available to current player
            # (unless they have mandatory actions like paying rent)
            mask[OFFSET_END_TURN] = True

        # Check Use Jail Card (index 147)
        if is_current_player and player.in_jail and player.jail_cards > 0:
            mask[OFFSET_USE_JAIL_CARD] = True

        # Check Pay Jail Fine (index 148)
        if is_current_player and player.in_jail:
            from monopoly_engine.types import JAIL_FINE
            if player.can_afford(JAIL_FINE):
                mask[OFFSET_PAY_JAIL_FINE] = True

        return mask

    def get_action_name(self, action_idx: int) -> str:
        """Get a human-readable name for an action index.

        Args:
            action_idx: The action index (0-148).

        Returns:
            A string describing the action.
        """
        if action_idx == OFFSET_BUY_PROPERTY:
            return "Buy Property"

        if action_idx == OFFSET_PASS_BUY:
            return "Pass Buy (Auction)"

        if OFFSET_BUILD_HOUSE <= action_idx < OFFSET_BUILD_HOTEL:
            prop_idx = action_idx - OFFSET_BUILD_HOUSE
            prop_pos = DEVELOPABLE_POSITIONS[prop_idx]
            space = Board.get_space(prop_pos)
            return f"Build House on {space.name}"

        if OFFSET_BUILD_HOTEL <= action_idx < OFFSET_SELL_HOUSE:
            prop_idx = action_idx - OFFSET_BUILD_HOTEL
            prop_pos = DEVELOPABLE_POSITIONS[prop_idx]
            space = Board.get_space(prop_pos)
            return f"Build Hotel on {space.name}"

        if OFFSET_SELL_HOUSE <= action_idx < OFFSET_SELL_HOTEL:
            prop_idx = action_idx - OFFSET_SELL_HOUSE
            prop_pos = DEVELOPABLE_POSITIONS[prop_idx]
            space = Board.get_space(prop_pos)
            return f"Sell House from {space.name}"

        if OFFSET_SELL_HOTEL <= action_idx < OFFSET_MORTGAGE:
            prop_idx = action_idx - OFFSET_SELL_HOTEL
            prop_pos = DEVELOPABLE_POSITIONS[prop_idx]
            space = Board.get_space(prop_pos)
            return f"Sell Hotel from {space.name}"

        if OFFSET_MORTGAGE <= action_idx < OFFSET_UNMORTGAGE:
            prop_idx = action_idx - OFFSET_MORTGAGE
            prop_pos = BUYABLE_POSITIONS[prop_idx]
            space = Board.get_space(prop_pos)
            return f"Mortgage {space.name}"

        if OFFSET_UNMORTGAGE <= action_idx < OFFSET_END_TURN:
            prop_idx = action_idx - OFFSET_UNMORTGAGE
            prop_pos = BUYABLE_POSITIONS[prop_idx]
            space = Board.get_space(prop_pos)
            return f"Unmortgage {space.name}"

        if action_idx == OFFSET_END_TURN:
            return "End Turn"

        if action_idx == OFFSET_USE_JAIL_CARD:
            return "Use Get Out of Jail Free Card"

        if action_idx == OFFSET_PAY_JAIL_FINE:
            return "Pay $50 Jail Fine"

        return f"Unknown Action ({action_idx})"
