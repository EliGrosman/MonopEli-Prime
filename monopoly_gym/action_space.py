"""Action space encoding for the Monopoly RL environment.

This module provides the ActionEncoder class that handles conversion between
the game engine's Action objects and integer action indices used by RL agents.

Action-v2 (158 decisions; legacy 907-action trade policies are incompatible):

Foundation gameplay actions:
- Buy Property: 1 (index 0) - buy the property you landed on
- Pass Buy: 1 (index 1) - decline purchase; no auction in foundation-v1
- Build House: 22 (indices 2-23) - on developable properties
- Build Hotel: 22 (indices 24-45) - on developable properties
- Sell House: 22 (indices 46-67) - from developable properties
- Sell Hotel: 22 (indices 68-89) - from developable properties
- Mortgage: 28 (indices 90-117) - any purchasable property
- Unmortgage: 28 (indices 118-145) - any purchasable property
- End Turn: 1 (index 146)
- Use Jail Card: 1 (index 147)
- Pay Jail Fine: 1 (index 148)

- Roll Dice: 1 (index 149)
- Sell Building Group: 8 (indices 150-157, ascending color-group board order)

Trading is disabled. Legacy trade helper code is not a supported encoding.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from monopoly_engine import (
    AcceptTrade,
    Action,
    Board,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    EndTurn,
    MortgageProperty,
    PayJailFine,
    ProposeTrade,
    RejectTrade,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)
from monopoly_engine.actions import PassBuy, RollDice, SellBuildingGroup
from monopoly_engine.foundation import GROUP_STARTS

from .trades import (
    OFFSET_ACCEPT_TRADE,
    OFFSET_REJECT_TRADE,
    OFFSET_SIMPLE_TRADE,
    SIMPLE_TRADE_DIM,
    decode_simple_trade,
    encode_simple_trade,
    find_trade_for_player,
    get_simple_trade_mask,
    get_trade_response_mask,
)

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame

# All 28 buyable property positions (colored properties, railroads, utilities)
BUYABLE_POSITIONS: tuple[int, ...] = (
    1,
    3,
    5,
    6,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    19,
    21,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    31,
    32,
    34,
    35,
    37,
    39,
)

# 22 developable positions (colored properties only - no railroads/utilities)
DEVELOPABLE_POSITIONS: tuple[int, ...] = (
    1,
    3,
    6,
    8,
    9,
    11,
    13,
    14,
    16,
    18,
    19,
    21,
    23,
    24,
    26,
    27,
    29,
    31,
    32,
    34,
    37,
    39,
)

# Position to index mappings for fast lookup
_BUYABLE_TO_INDEX: dict[int, int] = {pos: idx for idx, pos in enumerate(BUYABLE_POSITIONS)}
_DEVELOPABLE_TO_INDEX: dict[int, int] = {pos: idx for idx, pos in enumerate(DEVELOPABLE_POSITIONS)}

# Action space offsets - Gameplay (0-157)
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

# Gameplay-only action space size (for backwards compatibility)
OFFSET_ROLL_DICE = 149
OFFSET_SELL_GROUP = 150
ACTION_VERSION = "action-v2"
GAMEPLAY_ACTION_SPACE_SIZE = 158

# Total action space size (with Phase 2.5a trades)
ACTION_SPACE_SIZE = GAMEPLAY_ACTION_SPACE_SIZE

# Explicit exports for mypy strict mode
__all__ = [
    # Constants
    "ACTION_SPACE_SIZE",
    "GAMEPLAY_ACTION_SPACE_SIZE",
    "BUYABLE_POSITIONS",
    "DEVELOPABLE_POSITIONS",
    # Gameplay offsets
    "OFFSET_BUY_PROPERTY",
    "OFFSET_PASS_BUY",
    "OFFSET_BUILD_HOUSE",
    "OFFSET_BUILD_HOTEL",
    "OFFSET_SELL_HOUSE",
    "OFFSET_SELL_HOTEL",
    "OFFSET_MORTGAGE",
    "OFFSET_UNMORTGAGE",
    "OFFSET_END_TURN",
    "OFFSET_USE_JAIL_CARD",
    "OFFSET_PAY_JAIL_FINE",
    # Trade offsets (re-exported from trades.py)
    "OFFSET_SIMPLE_TRADE",
    "OFFSET_ACCEPT_TRADE",
    "OFFSET_REJECT_TRADE",
    "SIMPLE_TRADE_DIM",
    # Trade utilities (re-exported from trades.py)
    "decode_simple_trade",
    "encode_simple_trade",
    "get_simple_trade_mask",
    "get_trade_response_mask",
    "find_trade_for_player",
    # Encoder class
    "ActionEncoder",
]


class ActionEncoder:
    """Encodes and decodes actions between Action objects and integer indices.

    This class provides bidirectional conversion between the game engine's
    Action objects and the flat integer action space used by RL agents.
    It also provides action masking to identify valid actions in any game state.

    Attributes:
        action_space_size: The total number of possible actions.
        enable_trades: Whether trade actions are enabled (Phase 2.5a).
    """

    def __init__(self, enable_trades: bool = False) -> None:
        """Initialize the ActionEncoder.

        Args:
            enable_trades: If True, include trade actions (907 total).
                          If False, foundation actions (158 total).
        """
        if enable_trades:
            raise ValueError(
                "Trading is disabled in foundation-v1; "
                "legacy 907-action checkpoints are incompatible"
            )
        self.enable_trades = enable_trades
        if enable_trades:
            self.action_space_size = ACTION_SPACE_SIZE  # 907
        else:
            self.action_space_size = GAMEPLAY_ACTION_SPACE_SIZE  # 158

        # Mask cache: {player_id: (fingerprint_tuple, cached_mask)}
        self._mask_cache: dict[int, tuple[tuple[object, ...], NDArray[np.bool_]]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def _compute_state_fingerprint(
        self, game: "MonopolyGame", player_id: int
    ) -> tuple[object, ...]:
        """Compute a lightweight fingerprint of state relevant to action masking.

        Returns a tuple that changes if and only if the mask would differ.
        """
        player = game.players[player_id]
        # Property state: (owner, houses, mortgaged) for all buyable positions
        pm = game.property_manager
        prop_state = tuple(
            (p.owner, p.houses, p.mortgaged)
            for pos in BUYABLE_POSITIONS
            if (p := pm.properties.get(pos)) is not None
        )
        return (
            player_id,
            player.bankrupt,
            player.money,
            player.position,
            player.in_jail,
            player.jail_cards,
            game.current_player,
            game.houses_remaining,
            game.hotels_remaining,
            prop_state,
        )

    def invalidate_cache(self) -> None:
        """Clear the mask cache (call on env reset)."""
        self._mask_cache.clear()

    @property
    def cache_stats(self) -> tuple[int, int]:
        """Return (cache_hits, cache_misses)."""
        return (self._cache_hits, self._cache_misses)

    def encode(self, action: Action) -> int:
        """Convert an Action object to its corresponding action index.

        Args:
            action: The Action object to encode.

        Returns:
            The integer action index (0-157).

        Raises:
            ValueError: If the action type is not supported or property is invalid.
        """
        if isinstance(action, RollDice):
            return OFFSET_ROLL_DICE
        if isinstance(action, PassBuy):
            return OFFSET_PASS_BUY
        if isinstance(action, SellBuildingGroup):
            return OFFSET_SELL_GROUP + GROUP_STARTS.index(action.property_id)
        if isinstance(action, BuyProperty):
            return OFFSET_BUY_PROPERTY

        if isinstance(action, EndTurn):
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

        # Trade actions (Phase 2.5a)
        if self.enable_trades:
            if isinstance(action, ProposeTrade):
                # For simple 1-for-1 trades, encode using the properties
                if (
                    len(action.give_properties) == 1
                    and len(action.want_properties) == 1
                    and action.give_money == 0
                    and action.want_money == 0
                ):
                    return encode_simple_trade(
                        action.give_properties[0],
                        action.want_properties[0],
                    )
                # Complex trades not supported in Phase 2.5a
                raise ValueError("Only 1-for-1 property trades supported in Phase 2.5a")

            if isinstance(action, AcceptTrade):
                return OFFSET_ACCEPT_TRADE

            if isinstance(action, RejectTrade):
                return OFFSET_REJECT_TRADE

        raise ValueError(f"Unsupported action type: {type(action).__name__}")

    def decode(self, action_idx: int, player_id: int, game: "MonopolyGame") -> Action:
        """Convert an action index to the corresponding Action object.

        Args:
            action_idx: The action index (0-157).
            player_id: The ID of the player taking the action.
            game: The current game state (used to determine property positions).

        Returns:
            The corresponding Action object.

        Raises:
            ValueError: If the action index is out of range or invalid.
        """
        if action_idx < 0 or action_idx >= self.action_space_size:
            raise ValueError(
                f"Action index {action_idx} out of range [0, {self.action_space_size})"
            )

        if action_idx == OFFSET_ROLL_DICE:
            return RollDice(player_id)
        if OFFSET_SELL_GROUP <= action_idx < GAMEPLAY_ACTION_SPACE_SIZE:
            return SellBuildingGroup(player_id, GROUP_STARTS[action_idx - OFFSET_SELL_GROUP])

        # Buy Property (index 0)
        if action_idx == OFFSET_BUY_PROPERTY:
            player = game.players[player_id]
            return BuyProperty(player_id=player_id, property_id=player.position)

        # Pass Buy (index 1) - independent purchase refusal
        if action_idx == OFFSET_PASS_BUY:
            return PassBuy(player_id=player_id)

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

        # Trade actions (Phase 2.5a)
        if self.enable_trades:
            # Simple 1-for-1 trades (indices 149-904)
            if OFFSET_SIMPLE_TRADE <= action_idx < OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM:
                my_prop, their_prop = decode_simple_trade(action_idx)
                # Find the owner of their_prop
                prop = game.property_manager.get(their_prop)
                if prop is None or prop.owner is None:
                    raise ValueError(f"Property {their_prop} has no owner for trade")
                to_player = prop.owner
                return ProposeTrade(
                    player_id=player_id,
                    to_player=to_player,
                    give_properties=[my_prop],
                    want_properties=[their_prop],
                    give_money=0,
                    want_money=0,
                )

            # Accept Trade (index 905)
            if action_idx == OFFSET_ACCEPT_TRADE:
                # Find the pending trade for this player
                trade_id = find_trade_for_player(game, player_id)
                if trade_id is None:
                    raise ValueError("No pending trade to accept")
                return AcceptTrade(player_id=player_id, trade_id=trade_id)

            # Reject Trade (index 906)
            if action_idx == OFFSET_REJECT_TRADE:
                # Find the pending trade for this player
                trade_id = find_trade_for_player(game, player_id)
                if trade_id is None:
                    raise ValueError("No pending trade to reject")
                return RejectTrade(player_id=player_id, trade_id=trade_id)

        raise ValueError(f"Invalid action index: {action_idx}")

    def get_action_mask(
        self, game: "MonopolyGame", player_id: int, pending_trade_response: bool = False
    ) -> NDArray[np.bool_]:
        mask = np.zeros(self.action_space_size, dtype=np.bool_)
        if game.game_over or player_id != game.decision_player or game.players[player_id].bankrupt:
            return mask
        from monopoly_engine.foundation import legal_actions
        for action in legal_actions(game, player_id):
            mask[self.encode(action)] = True
        return mask

    def get_action_name(self, action_idx: int) -> str:
        """Get a human-readable name for an action index.

        Args:
            action_idx: The action index (0-157).

        Returns:
            A string describing the action.
        """
        if action_idx == OFFSET_ROLL_DICE:
            return "Roll Dice"
        if OFFSET_SELL_GROUP <= action_idx < GAMEPLAY_ACTION_SPACE_SIZE:
            return f"Sell Building Group {GROUP_STARTS[action_idx - OFFSET_SELL_GROUP]}"
        if action_idx == OFFSET_BUY_PROPERTY:
            return "Buy Property"

        if action_idx == OFFSET_PASS_BUY:
            return "Pass Buy"

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

        # Trade actions (Phase 2.5a)
        if self.enable_trades:
            if OFFSET_SIMPLE_TRADE <= action_idx < OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM:
                my_prop, their_prop = decode_simple_trade(action_idx)
                my_space = Board.get_space(my_prop)
                their_space = Board.get_space(their_prop)
                return f"Trade {my_space.name} for {their_space.name}"

            if action_idx == OFFSET_ACCEPT_TRADE:
                return "Accept Trade"

            if action_idx == OFFSET_REJECT_TRADE:
                return "Reject Trade"

        return f"Unknown Action ({action_idx})"
