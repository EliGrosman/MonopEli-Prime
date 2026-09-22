"""Trade encoding, decoding, masking and rewards for the Monopoly RL environment.

This module provides functions for handling simple 1-for-1 property trades
in Phase 2.5a. The action space adds 756 trade actions (28 x 27 property pairs).

Trade Action Space Structure:
- Simple trades: 756 actions (indices 149-904)
- Accept trade: 1 action (index 905)
- Reject trade: 1 action (index 906)

Total new actions: 758 (bringing total from 149 to 907)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from monopoly_engine.board import Board, PropertySpace

if TYPE_CHECKING:
    from monopoly_engine.game import MonopolyGame
    from monopoly_engine.property import PropertyManager

# All 28 buyable property positions (must match action_space.py)
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

# Position to index mappings
POSITION_TO_BUYABLE_IDX: dict[int, int] = {pos: idx for idx, pos in enumerate(BUYABLE_POSITIONS)}
BUYABLE_IDX_TO_POSITION: dict[int, int] = {idx: pos for idx, pos in enumerate(BUYABLE_POSITIONS)}

# Trade action space dimensions
NUM_BUYABLE = len(BUYABLE_POSITIONS)  # 28
SIMPLE_TRADE_DIM = NUM_BUYABLE * (NUM_BUYABLE - 1)  # 28 * 27 = 756

# Action offsets (relative to start of trade actions at 149)
OFFSET_SIMPLE_TRADE = 149  # Trades start after gameplay actions
OFFSET_ACCEPT_TRADE = OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM  # 905
OFFSET_REJECT_TRADE = OFFSET_ACCEPT_TRADE + 1  # 906

# Total action space size with trades
TOTAL_ACTION_SPACE_SIZE = OFFSET_REJECT_TRADE + 1  # 907


def encode_simple_trade(my_property: int, their_property: int) -> int:
    """Encode a 1-for-1 trade as an action index.

    Args:
        my_property: Position of property I'm offering (must be in BUYABLE_POSITIONS)
        their_property: Position of property I want (must be in BUYABLE_POSITIONS)

    Returns:
        Action index in range [OFFSET_SIMPLE_TRADE, OFFSET_SIMPLE_TRADE + 756)

    Raises:
        ValueError: If either property is not buyable or they are the same
    """
    if my_property not in POSITION_TO_BUYABLE_IDX:
        raise ValueError(f"my_property {my_property} is not a buyable property")
    if their_property not in POSITION_TO_BUYABLE_IDX:
        raise ValueError(f"their_property {their_property} is not a buyable property")
    if my_property == their_property:
        raise ValueError("Cannot trade property with itself")

    my_idx = POSITION_TO_BUYABLE_IDX[my_property]
    their_idx = POSITION_TO_BUYABLE_IDX[their_property]

    # Adjust their_idx to skip my_idx (so we use 0-26 for the 27 other properties)
    if their_idx > my_idx:
        their_idx -= 1

    return OFFSET_SIMPLE_TRADE + (my_idx * (NUM_BUYABLE - 1) + their_idx)


def decode_simple_trade(action_idx: int) -> tuple[int, int]:
    """Decode an action index to (my_property, their_property) positions.

    Args:
        action_idx: Action index in range [OFFSET_SIMPLE_TRADE, OFFSET_SIMPLE_TRADE + 756)

    Returns:
        Tuple of (my_property_position, their_property_position)

    Raises:
        ValueError: If action_idx is not a valid simple trade action
    """
    if action_idx < OFFSET_SIMPLE_TRADE or action_idx >= OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM:
        raise ValueError(
            f"action_idx {action_idx} is not a simple trade action "
            f"(expected {OFFSET_SIMPLE_TRADE} <= idx < {OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM})"
        )

    offset = action_idx - OFFSET_SIMPLE_TRADE
    my_idx = offset // (NUM_BUYABLE - 1)
    their_idx = offset % (NUM_BUYABLE - 1)

    # Adjust their_idx back (we skipped my_idx when encoding)
    if their_idx >= my_idx:
        their_idx += 1

    my_property = BUYABLE_IDX_TO_POSITION[my_idx]
    their_property = BUYABLE_IDX_TO_POSITION[their_idx]

    return my_property, their_property


def get_simple_trade_mask(
    game: "MonopolyGame",
    player_id: int,
) -> NDArray[np.bool_]:
    """Generate mask for valid 1-for-1 trades.

    A trade is valid if:
    1. I own the property I'm offering
    2. Someone else owns the property I want
    3. Neither property has houses/hotels
    4. Neither property is mortgaged
    5. The opponent is not bankrupt

    Args:
        game: The current game state
        player_id: The player who would be proposing trades

    Returns:
        Boolean mask of shape (756,) where True indicates valid trade
    """
    mask = np.zeros(SIMPLE_TRADE_DIM, dtype=np.bool_)
    pm = game.property_manager

    # Get my tradeable properties (owned, no houses, not mortgaged)
    my_tradeable: list[int] = []
    for pos in pm.get_owned_by(player_id):
        prop = pm.get(pos)
        if prop is not None and not prop.mortgaged and prop.houses == 0:
            my_tradeable.append(pos)

    # Get opponent tradeable properties
    opponent_tradeable: list[int] = []
    for opp_id in range(len(game.players)):
        if opp_id == player_id:
            continue
        if game.players[opp_id].bankrupt:
            continue
        for pos in pm.get_owned_by(opp_id):
            prop = pm.get(pos)
            if prop is not None and not prop.mortgaged and prop.houses == 0:
                opponent_tradeable.append(pos)

    # Mark valid trades
    for my_prop in my_tradeable:
        for their_prop in opponent_tradeable:
            action_idx = encode_simple_trade(my_prop, their_prop)
            # Convert to mask index (0-755)
            mask_idx = action_idx - OFFSET_SIMPLE_TRADE
            mask[mask_idx] = True

    return mask


def get_trade_response_mask(game: "MonopolyGame", player_id: int) -> NDArray[np.bool_]:
    """Get mask for accept/reject trade actions.

    Returns a 2-element mask where:
    - [0] = can accept trade (index OFFSET_ACCEPT_TRADE)
    - [1] = can reject trade (index OFFSET_REJECT_TRADE)

    Both are True if there's a pending trade for this player.

    Args:
        game: The current game state
        player_id: The player who would be responding

    Returns:
        Boolean mask of shape (2,) for accept/reject actions
    """
    mask = np.zeros(2, dtype=np.bool_)

    # Check if there's a pending trade for this player
    for trade_id, trade in game.state.pending_trades.items():
        if trade["to_player"] == player_id:
            # Found a trade for us - we can accept or reject
            mask[0] = True  # Can accept
            mask[1] = True  # Can reject
            break

    return mask


def find_trade_for_player(game: "MonopolyGame", player_id: int) -> int | None:
    """Find the ID of a pending trade directed at this player.

    Args:
        game: The current game state
        player_id: The player to find trades for

    Returns:
        Trade ID if found, None otherwise
    """
    for trade_id, trade in game.state.pending_trades.items():
        if trade["to_player"] == player_id:
            return trade_id
    return None


@dataclass
class TradeRewardConfig:
    """Configuration for trade reward shaping."""

    # Reward for completing a monopoly via trade
    monopoly_completion_bonus: float = 2.0

    # Penalty for giving opponent a monopoly
    gave_monopoly_penalty: float = -1.5

    # Small penalty for rejected proposals (encourages better proposals)
    rejected_proposal_penalty: float = -0.01

    # Base value comparison weight (reward = value_delta * weight)
    value_comparison_weight: float = 0.5


def calculate_trade_rewards(
    game: "MonopolyGame",
    proposer_id: int,
    responder_id: int,
    proposer_gave: int,  # Property position proposer gave
    proposer_got: int,  # Property position proposer received
    accepted: bool,
    config: TradeRewardConfig | None = None,
) -> dict[int, float]:
    """Calculate rewards for both parties in a trade.

    Args:
        game: Game state AFTER the trade was executed (or rejected)
        proposer_id: Player who proposed the trade
        responder_id: Player who accepted/rejected
        proposer_gave: Property position the proposer gave away
        proposer_got: Property position the proposer received
        accepted: Whether the trade was accepted
        config: Reward configuration (uses defaults if None)

    Returns:
        Dict mapping player_id to reward
    """
    if config is None:
        config = TradeRewardConfig()

    rewards: dict[int, float] = {proposer_id: 0.0, responder_id: 0.0}

    if not accepted:
        # Small penalty for proposer on rejection
        rewards[proposer_id] = config.rejected_proposal_penalty
        return rewards

    pm = game.property_manager

    # Check monopoly completion for proposer
    if _completed_monopoly_with_property(pm, proposer_id, proposer_got):
        rewards[proposer_id] += config.monopoly_completion_bonus

    # Check monopoly completion for responder
    if _completed_monopoly_with_property(pm, responder_id, proposer_gave):
        rewards[responder_id] += config.monopoly_completion_bonus

    # Penalty if you gave opponent a monopoly
    if _completed_monopoly_with_property(pm, responder_id, proposer_gave):
        rewards[proposer_id] += config.gave_monopoly_penalty
    if _completed_monopoly_with_property(pm, proposer_id, proposer_got):
        rewards[responder_id] += config.gave_monopoly_penalty

    return rewards


def _completed_monopoly_with_property(
    pm: "PropertyManager",
    player_id: int,
    prop_position: int,
) -> bool:
    """Check if player now has a complete monopoly containing this property.

    Args:
        pm: PropertyManager
        player_id: Player to check
        prop_position: Position of the property that was just acquired

    Returns:
        True if player now owns all properties in this color group
    """
    prop = pm.get(prop_position)
    if prop is None or prop.owner != player_id:
        return False

    # Get the color of this property
    space = Board.get_space(prop_position)
    if not isinstance(space, PropertySpace):
        return False  # Railroads and utilities don't form monopolies for building

    color = prop.color
    if color is None:
        return False

    return pm.has_monopoly(player_id, color)


def get_strategic_property_value(
    pm: "PropertyManager",
    prop_position: int,
    player_id: int,
) -> float:
    """Calculate the strategic value of a property for a player.

    Factors:
    - Base price
    - Proximity to monopoly (how many others in color do I own?)
    - Landing frequency (empirical data)

    Args:
        pm: PropertyManager
        prop_position: Position of the property
        player_id: Player whose perspective to evaluate from

    Returns:
        Strategic value in dollars
    """
    space = Board.get_space(prop_position)

    # Base value is the purchase price
    if hasattr(space, "cost"):
        base_value = float(space.cost)
    else:
        base_value = 100.0  # Default for non-standard spaces

    # Landing frequency multiplier (empirical from Monopoly probability analysis)
    # Properties 6-9 spaces from Jail are landed on more often
    landing_freq = LANDING_FREQUENCY.get(prop_position, 1.0)

    # Monopoly proximity multiplier
    prop = pm.get(prop_position)
    if prop is not None and isinstance(space, PropertySpace):
        color = prop.color
        if color is not None:
            from monopoly_engine.types import PROPERTY_GROUPS

            color_props = PROPERTY_GROUPS.get(color, ())
            owned_count = 0
            for p in color_props:
                prop_state = pm.get(p)
                if prop_state is not None and prop_state.owner == player_id:
                    owned_count += 1
            owned_in_color = owned_count
            total_in_color = len(color_props)
            if total_in_color > 0:
                # Value increases quadratically as we approach monopoly
                proximity_mult = 1.0 + (owned_in_color / total_in_color) ** 2 * 2.0
                base_value *= proximity_mult

    return base_value * landing_freq


# Empirical landing frequencies from Monopoly probability analysis
# Based on distance from Jail (position 10) and Go To Jail (position 30)
LANDING_FREQUENCY: dict[int, float] = {
    # Brown - low frequency
    1: 0.95,  # Mediterranean
    3: 0.95,  # Baltic
    # Light Blue - moderate (6-9 from Jail)
    6: 1.1,  # Oriental
    8: 1.1,  # Vermont
    9: 1.1,  # Connecticut
    # Magenta - moderate
    11: 1.05,  # St. Charles
    13: 1.05,  # States
    14: 1.05,  # Virginia
    # Orange - highest frequency! (6-9 from Jail)
    16: 1.2,  # St. James
    18: 1.2,  # Tennessee
    19: 1.2,  # New York
    # Red - high frequency
    21: 1.15,  # Kentucky
    23: 1.15,  # Indiana
    24: 1.15,  # Illinois
    # Yellow - moderate
    26: 1.05,  # Atlantic
    27: 1.05,  # Ventnor
    29: 1.05,  # Marvin Gardens
    # Green - lower (far from Jail)
    31: 1.0,  # Pacific
    32: 1.0,  # North Carolina
    34: 1.0,  # Pennsylvania
    # Dark Blue - low frequency but high rent
    37: 0.95,  # Park Place
    39: 0.95,  # Boardwalk
    # Railroads - moderate
    5: 1.1,  # Reading
    15: 1.1,  # Pennsylvania RR
    25: 1.1,  # B&O
    35: 1.05,  # Short Line
    # Utilities - moderate
    12: 1.0,  # Electric Company
    28: 1.0,  # Water Works
}
