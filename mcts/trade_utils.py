"""Trade analysis utilities for evaluating and suggesting trades.

These helper functions provide heuristic trade evaluation without using MCTS
or LLMs. They are used to:
1. Provide context to the LLM in trade prompts (monopoly proximity, impact).
2. Generate heuristic trade candidates for the LLM to refine.
3. Provide quick evaluations when LLM budget is exhausted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from monopoly_engine.board import Board, PropertySpace, RailroadSpace, UtilitySpace
from monopoly_engine.game import MonopolyGame
from monopoly_engine.property import PropertyManager
from monopoly_engine.rules import calculate_net_worth
from monopoly_engine.types import PROPERTY_GROUPS, PropertyColor, TradeOfferData


def monopoly_proximity(
    player_id: int,
    pm: PropertyManager,
) -> dict[PropertyColor, float]:
    """Calculate how close a player is to completing each color group.

    Returns:
        Dict mapping each PropertyColor to a float in [0.0, 1.0].
        0.0 = owns no properties in group.
        1.0 = has monopoly (owns all properties in group).
        Values in between = fraction owned (e.g., 0.67 for 2/3).
    """
    result: dict[PropertyColor, float] = {}
    for color, positions in PROPERTY_GROUPS.items():
        total = len(positions)
        owned = sum(
            1
            for pos in positions
            if pm.properties.get(pos) is not None and pm.properties[pos].owner == player_id
        )
        result[color] = owned / total if total > 0 else 0.0
    return result


def _count_monopolies(player_id: int, pm: PropertyManager) -> int:
    """Count how many monopolies a player has (excluding railroads/utilities)."""
    return sum(
        1
        for color in PropertyColor
        if color not in (PropertyColor.RAILROAD, PropertyColor.UTILITY)
        and pm.has_monopoly(player_id, color)
    )


def _player_metrics(
    game: MonopolyGame,
    player_id: int,
) -> dict[str, Any]:
    """Compute strategic metrics for a player."""
    player = game.players[player_id]
    pm = game.property_manager
    return {
        "net_worth": calculate_net_worth(player, pm),
        "monopoly_count": _count_monopolies(player_id, pm),
        "proximity": {color.name: val for color, val in monopoly_proximity(player_id, pm).items()},
    }


def _apply_trade_to_pm(
    pm: PropertyManager,
    trade: TradeOfferData,
) -> None:
    """Apply a trade's property transfers to a PropertyManager (in-place)."""
    from_id = trade["from_player"]
    to_id = trade["to_player"]

    for pos in trade["give_properties"]:
        prop = pm.properties.get(pos)
        if prop is not None:
            prop.owner = to_id

    for pos in trade["want_properties"]:
        prop = pm.properties.get(pos)
        if prop is not None:
            prop.owner = from_id


def trade_impact(
    game: MonopolyGame,
    trade: TradeOfferData,
) -> dict[str, Any]:
    """Evaluate how a trade changes each player's strategic position.

    Simulates the trade on a deep-copied PropertyManager to compute
    before/after metrics without mutating game state.

    Returns dict with:
        - from_player_id, to_player_id
        - before / after: per-player metrics (net_worth, monopoly_count, proximity)
        - from_player_net_change, to_player_net_change
        - monopolies_created, monopolies_broken
    """
    from_id = trade["from_player"]
    to_id = trade["to_player"]

    # --- Before metrics ---
    before_from = _player_metrics(game, from_id)
    before_to = _player_metrics(game, to_id)

    # --- Simulate trade on a copy ---
    pm_copy = copy.deepcopy(game.property_manager)
    _apply_trade_to_pm(pm_copy, trade)

    # Compute post-trade net worth accounting for cash transfers
    give_money = trade["give_money"]
    want_money = trade["want_money"]

    from_player = game.players[from_id]
    to_player = game.players[to_id]

    # Simulate cash changes on copies
    from_player_copy = copy.copy(from_player)
    to_player_copy = copy.copy(to_player)
    from_player_copy.money = from_player.money - give_money + want_money
    to_player_copy.money = to_player.money + give_money - want_money

    after_from_nw = calculate_net_worth(from_player_copy, pm_copy)
    after_to_nw = calculate_net_worth(to_player_copy, pm_copy)

    after_from_monopolies = _count_monopolies(from_id, pm_copy)
    after_to_monopolies = _count_monopolies(to_id, pm_copy)

    after_from_proximity = {
        color.name: val for color, val in monopoly_proximity(from_id, pm_copy).items()
    }
    after_to_proximity = {
        color.name: val for color, val in monopoly_proximity(to_id, pm_copy).items()
    }

    # --- Detect monopoly changes ---
    monopolies_created: list[tuple[int, str]] = []
    monopolies_broken: list[tuple[int, str]] = []

    for color in PropertyColor:
        if color in (PropertyColor.RAILROAD, PropertyColor.UTILITY):
            continue

        had_before_from = game.property_manager.has_monopoly(from_id, color)
        has_after_from = pm_copy.has_monopoly(from_id, color)
        had_before_to = game.property_manager.has_monopoly(to_id, color)
        has_after_to = pm_copy.has_monopoly(to_id, color)

        if not had_before_from and has_after_from:
            monopolies_created.append((from_id, color.name))
        if had_before_from and not has_after_from:
            monopolies_broken.append((from_id, color.name))
        if not had_before_to and has_after_to:
            monopolies_created.append((to_id, color.name))
        if had_before_to and not has_after_to:
            monopolies_broken.append((to_id, color.name))

    return {
        "from_player_id": from_id,
        "to_player_id": to_id,
        "before": {
            from_id: before_from,
            to_id: before_to,
        },
        "after": {
            from_id: {
                "net_worth": after_from_nw,
                "monopoly_count": after_from_monopolies,
                "proximity": after_from_proximity,
            },
            to_id: {
                "net_worth": after_to_nw,
                "monopoly_count": after_to_monopolies,
                "proximity": after_to_proximity,
            },
        },
        "from_player_net_change": after_from_nw - before_from["net_worth"],
        "to_player_net_change": after_to_nw - before_to["net_worth"],
        "monopolies_created": monopolies_created,
        "monopolies_broken": monopolies_broken,
    }


# ---------------------------------------------------------------------------
# Property strategic value
# ---------------------------------------------------------------------------

# Approximate landing frequency multiplier by color group (relative to average).
# Based on standard Monopoly probability analysis.
_COLOR_FREQUENCY: dict[PropertyColor, float] = {
    PropertyColor.BROWN: 0.6,
    PropertyColor.LIGHT_BLUE: 0.8,
    PropertyColor.MAGENTA: 0.9,
    PropertyColor.ORANGE: 1.3,
    PropertyColor.RED: 1.2,
    PropertyColor.YELLOW: 1.0,
    PropertyColor.GREEN: 0.9,
    PropertyColor.DARK_BLUE: 0.7,
    PropertyColor.RAILROAD: 1.0,
    PropertyColor.UTILITY: 0.5,
}


def property_strategic_value(
    game: MonopolyGame,
    position: int,
    for_player: int,
) -> float:
    """Estimate the strategic value of a property to a specific player.

    Factors:
        - Base property cost as starting value.
        - Proximity bonus: property is more valuable if it completes or
          nearly completes a monopoly for the player.
        - Frequency bonus: properties on high-traffic areas worth more.
        - Blocking value: holding a property an opponent needs.

    Returns a float score (higher = more valuable to the player).
    """
    pm = game.property_manager
    prop = pm.properties.get(position)
    if prop is None:
        return 0.0

    space = Board.get_space(position)
    if not isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
        return 0.0

    # Base value = property cost
    base_value = float(space.cost)

    # Color group info
    color = prop.color
    if color is None:
        return base_value

    group_positions = PROPERTY_GROUPS.get(color, ())
    group_size = len(group_positions)
    if group_size == 0:
        return base_value

    # How many of this group does the player already own (excluding this one)?
    owned_in_group = sum(
        1
        for pos in group_positions
        if pos != position
        and pm.properties.get(pos) is not None
        and pm.properties[pos].owner == for_player
    )

    # Proximity multiplier: more valuable when it completes or nearly completes
    # a monopoly. Completing a set -> 5x, 1-away -> 3x, partial -> 1.5x.
    if owned_in_group == group_size - 1:
        proximity_mult = 5.0  # This property completes the monopoly
    elif owned_in_group >= group_size - 2 and group_size >= 3:
        proximity_mult = 3.0  # 1 away from monopoly
    elif owned_in_group > 0:
        proximity_mult = 1.5  # Has some properties in group
    else:
        proximity_mult = 1.0  # No other properties in group

    # Landing frequency multiplier
    freq_mult = _COLOR_FREQUENCY.get(color, 1.0)

    # Blocking value: if an opponent is close to monopoly on this group
    blocking_bonus = 0.0
    for pid in range(len(game.players)):
        if pid == for_player:
            continue
        opp_owned = sum(
            1
            for pos in group_positions
            if pos != position
            and pm.properties.get(pos) is not None
            and pm.properties[pos].owner == pid
        )
        if opp_owned == group_size - 1:
            # This property blocks an opponent's monopoly
            blocking_bonus = max(blocking_bonus, base_value * 2.0)
        elif opp_owned >= group_size - 2 and group_size >= 3:
            blocking_bonus = max(blocking_bonus, base_value * 0.5)

    return base_value * proximity_mult * freq_mult + blocking_bonus


# ---------------------------------------------------------------------------
# Suggest valuable trades
# ---------------------------------------------------------------------------


@dataclass
class TradeCandidate:
    """A heuristic trade suggestion."""

    to_player: int
    give_properties: list[int]
    want_properties: list[int]
    give_money: int
    want_money: int
    estimated_value: float


def suggest_valuable_trades(
    game: MonopolyGame,
    player_id: int,
    max_suggestions: int = 5,
) -> list[TradeCandidate]:
    """Generate heuristic trade candidates sorted by estimated value.

    Strategy:
    1. Find color groups where we're 1 property away from a monopoly.
    2. For each missing property, find who owns it.
    3. Propose: offer a non-critical property (+ cash) for the missing one.
    4. Score candidates by the value of completing the monopoly.
    """
    pm = game.property_manager
    player = game.players[player_id]
    candidates: list[TradeCandidate] = []

    # Our properties available to offer (no buildings, not mortgaged)
    our_props = [
        pos
        for pos in pm.get_owned_by(player_id)
        if pm.properties[pos].houses == 0 and not pm.properties[pos].mortgaged
    ]

    for color, positions in PROPERTY_GROUPS.items():
        if color in (PropertyColor.RAILROAD, PropertyColor.UTILITY):
            continue

        group_size = len(positions)
        owned_positions = [
            pos
            for pos in positions
            if pm.properties.get(pos) is not None and pm.properties[pos].owner == player_id
        ]
        missing_positions = [pos for pos in positions if pos not in owned_positions]

        # We need exactly 1 more property to complete this monopoly
        if len(owned_positions) != group_size - 1 or len(missing_positions) != 1:
            continue

        missing_pos = missing_positions[0]
        missing_prop = pm.properties.get(missing_pos)
        if missing_prop is None or missing_prop.owner is None:
            continue  # Unowned, just buy it
        if missing_prop.houses > 0:
            continue  # Can't trade property with buildings

        target_player = missing_prop.owner

        # What can we offer? Find a property we own that isn't critical
        # (not part of a near-complete set for us).
        for offer_pos in our_props:
            if offer_pos in owned_positions:
                continue  # Don't give away our progress in this color

            offer_color = pm.properties[offer_pos].color
            # Don't give away a property from another set we're building
            if offer_color is not None:
                offer_group = PROPERTY_GROUPS.get(offer_color, ())
                offer_owned = sum(
                    1
                    for p in offer_group
                    if pm.properties.get(p) is not None and pm.properties[p].owner == player_id
                )
                if offer_owned >= len(offer_group) - 1 and len(offer_group) > 1:
                    continue  # This property is critical for another monopoly

            # Score: value of completing our monopoly minus what we give up
            want_value = property_strategic_value(game, missing_pos, player_id)
            give_value = property_strategic_value(game, offer_pos, player_id)
            estimated = want_value - give_value

            # Also consider offering cash if the property swap alone isn't enough
            # to entice the opponent (offer up to half the missing property's cost)
            space = Board.get_space(missing_pos)
            cash_offer = 0
            if isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
                cash_offer = min(space.cost // 2, player.money // 4)

            candidates.append(
                TradeCandidate(
                    to_player=target_player,
                    give_properties=[offer_pos],
                    want_properties=[missing_pos],
                    give_money=cash_offer,
                    want_money=0,
                    estimated_value=estimated,
                )
            )

    # Sort by estimated value descending, return top N
    candidates.sort(key=lambda c: c.estimated_value, reverse=True)
    return candidates[:max_suggestions]
