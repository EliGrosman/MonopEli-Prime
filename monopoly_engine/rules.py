"""Game rules for the Monopoly engine.

This module contains pure functions implementing Monopoly rules including
rent calculation, building rules, mortgage rules, and bankruptcy handling.
"""

from typing import TYPE_CHECKING

from .types import PropertyColor, PROPERTY_GROUPS, POSITION_TO_COLOR
from .board import Board, PropertySpace, RailroadSpace, UtilitySpace

if TYPE_CHECKING:
    from .player import Player
    from .property import Property, PropertyManager


def calculate_rent(
    property_manager: "PropertyManager",
    position: int,
    dice_roll: int = 0,
    is_from_card: bool = False,
) -> int:
    """Calculate rent for a property.

    Args:
        property_manager: The game's property manager
        position: The property position
        dice_roll: The dice roll (for utilities)
        is_from_card: True if player landed here from a Chance card
            (affects utility rent calculation)

    Returns:
        The rent amount (0 if property is unowned or mortgaged)
    """
    prop = property_manager.get(position)
    if prop is None:
        return 0

    if not prop.is_owned or prop.mortgaged:
        return 0

    space = Board.get_space(position)

    if isinstance(space, PropertySpace):
        return _calculate_property_rent(property_manager, prop, space)

    elif isinstance(space, RailroadSpace):
        return _calculate_railroad_rent(property_manager, prop, space, is_from_card)

    elif isinstance(space, UtilitySpace):
        return _calculate_utility_rent(property_manager, prop, dice_roll, is_from_card)

    return 0


def _calculate_property_rent(
    property_manager: "PropertyManager",
    prop: "Property",
    space: PropertySpace,
) -> int:
    """Calculate rent for a regular property."""
    if prop.owner is None:
        return 0

    # Hotel or houses
    if prop.houses > 0:
        return space.rent[prop.houses]

    # Monopoly bonus (double rent with no houses)
    if property_manager.has_monopoly(prop.owner, space.color):
        return space.rent[0] * 2

    # Base rent
    return space.rent[0]


def _calculate_railroad_rent(
    property_manager: "PropertyManager",
    prop: "Property",
    space: RailroadSpace,
    is_from_card: bool = False,
) -> int:
    """Calculate rent for a railroad."""
    if prop.owner is None:
        return 0

    railroads_owned = property_manager.count_railroads_owned(prop.owner)
    base_rent = space.rent[railroads_owned - 1]

    # From Chance card "Advance to nearest Railroad" = double rent
    if is_from_card:
        return base_rent * 2

    return base_rent


def _calculate_utility_rent(
    property_manager: "PropertyManager",
    prop: "Property",
    dice_roll: int,
    is_from_card: bool = False,
) -> int:
    """Calculate rent for a utility."""
    if prop.owner is None:
        return 0

    utilities_owned = property_manager.count_utilities_owned(prop.owner)

    # From Chance card "Advance to nearest Utility" = 10x dice roll
    if is_from_card:
        return dice_roll * 10

    # Normal rent: 4x or 10x dice roll
    multiplier = 10 if utilities_owned == 2 else 4
    return dice_roll * multiplier


def can_buy_property(
    player: "Player",
    property_manager: "PropertyManager",
    position: int,
) -> tuple[bool, str]:
    """Check if player can buy a property.

    Returns:
        Tuple of (can_buy, reason)
    """
    prop = property_manager.get(position)
    if prop is None:
        return False, "Not a buyable property"

    if prop.is_owned:
        return False, "Property already owned"

    space = Board.get_space(position)
    if not isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
        return False, "Not a buyable property"

    if player.money < space.cost:
        return False, "Insufficient funds"

    return True, ""


def get_property_cost(position: int) -> int:
    """Get the purchase cost of a property."""
    space = Board.get_space(position)
    if isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
        return space.cost
    return 0


def can_build_house(
    player: "Player",
    property_manager: "PropertyManager",
    position: int,
    houses_remaining: int,
    hotels_remaining: int,
) -> tuple[bool, str]:
    """Check if player can build a house on a property.

    Returns:
        Tuple of (can_build, reason)
    """
    prop = property_manager.get(position)
    if prop is None:
        return False, "Not a property"

    space = Board.get_space(position)
    if not isinstance(space, PropertySpace):
        return False, "Cannot build on this property type"

    if prop.owner != player.id:
        return False, "You don't own this property"

    if prop.mortgaged:
        return False, "Property is mortgaged"

    if prop.houses >= 5:
        return False, "Already has a hotel"

    color = POSITION_TO_COLOR.get(position)
    if color is None:
        return False, "Invalid property"

    if not property_manager.has_monopoly(player.id, color):
        return False, "You don't have a monopoly on this color"

    # Check even building rule
    min_houses = property_manager.get_min_houses_in_group(color)
    if prop.houses > min_houses:
        return False, "Must build evenly across color group"

    # Check house/hotel availability
    if prop.houses == 4:
        # Building a hotel
        if hotels_remaining < 1:
            return False, "No hotels remaining"
    else:
        # Building a house
        if houses_remaining < 1:
            return False, "No houses remaining"

    # Check funds
    if player.money < space.house_cost:
        return False, "Insufficient funds"

    return True, ""


def get_building_cost(position: int) -> int:
    """Get the cost to build a house/hotel on a property."""
    space = Board.get_space(position)
    if isinstance(space, PropertySpace):
        return space.house_cost
    return 0


def can_sell_house(
    player: "Player",
    property_manager: "PropertyManager",
    position: int,
) -> tuple[bool, str]:
    """Check if player can sell a house from a property.

    Returns:
        Tuple of (can_sell, reason)
    """
    prop = property_manager.get(position)
    if prop is None:
        return False, "Not a property"

    space = Board.get_space(position)
    if not isinstance(space, PropertySpace):
        return False, "Cannot sell houses on this property type"

    if prop.owner != player.id:
        return False, "You don't own this property"

    if prop.houses == 0:
        return False, "No houses to sell"

    color = POSITION_TO_COLOR.get(position)
    if color is None:
        return False, "Invalid property"

    # Check even selling rule
    max_houses = property_manager.get_max_houses_in_group(color)
    if prop.houses < max_houses:
        return False, "Must sell evenly across color group"

    return True, ""


def get_house_sale_value(position: int) -> int:
    """Get the sale value of a house (half of build cost)."""
    space = Board.get_space(position)
    if isinstance(space, PropertySpace):
        return space.house_cost // 2
    return 0


def can_mortgage_property(
    player: "Player",
    property_manager: "PropertyManager",
    position: int,
) -> tuple[bool, str]:
    """Check if player can mortgage a property.

    Returns:
        Tuple of (can_mortgage, reason)
    """
    prop = property_manager.get(position)
    if prop is None:
        return False, "Not a property"

    if prop.owner != player.id:
        return False, "You don't own this property"

    if prop.mortgaged:
        return False, "Property is already mortgaged"

    if prop.houses > 0:
        return False, "Must sell all houses first"

    # Check if any property in the color group has houses
    color = POSITION_TO_COLOR.get(position)
    if color is not None and color not in (PropertyColor.RAILROAD, PropertyColor.UTILITY):
        for pos in PROPERTY_GROUPS.get(color, ()):
            other_prop = property_manager.get(pos)
            if other_prop and other_prop.houses > 0:
                return False, "Must sell all houses in color group first"

    return True, ""


def get_mortgage_value(position: int) -> int:
    """Get the mortgage value of a property."""
    space = Board.get_space(position)
    if isinstance(space, PropertySpace):
        return space.mortgage_value
    elif isinstance(space, RailroadSpace):
        return space.mortgage_value
    elif isinstance(space, UtilitySpace):
        return space.mortgage_value
    return 0


def can_unmortgage_property(
    player: "Player",
    property_manager: "PropertyManager",
    position: int,
) -> tuple[bool, str]:
    """Check if player can unmortgage a property.

    Returns:
        Tuple of (can_unmortgage, reason)
    """
    prop = property_manager.get(position)
    if prop is None:
        return False, "Not a property"

    if prop.owner != player.id:
        return False, "You don't own this property"

    if not prop.mortgaged:
        return False, "Property is not mortgaged"

    unmortgage_cost = get_unmortgage_cost(position)
    if player.money < unmortgage_cost:
        return False, "Insufficient funds"

    return True, ""


def get_unmortgage_cost(position: int) -> int:
    """Get the cost to unmortgage a property (110% of mortgage value)."""
    mortgage_value = get_mortgage_value(position)
    return int(mortgage_value * 1.1)


def calculate_net_worth(
    player: "Player",
    property_manager: "PropertyManager",
) -> int:
    """Calculate a player's total net worth.

    Net worth includes:
    - Cash on hand
    - Property values (mortgage value if unmortgaged, 0 if mortgaged)
    - House/hotel values (half of build cost)
    """
    total = player.money

    for pos in property_manager.get_owned_by(player.id):
        prop = property_manager.get(pos)
        if prop is None:
            continue

        space = Board.get_space(pos)

        # Property value (can mortgage for this amount)
        if not prop.mortgaged:
            if isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
                total += space.mortgage_value

        # House value (can sell for half of build cost)
        if isinstance(space, PropertySpace) and prop.houses > 0:
            if prop.houses == 5:
                # Hotel = 4 houses worth + hotel itself
                total += space.house_cost // 2 * 5
            else:
                total += space.house_cost // 2 * prop.houses

    return total


def can_afford_rent(
    player: "Player",
    property_manager: "PropertyManager",
    rent_amount: int,
) -> tuple[bool, int]:
    """Check if player can afford rent, possibly by selling/mortgaging.

    Returns:
        Tuple of (can_afford, shortfall)
        shortfall is 0 if can afford, otherwise how much more is needed
    """
    net_worth = calculate_net_worth(player, property_manager)

    if net_worth >= rent_amount:
        return True, 0

    return False, rent_amount - net_worth


def is_bankrupt(
    player: "Player",
    property_manager: "PropertyManager",
    debt: int,
) -> bool:
    """Check if player is bankrupt (cannot pay a debt).

    A player is bankrupt if their total net worth is less than the debt.
    """
    net_worth = calculate_net_worth(player, property_manager)
    return net_worth < debt


def get_buildable_properties(
    player: "Player",
    property_manager: "PropertyManager",
    houses_remaining: int,
    hotels_remaining: int,
) -> list[int]:
    """Get all property positions where player can build."""
    buildable = []
    for pos in property_manager.get_owned_by(player.id):
        can_build, _ = can_build_house(
            player, property_manager, pos, houses_remaining, hotels_remaining
        )
        if can_build:
            buildable.append(pos)
    return buildable


def get_sellable_houses(
    player: "Player",
    property_manager: "PropertyManager",
) -> list[int]:
    """Get all property positions where player can sell houses."""
    sellable = []
    for pos in property_manager.get_owned_by(player.id):
        can_sell, _ = can_sell_house(player, property_manager, pos)
        if can_sell:
            sellable.append(pos)
    return sellable


def get_mortgageable_properties(
    player: "Player",
    property_manager: "PropertyManager",
) -> list[int]:
    """Get all property positions that player can mortgage."""
    mortgageable = []
    for pos in property_manager.get_owned_by(player.id):
        can_mortgage, _ = can_mortgage_property(player, property_manager, pos)
        if can_mortgage:
            mortgageable.append(pos)
    return mortgageable


def get_unmortgageable_properties(
    player: "Player",
    property_manager: "PropertyManager",
) -> list[int]:
    """Get all property positions that player can unmortgage."""
    unmortgageable = []
    for pos in property_manager.get_owned_by(player.id):
        can_unmortgage, _ = can_unmortgage_property(player, property_manager, pos)
        if can_unmortgage:
            unmortgageable.append(pos)
    return unmortgageable


def validate_trade(
    from_player: "Player",
    to_player: "Player",
    property_manager: "PropertyManager",
    give_properties: list[int],
    give_money: int,
    want_properties: list[int],
    want_money: int,
) -> tuple[bool, str]:
    """Validate a trade between two players.

    Returns:
        Tuple of (is_valid, reason)
    """
    # Check that from_player owns all give_properties
    for pos in give_properties:
        prop = property_manager.get(pos)
        if prop is None or prop.owner != from_player.id:
            return False, f"Player doesn't own property at position {pos}"
        if prop.houses > 0:
            return False, f"Cannot trade property with houses at position {pos}"

    # Check that to_player owns all want_properties
    for pos in want_properties:
        prop = property_manager.get(pos)
        if prop is None or prop.owner != to_player.id:
            return False, f"Other player doesn't own property at position {pos}"
        if prop.houses > 0:
            return False, f"Cannot trade property with houses at position {pos}"

    # Check money constraints
    if give_money > 0 and from_player.money < give_money:
        return False, "Insufficient funds for trade"

    if want_money > 0 and to_player.money < want_money:
        return False, "Other player has insufficient funds"

    return True, ""
