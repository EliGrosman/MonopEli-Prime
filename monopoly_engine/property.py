"""
Property State Management Module

This module tracks the mutable state of properties, railroads, and utilities
in a Monopoly game. It manages ownership, house/hotel counts, and mortgage status.

The Property class represents the runtime state of a property (mutable),
while the immutable property definitions live in board.py.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .types import PropertyColor


@dataclass
class Property:
    """
    Mutable property state tracking ownership, houses, and mortgage status.

    A Property represents the current state of a buyable space on the board.
    It tracks who owns it, how many houses/hotels are built, and whether
    it's mortgaged.

    Attributes:
        position: Board position (0-39) of this property
        owner: Player ID of owner, or None if unowned
        houses: Number of houses (0-4) or 5 for hotel
        mortgaged: Whether the property is currently mortgaged

    Examples:
        >>> prop = Property(position=1)
        >>> prop.is_owned
        False
        >>> prop.owner = 0
        >>> prop.is_owned
        True
        >>> prop.houses = 3
        >>> prop.can_build()
        True
        >>> prop.houses = 5
        >>> prop.is_hotel
        True
    """

    position: int
    owner: int | None = None
    houses: int = 0  # 0-4 houses, 5 = hotel
    mortgaged: bool = False

    @property
    def is_owned(self) -> bool:
        """
        Check if the property is owned by any player.

        Returns:
            True if the property has an owner, False otherwise
        """
        return self.owner is not None

    @property
    def is_hotel(self) -> bool:
        """
        Check if the property has a hotel (represented as 5 houses).

        Returns:
            True if the property has a hotel, False otherwise
        """
        return self.houses == 5

    def can_build(self) -> bool:
        """
        Check if houses can be built on this property.

        Building is allowed if:
        - The property is owned
        - The property is not mortgaged
        - There are fewer than 5 houses (hotel is max)

        Note: This checks local property state only. Additional game rules
        (monopoly ownership, even building, house availability) are checked
        in the rules module.

        Returns:
            True if building is allowed based on property state
        """
        return self.is_owned and not self.mortgaged and self.houses < 5

    def can_mortgage(self) -> bool:
        """
        Check if the property can be mortgaged.

        Mortgaging is allowed if:
        - The property is owned
        - There are no houses on the property
        - The property is not already mortgaged

        Returns:
            True if the property can be mortgaged
        """
        return self.is_owned and self.houses == 0 and not self.mortgaged

    def can_unmortgage(self) -> bool:
        """
        Check if the property can be unmortgaged.

        Unmortgaging is allowed if:
        - The property is owned
        - The property is currently mortgaged

        Note: Player's available funds are checked in the rules module.

        Returns:
            True if the property can be unmortgaged
        """
        return self.is_owned and self.mortgaged

    def to_dict(self) -> dict[str, int | bool | None]:
        """
        Convert property state to a JSON-serializable dictionary.

        Returns:
            Dictionary with position, owner, houses, and mortgaged status
        """
        return {
            "position": self.position,
            "owner": self.owner,
            "houses": self.houses,
            "mortgaged": self.mortgaged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int | bool | None]) -> "Property":
        """
        Create a Property instance from a dictionary.

        Args:
            data: Dictionary containing property state

        Returns:
            New Property instance with the specified state
        """
        # Extract and convert values with proper type handling
        position_val: Any = data["position"]
        owner_val: Any = data["owner"]
        houses_val: Any = data["houses"]
        mortgaged_val: Any = data["mortgaged"]

        # Convert to proper types
        position = int(position_val)
        houses = int(houses_val)
        owner = int(owner_val) if owner_val is not None else None
        mortgaged = bool(mortgaged_val)

        return cls(
            position=position,
            owner=owner,
            houses=houses,
            mortgaged=mortgaged,
        )


class PropertyManager:
    """
    Manages all properties in the game.

    The PropertyManager maintains the state of all 28 buyable properties
    (22 properties + 4 railroads + 2 utilities) and provides methods for
    querying ownership and monopoly status.

    Attributes:
        properties: Dictionary mapping board position to Property state

    Examples:
        >>> manager = PropertyManager()
        >>> len(manager.properties)
        28
        >>> prop = manager.properties[1]
        >>> prop.owner = 0
        >>> manager.get_owned_by_player(0)
        [1]
    """

    def __init__(self) -> None:
        """Initialize the PropertyManager with all buyable properties."""
        self.properties: dict[int, Property] = {}
        self._init_properties()

    def _init_properties(self) -> None:
        """
        Initialize all 28 buyable properties on the board.

        This scans the Board.SPACES tuple and creates a Property instance
        for each space that has a 'cost' attribute (properties, railroads,
        and utilities).
        """
        # Import here to avoid circular dependency
        from .board import Board

        for i, space in enumerate(Board.SPACES):
            # Buyable spaces have a 'cost' attribute
            if hasattr(space, "cost"):
                self.properties[i] = Property(position=i)

    def get_owned_by_player(self, player_id: int) -> list[int]:
        """
        Get all property positions owned by a specific player.

        Args:
            player_id: The player's ID

        Returns:
            List of board positions owned by the player, sorted by position
        """
        return sorted([
            pos for pos, prop in self.properties.items()
            if prop.owner == player_id
        ])

    def has_monopoly(self, player_id: int, color: "PropertyColor") -> bool:
        """
        Check if a player owns all properties of a given color (monopoly).

        A monopoly is when a player owns all properties in a color group.
        This is required for building houses and doubles the base rent.

        Args:
            player_id: The player's ID
            color: The property color to check

        Returns:
            True if the player owns all properties of the specified color

        Examples:
            >>> manager = PropertyManager()
            >>> # Brown properties are at positions 1 and 3
            >>> manager.properties[1].owner = 0
            >>> manager.properties[3].owner = 0
            >>> from .types import PropertyColor
            >>> manager.has_monopoly(0, PropertyColor.BROWN)
            True
        """
        # Import here to avoid circular dependency
        from .board import Board

        # Get all positions for this color group
        group = Board.get_property_group(color)

        # Check if player owns all properties in the group
        return all(
            self.properties[pos].owner == player_id
            for pos in group
        )

    def get_monopolies(self, player_id: int) -> list["PropertyColor"]:
        """
        Get all color groups where the player has a monopoly.

        Args:
            player_id: The player's ID

        Returns:
            List of PropertyColor enums for which the player has a monopoly
        """
        from .types import PropertyColor
        from .board import Board, PropertySpace

        monopolies: list["PropertyColor"] = []

        # Check each color group
        checked_colors: set[PropertyColor] = set()
        for space in Board.SPACES:
            if isinstance(space, PropertySpace):
                color = space.color
                if color not in checked_colors:
                    checked_colors.add(color)
                    if self.has_monopoly(player_id, color):
                        monopolies.append(color)

        return monopolies

    def get_properties_in_group(self, color: "PropertyColor") -> list[int]:
        """
        Get all property positions in a color group.

        Args:
            color: The property color

        Returns:
            List of board positions for properties of the specified color
        """
        from .board import Board
        return Board.get_property_group(color)

    def count_houses_in_group(self, color: "PropertyColor") -> int:
        """
        Count total houses in a color group (for even building rule).

        Args:
            color: The property color

        Returns:
            Total number of houses across all properties in the group
        """
        group = self.get_properties_in_group(color)
        return sum(
            self.properties[pos].houses
            for pos in group
            if self.properties[pos].houses < 5  # Don't count hotels
        )

    def get_min_houses_in_group(self, color: "PropertyColor") -> int:
        """
        Get the minimum number of houses on any property in a color group.

        This is used to enforce even building - you can't build if another
        property in the group has fewer houses.

        Args:
            color: The property color

        Returns:
            Minimum house count in the group (hotels count as 5)
        """
        group = self.get_properties_in_group(color)
        return min(self.properties[pos].houses for pos in group)

    def get_max_houses_in_group(self, color: "PropertyColor") -> int:
        """
        Get the maximum number of houses on any property in a color group.

        This is used to enforce even building - you can't have more than
        one house difference between properties in the group.

        Args:
            color: The property color

        Returns:
            Maximum house count in the group (hotels count as 5)
        """
        group = self.get_properties_in_group(color)
        return max(self.properties[pos].houses for pos in group)

    def to_dict(self) -> dict[str, list[dict[str, int | bool | None]]]:
        """
        Convert all property states to a JSON-serializable dictionary.

        Returns:
            Dictionary with 'properties' key containing list of property states
        """
        return {
            "properties": [
                prop.to_dict()
                for prop in sorted(self.properties.values(), key=lambda p: p.position)
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[dict[str, int | bool | None]]]) -> "PropertyManager":
        """
        Create a PropertyManager instance from a dictionary.

        Args:
            data: Dictionary containing property states

        Returns:
            New PropertyManager instance with the specified state
        """
        manager = cls()
        for prop_data in data["properties"]:
            pos_val: Any = prop_data["position"]
            pos = int(pos_val)
            manager.properties[pos] = Property.from_dict(prop_data)
        return manager

    def reset(self) -> None:
        """
        Reset all properties to unowned state.

        This clears all ownership, houses, and mortgage status.
        Useful for starting a new game with the same PropertyManager instance.
        """
        for prop in self.properties.values():
            prop.owner = None
            prop.houses = 0
            prop.mortgaged = False
