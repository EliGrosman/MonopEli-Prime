"""Property state management for the Monopoly game engine.

This module handles the mutable state of properties - ownership, houses,
and mortgage status. The immutable property definitions are in board.py.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import PropertyColor, PropertyStateData, PROPERTY_GROUPS, POSITION_TO_COLOR
from .board import Board, PropertySpace, RailroadSpace, UtilitySpace

if TYPE_CHECKING:
    pass


@dataclass
class Property:
    """Mutable state for a single property.

    This tracks ownership, houses built, and mortgage status.
    The static property data (cost, rent, etc.) is in board.py.
    """

    position: int
    owner: int | None = None
    houses: int = 0  # 0-4 for houses, 5 for hotel
    mortgaged: bool = False

    @property
    def is_owned(self) -> bool:
        """Check if this property has an owner."""
        return self.owner is not None

    @property
    def has_hotel(self) -> bool:
        """Check if this property has a hotel."""
        return self.houses == 5

    @property
    def has_houses(self) -> bool:
        """Check if this property has any houses (not hotel)."""
        return 0 < self.houses < 5

    @property
    def has_buildings(self) -> bool:
        """Check if this property has any buildings (houses or hotel)."""
        return self.houses > 0

    @property
    def color(self) -> PropertyColor | None:
        """Get the color of this property."""
        return POSITION_TO_COLOR.get(self.position)

    def can_build(self) -> bool:
        """Check basic buildability (owned, not mortgaged, room for more)."""
        return self.is_owned and not self.mortgaged and self.houses < 5

    def can_mortgage(self) -> bool:
        """Check if property can be mortgaged (no buildings, owned, not mortgaged)."""
        return self.is_owned and self.houses == 0 and not self.mortgaged

    def can_unmortgage(self) -> bool:
        """Check if property can be unmortgaged."""
        return self.is_owned and self.mortgaged

    def to_dict(self) -> PropertyStateData:
        """Serialize to JSON-compatible dict."""
        return PropertyStateData(
            position=self.position,
            owner=self.owner,
            houses=self.houses,
            mortgaged=self.mortgaged,
        )

    @classmethod
    def from_dict(cls, data: PropertyStateData) -> "Property":
        """Create Property from serialized dict."""
        return cls(
            position=data["position"],
            owner=data.get("owner"),
            houses=data.get("houses", 0),
            mortgaged=data.get("mortgaged", False),
        )


@dataclass
class PropertyManager:
    """Manages all properties in the game.

    This class tracks the state of all 28 buyable properties and provides
    methods for querying ownership, monopolies, and building eligibility.
    """

    properties: dict[int, Property] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize all properties if not provided."""
        if not self.properties:
            self._init_properties()

    def _init_properties(self) -> None:
        """Initialize all 28 buyable properties."""
        for i, space in enumerate(Board.SPACES):
            if isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
                self.properties[i] = Property(position=i)

    def get(self, position: int) -> Property | None:
        """Get property at a position, or None if not a property."""
        return self.properties.get(position)

    def get_owned_by(self, player_id: int) -> list[int]:
        """Get all property positions owned by a player."""
        return [
            pos for pos, prop in self.properties.items()
            if prop.owner == player_id
        ]

    def get_unowned(self) -> list[int]:
        """Get all unowned property positions."""
        return [
            pos for pos, prop in self.properties.items()
            if prop.owner is None
        ]

    def get_mortgaged_by(self, player_id: int) -> list[int]:
        """Get all mortgaged property positions owned by a player."""
        return [
            pos for pos, prop in self.properties.items()
            if prop.owner == player_id and prop.mortgaged
        ]

    def get_unmortgaged_by(self, player_id: int) -> list[int]:
        """Get all unmortgaged property positions owned by a player."""
        return [
            pos for pos, prop in self.properties.items()
            if prop.owner == player_id and not prop.mortgaged
        ]

    def has_monopoly(self, player_id: int, color: PropertyColor) -> bool:
        """Check if player owns all properties of a color group."""
        group = PROPERTY_GROUPS.get(color, ())
        if not group:
            return False
        return all(
            self.properties.get(pos) is not None
            and self.properties[pos].owner == player_id
            for pos in group
        )

    def get_monopolies(self, player_id: int) -> list[PropertyColor]:
        """Get all color groups where player has a monopoly."""
        return [
            color for color in PropertyColor
            if self.has_monopoly(player_id, color)
        ]

    def count_railroads_owned(self, player_id: int) -> int:
        """Count how many railroads a player owns."""
        return sum(
            1 for pos in PROPERTY_GROUPS[PropertyColor.RAILROAD]
            if self.properties.get(pos) is not None
            and self.properties[pos].owner == player_id
        )

    def count_utilities_owned(self, player_id: int) -> int:
        """Count how many utilities a player owns."""
        return sum(
            1 for pos in PROPERTY_GROUPS[PropertyColor.UTILITY]
            if self.properties.get(pos) is not None
            and self.properties[pos].owner == player_id
        )

    def count_houses_in_group(self, player_id: int, color: PropertyColor) -> int:
        """Count total houses in a color group owned by player."""
        if not self.has_monopoly(player_id, color):
            return 0
        return sum(
            self.properties[pos].houses
            for pos in PROPERTY_GROUPS.get(color, ())
            if pos in self.properties and self.properties[pos].houses < 5
        )

    def count_hotels_in_group(self, player_id: int, color: PropertyColor) -> int:
        """Count total hotels in a color group owned by player."""
        if not self.has_monopoly(player_id, color):
            return 0
        return sum(
            1
            for pos in PROPERTY_GROUPS.get(color, ())
            if pos in self.properties and self.properties[pos].houses == 5
        )

    def count_total_houses(self, player_id: int) -> int:
        """Count total houses owned by player (not hotels)."""
        return sum(
            prop.houses
            for prop in self.properties.values()
            if prop.owner == player_id and 0 < prop.houses < 5
        )

    def count_total_hotels(self, player_id: int) -> int:
        """Count total hotels owned by player."""
        return sum(
            1
            for prop in self.properties.values()
            if prop.owner == player_id and prop.houses == 5
        )

    def get_min_houses_in_group(self, color: PropertyColor) -> int:
        """Get minimum number of houses on any property in a group."""
        group = PROPERTY_GROUPS.get(color, ())
        if not group:
            return 0
        return min(
            self.properties[pos].houses
            for pos in group
            if pos in self.properties
        )

    def get_max_houses_in_group(self, color: PropertyColor) -> int:
        """Get maximum number of houses on any property in a group."""
        group = PROPERTY_GROUPS.get(color, ())
        if not group:
            return 0
        return max(
            self.properties[pos].houses
            for pos in group
            if pos in self.properties
        )

    def can_build_house_on(self, player_id: int, position: int) -> bool:
        """Check if a house can be built on a property (basic check).

        Full validation including money and house supply is in rules.py.
        """
        prop = self.properties.get(position)
        if prop is None:
            return False

        # Must own it
        if prop.owner != player_id:
            return False

        # Must not be mortgaged
        if prop.mortgaged:
            return False

        # Must not already have hotel
        if prop.houses >= 5:
            return False

        # Must be a regular property (not railroad/utility)
        space = Board.get_space(position)
        if not isinstance(space, PropertySpace):
            return False

        color = POSITION_TO_COLOR.get(position)
        if color is None:
            return False

        # Must have monopoly
        if not self.has_monopoly(player_id, color):
            return False

        # Must build evenly - can only build if at min or tied for min
        min_houses = self.get_min_houses_in_group(color)
        if prop.houses > min_houses:
            return False

        return True

    def can_sell_house_on(self, player_id: int, position: int) -> bool:
        """Check if a house can be sold from a property (basic check)."""
        prop = self.properties.get(position)
        if prop is None:
            return False

        # Must own it
        if prop.owner != player_id:
            return False

        # Must have at least one building
        if prop.houses == 0:
            return False

        # Must be a regular property
        space = Board.get_space(position)
        if not isinstance(space, PropertySpace):
            return False

        color = POSITION_TO_COLOR.get(position)
        if color is None:
            return False

        # Must sell evenly - can only sell if at max or tied for max
        max_houses = self.get_max_houses_in_group(color)
        if prop.houses < max_houses:
            return False

        return True

    def transfer_property(self, position: int, to_player: int | None) -> bool:
        """Transfer property ownership. Returns True if successful."""
        prop = self.properties.get(position)
        if prop is None:
            return False
        prop.owner = to_player
        return True

    def reset_property(self, position: int) -> bool:
        """Reset property to unowned state with no buildings."""
        prop = self.properties.get(position)
        if prop is None:
            return False
        prop.owner = None
        prop.houses = 0
        prop.mortgaged = False
        return True

    def to_dict(self) -> dict[str, PropertyStateData]:
        """Serialize all properties to JSON-compatible dict."""
        return {
            str(pos): prop.to_dict()
            for pos, prop in self.properties.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, PropertyStateData]) -> "PropertyManager":
        """Create PropertyManager from serialized dict."""
        manager = cls()
        for pos_str, prop_data in data.items():
            pos = int(pos_str)
            if pos in manager.properties:
                manager.properties[pos] = Property.from_dict(prop_data)
        return manager
