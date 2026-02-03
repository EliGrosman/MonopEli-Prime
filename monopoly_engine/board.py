"""Board definitions for the Monopoly game engine.

This module defines the static board layout including all 40 spaces.
All definitions are frozen dataclasses for immutability.
"""

from dataclasses import dataclass
from typing import ClassVar

from .types import PROPERTY_GROUPS, PropertyColor, SpaceType


@dataclass(frozen=True)
class Space:
    """Base class for all board spaces."""

    position: int
    name: str
    space_type: SpaceType

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        return {
            "position": self.position,
            "name": self.name,
            "space_type": self.space_type.name,
        }


@dataclass(frozen=True)
class PropertySpace(Space):
    """A property that can be bought, built on, and rented."""

    color: PropertyColor
    cost: int
    rent: tuple[int, ...]  # (base, 1h, 2h, 3h, 4h, hotel)
    house_cost: int
    mortgage_value: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        base = super().to_dict()
        base.update(
            {
                "color": self.color.name,
                "cost": self.cost,
                "rent": list(self.rent),
                "house_cost": self.house_cost,
                "mortgage_value": self.mortgage_value,
            }
        )
        return base


@dataclass(frozen=True)
class RailroadSpace(Space):
    """A railroad property."""

    cost: int
    rent: tuple[int, int, int, int]  # (1 owned, 2 owned, 3 owned, 4 owned)
    mortgage_value: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        base = super().to_dict()
        base.update(
            {
                "cost": self.cost,
                "rent": list(self.rent),
                "mortgage_value": self.mortgage_value,
            }
        )
        return base


@dataclass(frozen=True)
class UtilitySpace(Space):
    """A utility property (Electric Company or Water Works)."""

    cost: int
    mortgage_value: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        base = super().to_dict()
        base.update(
            {
                "cost": self.cost,
                "mortgage_value": self.mortgage_value,
            }
        )
        return base


@dataclass(frozen=True)
class TaxSpace(Space):
    """A tax space (Income Tax or Luxury Tax)."""

    amount: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        base = super().to_dict()
        base["amount"] = self.amount
        return base


class Board:
    """The Monopoly board definition.

    This class provides access to all 40 board spaces and utility methods
    for querying the board layout.
    """

    # All 40 board spaces
    SPACES: ClassVar[tuple[Space, ...]] = (
        # Bottom row (right to left)
        Space(0, "GO", SpaceType.GO),
        PropertySpace(
            1,
            "Mediterranean Avenue",
            SpaceType.PROPERTY,
            PropertyColor.BROWN,
            60,
            (2, 10, 30, 90, 160, 250),
            50,
            30,
        ),
        Space(2, "Community Chest", SpaceType.COMMUNITY_CHEST),
        PropertySpace(
            3,
            "Baltic Avenue",
            SpaceType.PROPERTY,
            PropertyColor.BROWN,
            60,
            (4, 20, 60, 180, 320, 450),
            50,
            30,
        ),
        TaxSpace(4, "Income Tax", SpaceType.INCOME_TAX, 200),
        RailroadSpace(
            5, "Reading Railroad", SpaceType.RAILROAD, 200, (25, 50, 100, 200), 100
        ),
        PropertySpace(
            6,
            "Oriental Avenue",
            SpaceType.PROPERTY,
            PropertyColor.LIGHT_BLUE,
            100,
            (6, 30, 90, 270, 400, 550),
            50,
            50,
        ),
        Space(7, "Chance", SpaceType.CHANCE),
        PropertySpace(
            8,
            "Vermont Avenue",
            SpaceType.PROPERTY,
            PropertyColor.LIGHT_BLUE,
            100,
            (6, 30, 90, 270, 400, 550),
            50,
            50,
        ),
        PropertySpace(
            9,
            "Connecticut Avenue",
            SpaceType.PROPERTY,
            PropertyColor.LIGHT_BLUE,
            120,
            (8, 40, 100, 300, 450, 600),
            50,
            60,
        ),
        # Left column (bottom to top)
        Space(10, "Jail / Just Visiting", SpaceType.JAIL),
        PropertySpace(
            11,
            "St. Charles Place",
            SpaceType.PROPERTY,
            PropertyColor.MAGENTA,
            140,
            (10, 50, 150, 450, 625, 750),
            100,
            70,
        ),
        UtilitySpace(12, "Electric Company", SpaceType.UTILITY, 150, 75),
        PropertySpace(
            13,
            "States Avenue",
            SpaceType.PROPERTY,
            PropertyColor.MAGENTA,
            140,
            (10, 50, 150, 450, 625, 750),
            100,
            70,
        ),
        PropertySpace(
            14,
            "Virginia Avenue",
            SpaceType.PROPERTY,
            PropertyColor.MAGENTA,
            160,
            (12, 60, 180, 500, 700, 900),
            100,
            80,
        ),
        RailroadSpace(
            15,
            "Pennsylvania Railroad",
            SpaceType.RAILROAD,
            200,
            (25, 50, 100, 200),
            100,
        ),
        PropertySpace(
            16,
            "St. James Place",
            SpaceType.PROPERTY,
            PropertyColor.ORANGE,
            180,
            (14, 70, 200, 550, 750, 950),
            100,
            90,
        ),
        Space(17, "Community Chest", SpaceType.COMMUNITY_CHEST),
        PropertySpace(
            18,
            "Tennessee Avenue",
            SpaceType.PROPERTY,
            PropertyColor.ORANGE,
            180,
            (14, 70, 200, 550, 750, 950),
            100,
            90,
        ),
        PropertySpace(
            19,
            "New York Avenue",
            SpaceType.PROPERTY,
            PropertyColor.ORANGE,
            200,
            (16, 80, 220, 600, 800, 1000),
            100,
            100,
        ),
        # Top row (left to right)
        Space(20, "Free Parking", SpaceType.FREE_PARKING),
        PropertySpace(
            21,
            "Kentucky Avenue",
            SpaceType.PROPERTY,
            PropertyColor.RED,
            220,
            (18, 90, 250, 700, 875, 1050),
            150,
            110,
        ),
        Space(22, "Chance", SpaceType.CHANCE),
        PropertySpace(
            23,
            "Indiana Avenue",
            SpaceType.PROPERTY,
            PropertyColor.RED,
            220,
            (18, 90, 250, 700, 875, 1050),
            150,
            110,
        ),
        PropertySpace(
            24,
            "Illinois Avenue",
            SpaceType.PROPERTY,
            PropertyColor.RED,
            240,
            (20, 100, 300, 750, 925, 1100),
            150,
            120,
        ),
        RailroadSpace(
            25, "B&O Railroad", SpaceType.RAILROAD, 200, (25, 50, 100, 200), 100
        ),
        PropertySpace(
            26,
            "Atlantic Avenue",
            SpaceType.PROPERTY,
            PropertyColor.YELLOW,
            260,
            (22, 110, 330, 800, 975, 1150),
            150,
            130,
        ),
        PropertySpace(
            27,
            "Ventnor Avenue",
            SpaceType.PROPERTY,
            PropertyColor.YELLOW,
            260,
            (22, 110, 330, 800, 975, 1150),
            150,
            130,
        ),
        UtilitySpace(28, "Water Works", SpaceType.UTILITY, 150, 75),
        PropertySpace(
            29,
            "Marvin Gardens",
            SpaceType.PROPERTY,
            PropertyColor.YELLOW,
            280,
            (24, 120, 360, 850, 1025, 1200),
            150,
            140,
        ),
        # Right column (top to bottom)
        Space(30, "Go To Jail", SpaceType.GO_TO_JAIL),
        PropertySpace(
            31,
            "Pacific Avenue",
            SpaceType.PROPERTY,
            PropertyColor.GREEN,
            300,
            (26, 130, 390, 900, 1100, 1275),
            200,
            150,
        ),
        PropertySpace(
            32,
            "North Carolina Avenue",
            SpaceType.PROPERTY,
            PropertyColor.GREEN,
            300,
            (26, 130, 390, 900, 1100, 1275),
            200,
            150,
        ),
        Space(33, "Community Chest", SpaceType.COMMUNITY_CHEST),
        PropertySpace(
            34,
            "Pennsylvania Avenue",
            SpaceType.PROPERTY,
            PropertyColor.GREEN,
            320,
            (28, 150, 450, 1000, 1200, 1400),
            200,
            160,
        ),
        RailroadSpace(
            35, "Short Line", SpaceType.RAILROAD, 200, (25, 50, 100, 200), 100
        ),
        Space(36, "Chance", SpaceType.CHANCE),
        PropertySpace(
            37,
            "Park Place",
            SpaceType.PROPERTY,
            PropertyColor.DARK_BLUE,
            350,
            (35, 175, 500, 1100, 1300, 1500),
            200,
            175,
        ),
        TaxSpace(38, "Luxury Tax", SpaceType.LUXURY_TAX, 100),
        PropertySpace(
            39,
            "Boardwalk",
            SpaceType.PROPERTY,
            PropertyColor.DARK_BLUE,
            400,
            (50, 200, 600, 1400, 1700, 2000),
            200,
            200,
        ),
    )

    # Pre-computed indices for quick lookups
    _property_positions: ClassVar[tuple[int, ...]] = tuple(
        i for i, s in enumerate(SPACES) if isinstance(s, PropertySpace)
    )
    _railroad_positions: ClassVar[tuple[int, ...]] = (5, 15, 25, 35)
    _utility_positions: ClassVar[tuple[int, ...]] = (12, 28)
    _chance_positions: ClassVar[tuple[int, ...]] = (7, 22, 36)
    _chest_positions: ClassVar[tuple[int, ...]] = (2, 17, 33)

    @classmethod
    def get_space(cls, position: int) -> Space:
        """Get the space at a given position (wraps around)."""
        return cls.SPACES[position % 40]

    @classmethod
    def get_property_group(cls, color: PropertyColor) -> tuple[int, ...]:
        """Get all positions for a property color group."""
        return PROPERTY_GROUPS.get(color, ())

    @classmethod
    def get_group_size(cls, color: PropertyColor) -> int:
        """Get the number of properties in a color group."""
        return len(PROPERTY_GROUPS.get(color, ()))

    @classmethod
    def is_property(cls, position: int) -> bool:
        """Check if position is a regular property."""
        return position in cls._property_positions

    @classmethod
    def is_railroad(cls, position: int) -> bool:
        """Check if position is a railroad."""
        return position in cls._railroad_positions

    @classmethod
    def is_utility(cls, position: int) -> bool:
        """Check if position is a utility."""
        return position in cls._utility_positions

    @classmethod
    def is_buyable(cls, position: int) -> bool:
        """Check if the space at this position can be purchased."""
        space = cls.get_space(position)
        return isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace))

    @classmethod
    def get_buyable_positions(cls) -> tuple[int, ...]:
        """Get all positions that can be purchased."""
        return tuple(
            i
            for i, s in enumerate(cls.SPACES)
            if isinstance(s, (PropertySpace, RailroadSpace, UtilitySpace))
        )

    @classmethod
    def get_nearest_railroad(cls, position: int) -> int:
        """Get the nearest railroad position (moving forward)."""
        for rr in cls._railroad_positions:
            if rr > position:
                return rr
        return cls._railroad_positions[0]  # Wrap around to Reading Railroad

    @classmethod
    def get_nearest_utility(cls, position: int) -> int:
        """Get the nearest utility position (moving forward)."""
        for util in cls._utility_positions:
            if util > position:
                return util
        return cls._utility_positions[0]  # Wrap around to Electric Company

    @classmethod
    def calculate_distance(cls, from_pos: int, to_pos: int) -> int:
        """Calculate spaces between two positions (moving forward)."""
        if to_pos >= from_pos:
            return to_pos - from_pos
        return 40 - from_pos + to_pos

    @classmethod
    def get_all_spaces(cls) -> tuple[Space, ...]:
        """Get all spaces on the board."""
        return cls.SPACES

    @classmethod
    def to_dict(cls) -> dict[str, object]:
        """Serialize the board to a JSON-compatible dict."""
        return {
            "spaces": [space.to_dict() for space in cls.SPACES],
            "property_groups": {
                color.name: list(positions)
                for color, positions in PROPERTY_GROUPS.items()
            },
        }
