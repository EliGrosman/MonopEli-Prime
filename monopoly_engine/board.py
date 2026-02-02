"""Board definition for the Monopoly game engine.

This module defines the immutable board layout for Monopoly, including all 40
spaces and their properties. The board is represented as a tuple of frozen
dataclasses to ensure immutability and type safety.

The standard Monopoly board has:
- 28 properties (22 streets, 4 railroads, 2 utilities)
- 3 Chance spaces
- 3 Community Chest spaces
- 2 Tax spaces (Income Tax, Luxury Tax)
- 4 corner spaces (GO, Jail/Just Visiting, Free Parking, Go to Jail)
"""

from dataclasses import dataclass
from typing import ClassVar

from .types import SpaceType, PropertyColor


@dataclass(frozen=True)
class Space:
    """Base class for a space on the Monopoly board.

    All spaces are immutable to prevent accidental modification of the board
    definition. This is the base class for all space types.

    Attributes:
        position: The position on the board (0-39)
        name: The display name of the space
        type: The type of space (GO, PROPERTY, CHANCE, etc.)
    """

    position: int
    name: str
    type: SpaceType


@dataclass(frozen=True)
class PropertySpace(Space):
    """A purchasable street property with houses/hotels.

    PropertySpace represents one of the 22 colored street properties on the
    board. These properties can have houses and hotels built on them when the
    player owns a complete color group (monopoly).

    Attributes:
        position: The position on the board (0-39)
        name: The display name of the property
        type: Always SpaceType.PROPERTY
        color: The color group this property belongs to
        cost: Purchase price of the property
        rent: Tuple of rent values (base, 1H, 2H, 3H, 4H, hotel)
        house_cost: Cost to build one house on this property
        mortgage_value: Amount received when mortgaging the property
    """

    color: PropertyColor
    cost: int
    rent: tuple[int, int, int, int, int, int]  # Base, +1H, +2H, +3H, +4H, Hotel
    house_cost: int
    mortgage_value: int


@dataclass(frozen=True)
class RailroadSpace(Space):
    """A railroad property.

    Railroads are special properties whose rent depends on how many railroads
    the owner controls (1, 2, 3, or all 4).

    Attributes:
        position: The position on the board (0-39)
        name: The display name of the railroad
        type: Always SpaceType.RAILROAD
        cost: Purchase price of the railroad
        rent: Tuple of rent values for owning 1, 2, 3, or 4 railroads
        mortgage_value: Amount received when mortgaging the railroad
    """

    cost: int
    rent: tuple[int, int, int, int]  # 1, 2, 3, 4 owned
    mortgage_value: int


@dataclass(frozen=True)
class UtilitySpace(Space):
    """A utility property (Electric Company or Water Works).

    Utilities have variable rent based on the dice roll and number of utilities
    owned. Rent is calculated as dice_roll * multiplier, where multiplier is
    4x if one utility is owned, or 10x if both are owned.

    Attributes:
        position: The position on the board (0-39)
        name: The display name of the utility
        type: Always SpaceType.UTILITY
        cost: Purchase price of the utility
        mortgage_value: Amount received when mortgaging the utility
    """

    cost: int
    mortgage_value: int


@dataclass(frozen=True)
class TaxSpace(Space):
    """A tax space (Income Tax or Luxury Tax).

    Tax spaces require the player to pay a fixed amount to the bank when
    landing on them.

    Attributes:
        position: The position on the board (0-39)
        name: The display name of the tax space
        type: Always SpaceType.TAX
        amount: The tax amount to be paid
    """

    amount: int


class Board:
    """The Monopoly board definition.

    This class contains the complete board layout as a class variable. The
    board is immutable and shared across all game instances. It provides
    utility methods for accessing spaces and querying property groups.

    The board follows the standard US Monopoly layout with 40 spaces numbered
    0-39, starting from GO and proceeding clockwise around the board.
    """

    SPACES: ClassVar[tuple[Space, ...]] = (
        # Position 0: GO
        Space(0, "GO", SpaceType.GO),
        # Position 1: Mediterranean Avenue (Brown)
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
        # Position 2: Community Chest
        Space(2, "Community Chest", SpaceType.COMMUNITY_CHEST),
        # Position 3: Baltic Avenue (Brown)
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
        # Position 4: Income Tax
        TaxSpace(4, "Income Tax", SpaceType.TAX, 200),
        # Position 5: Reading Railroad
        RailroadSpace(
            5, "Reading Railroad", SpaceType.RAILROAD, 200, (25, 50, 100, 200), 100
        ),
        # Position 6: Oriental Avenue (Light Blue)
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
        # Position 7: Chance
        Space(7, "Chance", SpaceType.CHANCE),
        # Position 8: Vermont Avenue (Light Blue)
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
        # Position 9: Connecticut Avenue (Light Blue)
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
        # Position 10: Jail (Just Visiting)
        Space(10, "Just Visiting", SpaceType.JAIL),
        # Position 11: St. Charles Place (Magenta)
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
        # Position 12: Electric Company
        UtilitySpace(12, "Electric Company", SpaceType.UTILITY, 150, 75),
        # Position 13: States Avenue (Magenta)
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
        # Position 14: Virginia Avenue (Magenta)
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
        # Position 15: Pennsylvania Railroad
        RailroadSpace(
            15,
            "Pennsylvania Railroad",
            SpaceType.RAILROAD,
            200,
            (25, 50, 100, 200),
            100,
        ),
        # Position 16: St. James Place (Orange)
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
        # Position 17: Community Chest
        Space(17, "Community Chest", SpaceType.COMMUNITY_CHEST),
        # Position 18: Tennessee Avenue (Orange)
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
        # Position 19: New York Avenue (Orange)
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
        # Position 20: Free Parking
        Space(20, "Free Parking", SpaceType.FREE_PARKING),
        # Position 21: Kentucky Avenue (Red)
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
        # Position 22: Chance
        Space(22, "Chance", SpaceType.CHANCE),
        # Position 23: Indiana Avenue (Red)
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
        # Position 24: Illinois Avenue (Red)
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
        # Position 25: B & O Railroad
        RailroadSpace(
            25, "B & O Railroad", SpaceType.RAILROAD, 200, (25, 50, 100, 200), 100
        ),
        # Position 26: Atlantic Avenue (Yellow)
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
        # Position 27: Ventnor Avenue (Yellow)
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
        # Position 28: Water Works
        UtilitySpace(28, "Water Works", SpaceType.UTILITY, 150, 75),
        # Position 29: Marvin Gardens (Yellow)
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
        # Position 30: Go to Jail
        Space(30, "Go to Jail", SpaceType.GO_TO_JAIL),
        # Position 31: Pacific Avenue (Green)
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
        # Position 32: North Carolina Avenue (Green)
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
        # Position 33: Community Chest
        Space(33, "Community Chest", SpaceType.COMMUNITY_CHEST),
        # Position 34: Pennsylvania Avenue (Green)
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
        # Position 35: Short Line Railroad
        RailroadSpace(
            35, "Short Line", SpaceType.RAILROAD, 200, (25, 50, 100, 200), 100
        ),
        # Position 36: Chance
        Space(36, "Chance", SpaceType.CHANCE),
        # Position 37: Park Place (Blue)
        PropertySpace(
            37,
            "Park Place",
            SpaceType.PROPERTY,
            PropertyColor.BLUE,
            350,
            (35, 175, 500, 1100, 1300, 1500),
            200,
            175,
        ),
        # Position 38: Luxury Tax
        TaxSpace(38, "Luxury Tax", SpaceType.TAX, 100),
        # Position 39: Boardwalk (Blue)
        PropertySpace(
            39,
            "Boardwalk",
            SpaceType.PROPERTY,
            PropertyColor.BLUE,
            400,
            (50, 200, 600, 1400, 1700, 2000),
            200,
            200,
        ),
    )

    @classmethod
    def get_space(cls, position: int) -> Space:
        """Get a space by its position on the board.

        The position wraps around the board, so position 40 is the same as
        position 0 (GO), position 41 is the same as position 1, etc.

        Args:
            position: The position on the board (can be >= 40)

        Returns:
            The Space object at that position

        Examples:
            >>> Board.get_space(0)
            Space(position=0, name='GO', type=SpaceType.GO)
            >>> Board.get_space(40)  # Wraps to GO
            Space(position=0, name='GO', type=SpaceType.GO)
        """
        return cls.SPACES[position % 40]

    @classmethod
    def get_property_group(cls, color: PropertyColor) -> list[int]:
        """Get all positions for properties in a color group.

        This is useful for checking monopolies and enforcing even building
        rules. Properties must be owned as a complete group before houses
        can be built, and houses must be built evenly across the group.

        Args:
            color: The property color group to query

        Returns:
            A list of positions (integers) for all properties in that color
            group, sorted by position

        Examples:
            >>> Board.get_property_group(PropertyColor.BROWN)
            [1, 3]
            >>> Board.get_property_group(PropertyColor.BLUE)
            [37, 39]
            >>> Board.get_property_group(PropertyColor.RAILROAD)
            [5, 15, 25, 35]
        """
        positions = [
            i
            for i, space in enumerate(cls.SPACES)
            if isinstance(space, PropertySpace) and space.color == color
        ]
        return positions

    @classmethod
    def get_railroads(cls) -> list[int]:
        """Get all railroad positions.

        Returns:
            A list of positions for all four railroads

        Examples:
            >>> Board.get_railroads()
            [5, 15, 25, 35]
        """
        return [i for i, space in enumerate(cls.SPACES) if isinstance(space, RailroadSpace)]

    @classmethod
    def get_utilities(cls) -> list[int]:
        """Get all utility positions.

        Returns:
            A list of positions for both utilities

        Examples:
            >>> Board.get_utilities()
            [12, 28]
        """
        return [i for i, space in enumerate(cls.SPACES) if isinstance(space, UtilitySpace)]
