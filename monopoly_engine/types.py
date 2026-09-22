"""Type definitions for the Monopoly game engine.

This module contains all enums, TypedDicts, and type aliases used throughout
the game engine. No dependencies on other engine modules.
"""

from enum import Enum, auto
from typing import NotRequired, TypedDict


class SpaceType(Enum):
    """Types of spaces on the Monopoly board."""

    GO = auto()
    PROPERTY = auto()
    RAILROAD = auto()
    UTILITY = auto()
    CHANCE = auto()
    COMMUNITY_CHEST = auto()
    INCOME_TAX = auto()
    LUXURY_TAX = auto()
    JAIL = auto()
    GO_TO_JAIL = auto()
    FREE_PARKING = auto()


class PropertyColor(Enum):
    """Property color groups."""

    BROWN = auto()
    LIGHT_BLUE = auto()
    MAGENTA = auto()
    ORANGE = auto()
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    DARK_BLUE = auto()
    RAILROAD = auto()
    UTILITY = auto()


class ActionType(Enum):
    """Types of actions a player can take."""

    ROLL_DICE = auto()
    BUY_PROPERTY = auto()
    AUCTION_PROPERTY = auto()
    BUILD_HOUSE = auto()
    BUILD_HOTEL = auto()
    SELL_HOUSE = auto()
    SELL_HOTEL = auto()
    MORTGAGE_PROPERTY = auto()
    UNMORTGAGE_PROPERTY = auto()
    PROPOSE_TRADE = auto()
    ACCEPT_TRADE = auto()
    REJECT_TRADE = auto()
    PAY_JAIL_FINE = auto()
    USE_JAIL_CARD = auto()
    ATTEMPT_JAIL_ROLL = auto()
    DECLARE_BANKRUPTCY = auto()
    END_TURN = auto()


class CardType(Enum):
    """Types of Chance and Community Chest cards."""

    MOVE = auto()  # Move to a specific position
    MOVE_NEAREST = auto()  # Move to nearest railroad/utility
    MOVE_BACK = auto()  # Move back N spaces
    COLLECT = auto()  # Collect money from bank
    PAY = auto()  # Pay money to bank
    PAY_PER_BUILDING = auto()  # Pay per house/hotel
    COLLECT_FROM_PLAYERS = auto()  # Collect from each player
    PAY_TO_PLAYERS = auto()  # Pay to each player
    GET_OUT_OF_JAIL = auto()  # Get out of jail free card
    GO_TO_JAIL = auto()  # Go directly to jail


class GamePhase(Enum):
    """Current phase of the game turn."""

    WAITING_FOR_ROLL = auto()
    ROLLED = auto()
    LANDED = auto()
    PURCHASE_DECISION = auto()
    AUCTION = auto()
    PAYING_RENT = auto()
    DRAWING_CARD = auto()
    IN_JAIL = auto()
    BANKRUPT = auto()
    GAME_OVER = auto()


# TypedDicts for structured data serialization


class PropertyData(TypedDict):
    """Static property definition data."""

    position: int
    name: str
    color: str  # PropertyColor value name
    cost: int
    rent: list[int]  # [base, 1h, 2h, 3h, 4h, hotel]
    house_cost: int
    mortgage_value: int


class RailroadData(TypedDict):
    """Static railroad definition data."""

    position: int
    name: str
    cost: int
    rent: list[int]  # [1 owned, 2 owned, 3 owned, 4 owned]
    mortgage_value: int


class UtilityData(TypedDict):
    """Static utility definition data."""

    position: int
    name: str
    cost: int
    mortgage_value: int


class PropertyStateData(TypedDict):
    """Mutable property state for serialization."""

    position: int
    owner: int | None
    houses: int  # 0-4 for houses, 5 for hotel
    mortgaged: bool


class PlayerStateData(TypedDict):
    """Player state for serialization."""

    id: int
    name: str
    money: int
    position: int
    jail_cards: int
    in_jail: bool
    jail_turns: int
    bankrupt: bool


class GameStateData(TypedDict):
    """Complete game state for serialization."""

    players: list[PlayerStateData]
    properties: dict[str, PropertyStateData]  # position -> state
    current_player: int
    turn_number: int
    phase: str  # GamePhase value name
    houses_remaining: int
    hotels_remaining: int
    chance_deck: list[int]  # Card IDs in draw order
    chest_deck: list[int]  # Card IDs in draw order
    last_roll: tuple[int, int] | None
    doubles_count: int
    game_over: bool
    winner: int | None


class CardData(TypedDict, total=False):
    """Card definition data for serialization."""

    id: int
    text: str
    card_type: str  # CardType value name
    # Optional fields based on card type
    move_to: int
    move_to_type: str  # "railroad" or "utility" for nearest
    move_spaces: int  # For move back cards
    amount: int  # Money amount
    per_house: int  # For pay per building
    per_hotel: int


class TradeOfferData(TypedDict):
    """Trade offer data for serialization."""

    from_player: int
    to_player: int
    give_properties: list[int]  # Property positions
    give_money: int
    want_properties: list[int]  # Property positions
    want_money: int
    trade_id: NotRequired[int]
    created_revision: NotRequired[int]


# Constants
BOARD_SIZE: int = 40
STARTING_MONEY: int = 1500
GO_SALARY: int = 200
JAIL_POSITION: int = 10
GO_TO_JAIL_POSITION: int = 30
TOTAL_HOUSES: int = 32
TOTAL_HOTELS: int = 12
JAIL_FINE: int = 50
MAX_JAIL_TURNS: int = 3
INCOME_TAX_AMOUNT: int = 200
LUXURY_TAX_AMOUNT: int = 100

# Property color groups - which positions belong to each color
PROPERTY_GROUPS: dict[PropertyColor, tuple[int, ...]] = {
    PropertyColor.BROWN: (1, 3),
    PropertyColor.LIGHT_BLUE: (6, 8, 9),
    PropertyColor.MAGENTA: (11, 13, 14),
    PropertyColor.ORANGE: (16, 18, 19),
    PropertyColor.RED: (21, 23, 24),
    PropertyColor.YELLOW: (26, 27, 29),
    PropertyColor.GREEN: (31, 32, 34),
    PropertyColor.DARK_BLUE: (37, 39),
    PropertyColor.RAILROAD: (5, 15, 25, 35),
    PropertyColor.UTILITY: (12, 28),
}

# Reverse mapping: position -> color
POSITION_TO_COLOR: dict[int, PropertyColor] = {}
for color, positions in PROPERTY_GROUPS.items():
    for pos in positions:
        POSITION_TO_COLOR[pos] = color
