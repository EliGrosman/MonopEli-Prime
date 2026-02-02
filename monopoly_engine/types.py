"""Type definitions, enums, and protocols for the Monopoly game engine.

This module provides all type definitions used throughout the game engine,
including enums for categorizing game elements, TypedDicts for structured
data, and protocols for duck typing.

All types are designed to be JSON-serializable and compatible with mypy
strict mode type checking.
"""

from enum import Enum, auto
from typing import Optional, Protocol, TypedDict


# =============================================================================
# Enums
# =============================================================================


class SpaceType(Enum):
    """Types of spaces on the Monopoly board.

    Attributes:
        GO: The starting space, awards $200 when passed.
        PROPERTY: A purchasable property with a color group.
        RAILROAD: A purchasable railroad (4 total on board).
        UTILITY: A purchasable utility (2 total on board).
        CHANCE: Draw a Chance card.
        COMMUNITY_CHEST: Draw a Community Chest card.
        TAX: Pay a fixed tax amount.
        JAIL: Just visiting or in jail (position 10).
        GO_TO_JAIL: Sends player directly to jail (position 30).
        FREE_PARKING: Free space with no effect (position 20).
    """

    GO = auto()
    PROPERTY = auto()
    RAILROAD = auto()
    UTILITY = auto()
    CHANCE = auto()
    COMMUNITY_CHEST = auto()
    TAX = auto()
    JAIL = auto()
    GO_TO_JAIL = auto()
    FREE_PARKING = auto()


class PropertyColor(Enum):
    """Color groups for properties on the Monopoly board.

    Owning all properties in a color group constitutes a monopoly,
    which doubles base rent and allows building houses/hotels.

    Attributes:
        BROWN: Mediterranean Ave, Baltic Ave (2 properties).
        LIGHT_BLUE: Oriental Ave, Vermont Ave, Connecticut Ave (3 properties).
        MAGENTA: St. Charles Place, States Ave, Virginia Ave (3 properties).
        ORANGE: St. James Place, Tennessee Ave, New York Ave (3 properties).
        RED: Kentucky Ave, Indiana Ave, Illinois Ave (3 properties).
        YELLOW: Atlantic Ave, Ventnor Ave, Marvin Gardens (3 properties).
        GREEN: Pacific Ave, N. Carolina Ave, Pennsylvania Ave (3 properties).
        BLUE: Park Place, Boardwalk (2 properties).
        RAILROAD: Reading, Pennsylvania, B&O, Short Line (4 railroads).
        UTILITY: Electric Company, Water Works (2 utilities).
    """

    BROWN = auto()
    LIGHT_BLUE = auto()
    MAGENTA = auto()
    ORANGE = auto()
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    BLUE = auto()
    RAILROAD = auto()
    UTILITY = auto()


class ActionType(Enum):
    """Types of actions a player can take during the game.

    Attributes:
        ROLL_DICE: Roll dice to move (required at start of turn).
        BUY_PROPERTY: Purchase an unowned property after landing on it.
        BUILD_HOUSE: Build a house on a property (requires monopoly).
        BUILD_HOTEL: Build a hotel on a property with 4 houses.
        SELL_HOUSE: Sell a house back to the bank.
        SELL_HOTEL: Sell a hotel back to the bank (returns 4 houses).
        MORTGAGE_PROPERTY: Mortgage a property for cash.
        UNMORTGAGE_PROPERTY: Unmortgage a property (costs 110% of mortgage value).
        PROPOSE_TRADE: Propose a trade with another player.
        ACCEPT_TRADE: Accept a pending trade proposal.
        REJECT_TRADE: Reject a pending trade proposal.
        PAY_JAIL_FINE: Pay $50 to get out of jail.
        USE_JAIL_CARD: Use a Get Out of Jail Free card.
        DECLARE_BANKRUPTCY: Declare bankruptcy when unable to pay debts.
        END_TURN: End the current turn.
    """

    ROLL_DICE = auto()
    BUY_PROPERTY = auto()
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
    DECLARE_BANKRUPTCY = auto()
    END_TURN = auto()


class CardType(Enum):
    """Types of Chance and Community Chest cards.

    Attributes:
        MOVE: Move to a specific position on the board.
        MONEY: Receive or pay a fixed amount of money.
        GET_OUT_OF_JAIL: Receive a Get Out of Jail Free card.
        PAY_PER_HOUSE: Pay an amount per house/hotel owned.
        GO_TO_JAIL: Go directly to jail.
    """

    MOVE = auto()
    MONEY = auto()
    GET_OUT_OF_JAIL = auto()
    PAY_PER_HOUSE = auto()
    GO_TO_JAIL = auto()


# =============================================================================
# TypedDicts for Structured Data
# =============================================================================


class PropertyData(TypedDict):
    """Static data for a purchasable property.

    This represents the immutable characteristics of a property as defined
    by the board layout. Does not include mutable state like owner or houses.

    Attributes:
        position: Board position (0-39).
        name: Display name of the property.
        color: Color group the property belongs to.
        cost: Purchase price.
        rent: List of rent values [base, 1h, 2h, 3h, 4h, hotel].
              For railroads/utilities, may have fewer elements.
        house_cost: Cost to build one house (0 for railroads/utilities).
        mortgage_value: Amount received when mortgaging.
    """

    position: int
    name: str
    color: PropertyColor
    cost: int
    rent: list[int]
    house_cost: int
    mortgage_value: int


class PlayerState(TypedDict):
    """Complete state of a single player.

    This represents all mutable data associated with a player during
    the game. Can be serialized to/from JSON for persistence.

    Attributes:
        id: Unique player identifier (0-7).
        name: Player's display name.
        money: Current cash balance.
        position: Current board position (0-39, or 40 for "in jail").
        properties: List of owned property positions.
        houses: Dict mapping property position to house count (5 = hotel).
        jail_cards: Number of Get Out of Jail Free cards held.
        in_jail: Whether the player is currently in jail.
        jail_turns: Number of turns spent in jail (max 3).
        bankrupt: Whether the player has declared bankruptcy.
    """

    id: int
    name: str
    money: int
    position: int
    properties: list[int]
    houses: dict[int, int]
    jail_cards: int
    in_jail: bool
    jail_turns: int
    bankrupt: bool


class GameState(TypedDict):
    """Complete state of the Monopoly game.

    This represents all information needed to fully reconstruct or
    continue a game. Can be serialized to/from JSON.

    Attributes:
        players: List of all player states.
        current_player: Index of the player whose turn it is.
        turn_number: Total number of turns elapsed.
        houses_remaining: Number of houses left in the bank (max 32).
        hotels_remaining: Number of hotels left in the bank (max 12).
        chance_deck: Ordered list of Chance card IDs (shuffled).
        community_chest_deck: Ordered list of Community Chest card IDs.
        last_roll: Last dice roll as (die1, die2), or None.
        doubles_count: Consecutive doubles rolled this turn (max 3).
        game_over: Whether the game has ended.
        winner: Player ID of the winner, or None if game ongoing.
    """

    players: list[PlayerState]
    current_player: int
    turn_number: int
    houses_remaining: int
    hotels_remaining: int
    chance_deck: list[int]
    community_chest_deck: list[int]
    last_roll: Optional[tuple[int, int]]
    doubles_count: int
    game_over: bool
    winner: Optional[int]


class TradeOffer(TypedDict):
    """Structured data for a trade proposal between players.

    Attributes:
        from_player: ID of the player proposing the trade.
        to_player: ID of the player receiving the trade offer.
        from_properties: List of property positions offered by proposer.
        to_properties: List of property positions requested from recipient.
        from_money: Amount of money offered by proposer.
        to_money: Amount of money requested from recipient.
    """

    from_player: int
    to_player: int
    from_properties: list[int]
    to_properties: list[int]
    from_money: int
    to_money: int


class CardData(TypedDict):
    """Static data for a Chance or Community Chest card.

    Attributes:
        id: Unique card identifier.
        deck: Which deck the card belongs to ("chance" or "community_chest").
        type: The type of effect the card has.
        description: Human-readable card text.
        value: Numeric value (amount, position, or cost multiplier).
        keep: Whether this card can be kept (Get Out of Jail Free).
    """

    id: int
    deck: str
    type: CardType
    description: str
    value: int
    keep: bool


class SpaceData(TypedDict):
    """Static data for a board space.

    Attributes:
        position: Board position (0-39).
        name: Display name of the space.
        type: The type of space.
        value: Associated value (tax amount, 0 for special spaces).
    """

    position: int
    name: str
    type: SpaceType
    value: int


# =============================================================================
# Protocols for Duck Typing
# =============================================================================


class Serializable(Protocol):
    """Protocol for objects that can be serialized to dictionaries.

    Any class implementing this protocol can be converted to/from
    JSON-compatible dictionaries for persistence or network transport.
    """

    def to_dict(self) -> dict[str, object]:
        """Convert the object to a JSON-compatible dictionary.

        Returns:
            A dictionary representation of the object's state.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Serializable":
        """Reconstruct an object from a dictionary.

        Args:
            data: Dictionary representation of the object.

        Returns:
            A new instance of the object with state from the dictionary.
        """
        ...


class Validateable(Protocol):
    """Protocol for actions that can be validated before execution.

    Any action class implementing this protocol must provide a validate
    method that checks if the action can be legally performed.
    """

    def validate(self, game_state: GameState) -> tuple[bool, str]:
        """Check if the action can be legally performed.

        Args:
            game_state: Current state of the game.

        Returns:
            A tuple of (is_valid, error_message). If valid, error_message
            is an empty string.
        """
        ...


class Executable(Protocol):
    """Protocol for actions that can be executed to modify game state.

    Any action class implementing this protocol must provide an execute
    method that performs the action and returns the result.
    """

    def execute(self, game_state: GameState) -> tuple[bool, str]:
        """Execute the action and modify the game state.

        This method assumes validation has already been performed.
        It should modify the game_state in place.

        Args:
            game_state: Current state of the game to be modified.

        Returns:
            A tuple of (success, message). If successful, message describes
            what happened. If failed, message explains why.
        """
        ...
