"""Monopoly Game Engine - A pure, framework-agnostic Monopoly implementation.

This package provides a complete Monopoly game engine with:
- Full game rules implementation
- Type-safe API with mypy strict mode compatibility
- JSON serialization for web transport
- Deterministic gameplay with seed support

Modules:
    types: Type definitions, enums, and TypedDicts
    board: Board layout and space definitions
    property: Property state management
    player: Player state management
    cards: Chance and Community Chest cards
    rules: Game rules (rent, building, etc.)
    actions: Player actions and validation
    state: Game state container with serialization
    exceptions: Custom exception types
"""

from .actions import (
    AcceptTrade,
    Action,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    DeclareBankruptcy,
    EndTurn,
    MortgageProperty,
    PayJailFine,
    ProposeTrade,
    RejectTrade,
    RollDice,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)
from .board import (
    Board,
    PropertySpace,
    RailroadSpace,
    Space,
    TaxSpace,
    UtilitySpace,
)
from .cards import (
    CHANCE_CARDS,
    COMMUNITY_CHEST_CARDS,
    Card,
    CardDeck,
    get_chance_card,
    get_community_chest_card,
)
from .exceptions import (
    BankruptcyError,
    CannotBuildError,
    CannotMortgageError,
    GameOverError,
    InsufficientFundsError,
    InvalidActionError,
    InvalidPlayerError,
    InvalidPropertyError,
    MonopolyError,
    NotYourTurnError,
    PropertyAlreadyOwnedError,
    PropertyNotOwnedError,
)
from .game import (
    MonopolyGame,
)
from .player import (
    Player,
)
from .property import (
    Property,
    PropertyManager,
)
from .rules import (
    calculate_net_worth,
    calculate_rent,
    can_afford_rent,
    can_build_house,
    can_buy_property,
    can_mortgage_property,
    can_sell_house,
    can_unmortgage_property,
    get_buildable_properties,
    get_building_cost,
    get_house_sale_value,
    get_mortgage_value,
    get_mortgageable_properties,
    get_property_cost,
    get_sellable_houses,
    get_unmortgage_cost,
    get_unmortgageable_properties,
    is_bankrupt,
    validate_trade,
)
from .state import (
    GameState,
)
from .types import (
    BOARD_SIZE,
    GO_SALARY,
    GO_TO_JAIL_POSITION,
    INCOME_TAX_AMOUNT,
    JAIL_FINE,
    JAIL_POSITION,
    LUXURY_TAX_AMOUNT,
    MAX_JAIL_TURNS,
    POSITION_TO_COLOR,
    PROPERTY_GROUPS,
    STARTING_MONEY,
    TOTAL_HOTELS,
    TOTAL_HOUSES,
    ActionType,
    CardData,
    CardType,
    GamePhase,
    GameStateData,
    PlayerStateData,
    PropertyColor,
    PropertyData,
    PropertyStateData,
    RailroadData,
    SpaceType,
    TradeOfferData,
    UtilityData,
)

__version__ = "0.1.0"
__all__ = [
    # Types
    "SpaceType",
    "PropertyColor",
    "ActionType",
    "CardType",
    "GamePhase",
    "PropertyData",
    "RailroadData",
    "UtilityData",
    "PropertyStateData",
    "PlayerStateData",
    "GameStateData",
    "CardData",
    "TradeOfferData",
    # Constants
    "BOARD_SIZE",
    "STARTING_MONEY",
    "GO_SALARY",
    "JAIL_POSITION",
    "GO_TO_JAIL_POSITION",
    "TOTAL_HOUSES",
    "TOTAL_HOTELS",
    "JAIL_FINE",
    "MAX_JAIL_TURNS",
    "INCOME_TAX_AMOUNT",
    "LUXURY_TAX_AMOUNT",
    "PROPERTY_GROUPS",
    "POSITION_TO_COLOR",
    # Board
    "Space",
    "PropertySpace",
    "RailroadSpace",
    "UtilitySpace",
    "TaxSpace",
    "Board",
    # Property
    "Property",
    "PropertyManager",
    # Player
    "Player",
    # Cards
    "Card",
    "CardDeck",
    "CHANCE_CARDS",
    "COMMUNITY_CHEST_CARDS",
    "get_chance_card",
    "get_community_chest_card",
    # Rules
    "calculate_rent",
    "can_buy_property",
    "get_property_cost",
    "can_build_house",
    "get_building_cost",
    "can_sell_house",
    "get_house_sale_value",
    "can_mortgage_property",
    "get_mortgage_value",
    "can_unmortgage_property",
    "get_unmortgage_cost",
    "calculate_net_worth",
    "can_afford_rent",
    "is_bankrupt",
    "get_buildable_properties",
    "get_sellable_houses",
    "get_mortgageable_properties",
    "get_unmortgageable_properties",
    "validate_trade",
    # Exceptions
    "MonopolyError",
    "InvalidActionError",
    "InsufficientFundsError",
    "GameOverError",
    "InvalidPlayerError",
    "InvalidPropertyError",
    "NotYourTurnError",
    "PropertyNotOwnedError",
    "PropertyAlreadyOwnedError",
    "CannotBuildError",
    "CannotMortgageError",
    "BankruptcyError",
    # Actions
    "Action",
    "RollDice",
    "BuyProperty",
    "BuildHouse",
    "BuildHotel",
    "SellHouse",
    "SellHotel",
    "MortgageProperty",
    "UnmortgageProperty",
    "ProposeTrade",
    "AcceptTrade",
    "RejectTrade",
    "PayJailFine",
    "UseJailCard",
    "DeclareBankruptcy",
    "EndTurn",
    # State
    "GameState",
    # Game
    "MonopolyGame",
]
