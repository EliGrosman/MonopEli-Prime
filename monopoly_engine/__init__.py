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
    exceptions: Custom exception types
"""

from .types import (
    SpaceType,
    PropertyColor,
    ActionType,
    CardType,
    GamePhase,
    PropertyData,
    RailroadData,
    UtilityData,
    PropertyStateData,
    PlayerStateData,
    GameStateData,
    CardData,
    TradeOfferData,
    BOARD_SIZE,
    STARTING_MONEY,
    GO_SALARY,
    JAIL_POSITION,
    GO_TO_JAIL_POSITION,
    TOTAL_HOUSES,
    TOTAL_HOTELS,
    JAIL_FINE,
    MAX_JAIL_TURNS,
    INCOME_TAX_AMOUNT,
    LUXURY_TAX_AMOUNT,
    PROPERTY_GROUPS,
    POSITION_TO_COLOR,
)

from .board import (
    Space,
    PropertySpace,
    RailroadSpace,
    UtilitySpace,
    TaxSpace,
    Board,
)

from .property import (
    Property,
    PropertyManager,
)

from .player import (
    Player,
)

from .cards import (
    Card,
    CardDeck,
    CHANCE_CARDS,
    COMMUNITY_CHEST_CARDS,
    get_chance_card,
    get_community_chest_card,
)

from .rules import (
    calculate_rent,
    can_buy_property,
    get_property_cost,
    can_build_house,
    get_building_cost,
    can_sell_house,
    get_house_sale_value,
    can_mortgage_property,
    get_mortgage_value,
    can_unmortgage_property,
    get_unmortgage_cost,
    calculate_net_worth,
    can_afford_rent,
    is_bankrupt,
    get_buildable_properties,
    get_sellable_houses,
    get_mortgageable_properties,
    get_unmortgageable_properties,
    validate_trade,
)

from .exceptions import (
    MonopolyError,
    InvalidActionError,
    InsufficientFundsError,
    GameOverError,
    InvalidPlayerError,
    InvalidPropertyError,
    NotYourTurnError,
    PropertyNotOwnedError,
    PropertyAlreadyOwnedError,
    CannotBuildError,
    CannotMortgageError,
    BankruptcyError,
)

from .actions import (
    Action,
    RollDice,
    BuyProperty,
    BuildHouse,
    BuildHotel,
    SellHouse,
    SellHotel,
    MortgageProperty,
    UnmortgageProperty,
    ProposeTrade,
    AcceptTrade,
    RejectTrade,
    PayJailFine,
    UseJailCard,
    DeclareBankruptcy,
    EndTurn,
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
]
