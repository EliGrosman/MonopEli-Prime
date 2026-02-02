"""
Pytest configuration and shared fixtures for Monopoly game engine tests.

This module provides common fixtures used across test files, including
property managers, board spaces, and test data.
"""

import pytest
from monopoly_engine.types import (
    PropertyColor,
    SpaceType,
    ActionType,
    CardType,
    PropertyData,
    PlayerState,
    GameState,
    TradeOffer,
    CardData,
    SpaceData,
)
from monopoly_engine.board import Board, PropertySpace, RailroadSpace, UtilitySpace
from monopoly_engine.property import Property, PropertyManager


@pytest.fixture
def property_manager() -> PropertyManager:
    """Create a fresh PropertyManager instance for testing.

    Returns:
        A new PropertyManager with all 28 properties initialized
    """
    return PropertyManager()


@pytest.fixture
def property_manager_with_brown_monopoly() -> PropertyManager:
    """Create a PropertyManager where player 0 owns the brown monopoly.

    Returns:
        PropertyManager with positions 1 and 3 owned by player 0
    """
    manager = PropertyManager()
    manager.properties[1].owner = 0
    manager.properties[3].owner = 0
    return manager


@pytest.fixture
def property_manager_with_railroad_monopoly() -> PropertyManager:
    """Create a PropertyManager where player 0 owns all railroads.

    Returns:
        PropertyManager with all 4 railroads owned by player 0
    """
    manager = PropertyManager()
    for pos in [5, 15, 25, 35]:
        manager.properties[pos].owner = 0
    return manager


@pytest.fixture
def property_manager_with_utility_monopoly() -> PropertyManager:
    """Create a PropertyManager where player 0 owns both utilities.

    Returns:
        PropertyManager with both utilities owned by player 0
    """
    manager = PropertyManager()
    manager.properties[12].owner = 0  # Electric Company
    manager.properties[28].owner = 0  # Water Works
    return manager


@pytest.fixture
def mediterranean_property() -> Property:
    """Create a Property instance for Mediterranean Avenue (position 1).

    Returns:
        Property at position 1 (brown, cheapest property)
    """
    return Property(position=1)


@pytest.fixture
def boardwalk_property() -> Property:
    """Create a Property instance for Boardwalk (position 39).

    Returns:
        Property at position 39 (blue, most expensive property)
    """
    return Property(position=39)


@pytest.fixture
def owned_property() -> Property:
    """Create a Property that is owned by player 0.

    Returns:
        Property owned by player 0 with no houses
    """
    return Property(position=1, owner=0)


@pytest.fixture
def property_with_houses() -> Property:
    """Create a Property with 3 houses owned by player 0.

    Returns:
        Property owned by player 0 with 3 houses
    """
    return Property(position=1, owner=0, houses=3)


@pytest.fixture
def property_with_hotel() -> Property:
    """Create a Property with a hotel owned by player 0.

    Returns:
        Property owned by player 0 with hotel (houses=5)
    """
    return Property(position=1, owner=0, houses=5)


@pytest.fixture
def mortgaged_property() -> Property:
    """Create a Property that is mortgaged.

    Returns:
        Property owned by player 0 that is mortgaged
    """
    return Property(position=1, owner=0, mortgaged=True)


@pytest.fixture
def sample_property_data() -> PropertyData:
    """Create sample PropertyData for testing serialization.

    Returns:
        PropertyData dict representing Mediterranean Avenue
    """
    return PropertyData(
        position=1,
        name="Mediterranean Avenue",
        color=PropertyColor.BROWN,
        cost=60,
        rent=[2, 10, 30, 90, 160, 250],
        house_cost=50,
        mortgage_value=30,
    )


@pytest.fixture
def sample_player_state() -> PlayerState:
    """Create a sample PlayerState for testing.

    Returns:
        PlayerState dict for player 0 with default values
    """
    return PlayerState(
        id=0,
        name="Player 1",
        money=1500,
        position=0,
        properties=[],
        houses={},
        jail_cards=0,
        in_jail=False,
        jail_turns=0,
        bankrupt=False,
    )


@pytest.fixture
def sample_game_state() -> GameState:
    """Create a sample GameState for testing.

    Returns:
        GameState dict for a 2-player game at start
    """
    return GameState(
        players=[
            PlayerState(
                id=0,
                name="Player 1",
                money=1500,
                position=0,
                properties=[],
                houses={},
                jail_cards=0,
                in_jail=False,
                jail_turns=0,
                bankrupt=False,
            ),
            PlayerState(
                id=1,
                name="Player 2",
                money=1500,
                position=0,
                properties=[],
                houses={},
                jail_cards=0,
                in_jail=False,
                jail_turns=0,
                bankrupt=False,
            ),
        ],
        current_player=0,
        turn_number=0,
        houses_remaining=32,
        hotels_remaining=12,
        chance_deck=[],
        community_chest_deck=[],
        last_roll=None,
        doubles_count=0,
        game_over=False,
        winner=None,
    )


@pytest.fixture
def sample_trade_offer() -> TradeOffer:
    """Create a sample TradeOffer for testing.

    Returns:
        TradeOffer dict between players 0 and 1
    """
    return TradeOffer(
        from_player=0,
        to_player=1,
        from_properties=[1],
        to_properties=[3],
        from_money=100,
        to_money=50,
    )


@pytest.fixture
def sample_card_data() -> CardData:
    """Create a sample CardData for testing.

    Returns:
        CardData dict for a Chance money card
    """
    return CardData(
        id=1,
        deck="chance",
        type=CardType.MONEY,
        description="Bank pays you dividend of $50",
        value=50,
        keep=False,
    )


@pytest.fixture
def sample_space_data() -> SpaceData:
    """Create a sample SpaceData for testing.

    Returns:
        SpaceData dict for Income Tax space
    """
    return SpaceData(
        position=4,
        name="Income Tax",
        type=SpaceType.TAX,
        value=200,
    )


@pytest.fixture
def all_property_colors() -> list[PropertyColor]:
    """Get a list of all property colors for comprehensive testing.

    Returns:
        List of all PropertyColor enum values
    """
    return [
        PropertyColor.BROWN,
        PropertyColor.LIGHT_BLUE,
        PropertyColor.MAGENTA,
        PropertyColor.ORANGE,
        PropertyColor.RED,
        PropertyColor.YELLOW,
        PropertyColor.GREEN,
        PropertyColor.BLUE,
        PropertyColor.RAILROAD,
        PropertyColor.UTILITY,
    ]


@pytest.fixture
def all_space_types() -> list[SpaceType]:
    """Get a list of all space types for comprehensive testing.

    Returns:
        List of all SpaceType enum values
    """
    return [
        SpaceType.GO,
        SpaceType.PROPERTY,
        SpaceType.RAILROAD,
        SpaceType.UTILITY,
        SpaceType.CHANCE,
        SpaceType.COMMUNITY_CHEST,
        SpaceType.TAX,
        SpaceType.JAIL,
        SpaceType.GO_TO_JAIL,
        SpaceType.FREE_PARKING,
    ]


@pytest.fixture
def all_action_types() -> list[ActionType]:
    """Get a list of all action types for comprehensive testing.

    Returns:
        List of all ActionType enum values
    """
    return [
        ActionType.ROLL_DICE,
        ActionType.BUY_PROPERTY,
        ActionType.BUILD_HOUSE,
        ActionType.BUILD_HOTEL,
        ActionType.SELL_HOUSE,
        ActionType.SELL_HOTEL,
        ActionType.MORTGAGE_PROPERTY,
        ActionType.UNMORTGAGE_PROPERTY,
        ActionType.PROPOSE_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.PAY_JAIL_FINE,
        ActionType.USE_JAIL_CARD,
        ActionType.DECLARE_BANKRUPTCY,
        ActionType.END_TURN,
    ]


@pytest.fixture
def all_card_types() -> list[CardType]:
    """Get a list of all card types for comprehensive testing.

    Returns:
        List of all CardType enum values
    """
    return [
        CardType.MOVE,
        CardType.MONEY,
        CardType.GET_OUT_OF_JAIL,
        CardType.PAY_PER_HOUSE,
        CardType.GO_TO_JAIL,
    ]
