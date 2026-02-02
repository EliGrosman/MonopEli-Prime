"""Pytest fixtures for Monopoly engine tests."""

import pytest

from monopoly_engine import (
    Player,
    Property,
    PropertyManager,
    Board,
    PropertySpace,
    RailroadSpace,
    UtilitySpace,
    CardDeck,
    CHANCE_CARDS,
    COMMUNITY_CHEST_CARDS,
    PropertyColor,
    PROPERTY_GROUPS,
)


@pytest.fixture
def player() -> Player:
    """Create a basic player with default settings."""
    return Player(id=0, name="Test Player")


@pytest.fixture
def player_with_money() -> Player:
    """Create a player with extra money for testing purchases."""
    return Player(id=0, name="Rich Player", money=5000)


@pytest.fixture
def two_players() -> list[Player]:
    """Create two players for testing interactions."""
    return [
        Player(id=0, name="Player 1"),
        Player(id=1, name="Player 2"),
    ]


@pytest.fixture
def four_players() -> list[Player]:
    """Create four players for a full game."""
    return [
        Player(id=i, name=f"Player {i + 1}")
        for i in range(4)
    ]


@pytest.fixture
def property_manager() -> PropertyManager:
    """Create a fresh property manager."""
    return PropertyManager()


@pytest.fixture
def property_manager_with_owner(property_manager: PropertyManager) -> PropertyManager:
    """Create a property manager with some owned properties."""
    # Player 0 owns Mediterranean and Baltic (brown monopoly)
    property_manager.properties[1].owner = 0
    property_manager.properties[3].owner = 0
    # Player 1 owns Reading Railroad
    property_manager.properties[5].owner = 1
    return property_manager


@pytest.fixture
def property_manager_with_monopoly(property_manager: PropertyManager) -> PropertyManager:
    """Create a property manager where player 0 has a monopoly."""
    # Player 0 owns all light blue properties
    for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
        property_manager.properties[pos].owner = 0
    return property_manager


@pytest.fixture
def property_manager_with_houses(
    property_manager_with_monopoly: PropertyManager,
) -> PropertyManager:
    """Create a property manager with some houses built."""
    # Add 2 houses to each light blue property
    for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
        property_manager_with_monopoly.properties[pos].houses = 2
    return property_manager_with_monopoly


@pytest.fixture
def chance_deck() -> CardDeck:
    """Create a fresh shuffled Chance deck."""
    return CardDeck.create_chance_deck(seed=42)


@pytest.fixture
def community_chest_deck() -> CardDeck:
    """Create a fresh shuffled Community Chest deck."""
    return CardDeck.create_community_chest_deck(seed=42)


@pytest.fixture
def mediterranean_space() -> PropertySpace:
    """Get the Mediterranean Avenue space."""
    space = Board.get_space(1)
    assert isinstance(space, PropertySpace)
    return space


@pytest.fixture
def reading_railroad() -> RailroadSpace:
    """Get the Reading Railroad space."""
    space = Board.get_space(5)
    assert isinstance(space, RailroadSpace)
    return space


@pytest.fixture
def electric_company() -> UtilitySpace:
    """Get the Electric Company space."""
    space = Board.get_space(12)
    assert isinstance(space, UtilitySpace)
    return space


@pytest.fixture
def all_property_positions() -> list[int]:
    """Get all regular property positions."""
    return [
        i for i, s in enumerate(Board.SPACES)
        if isinstance(s, PropertySpace)
    ]


@pytest.fixture
def all_railroad_positions() -> list[int]:
    """Get all railroad positions."""
    return [5, 15, 25, 35]


@pytest.fixture
def all_utility_positions() -> list[int]:
    """Get all utility positions."""
    return [12, 28]
