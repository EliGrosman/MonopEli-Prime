"""Tests for state.py - GameState container and serialization."""

import pytest
from monopoly_engine.state import GameState
from monopoly_engine.player import Player
from monopoly_engine.property import PropertyManager, Property
from monopoly_engine.cards import CardDeck, CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from monopoly_engine.types import (
    TradeOfferData,
    TOTAL_HOUSES,
    TOTAL_HOTELS,
)


@pytest.fixture
def basic_game_state() -> GameState:
    """Create a basic game state with 2 players."""
    players = [
        Player(id=0, name="Alice", money=1500, position=0),
        Player(id=1, name="Bob", money=1500, position=5),
    ]
    property_manager = PropertyManager()
    return GameState(
        players=players,
        property_manager=property_manager,
    )


@pytest.fixture
def game_state_with_properties() -> GameState:
    """Create a game state with some owned properties."""
    players = [
        Player(id=0, name="Alice", money=1200, position=10),
        Player(id=1, name="Bob", money=800, position=15),
    ]
    property_manager = PropertyManager()

    # Alice owns some properties
    property_manager.properties[1].owner = 0  # Mediterranean Ave
    property_manager.properties[3].owner = 0  # Baltic Ave
    property_manager.properties[1].houses = 2
    property_manager.properties[6].owner = 0  # Oriental Ave
    property_manager.properties[6].mortgaged = True

    # Bob owns a railroad
    property_manager.properties[5].owner = 1  # Reading Railroad

    return GameState(
        players=players,
        property_manager=property_manager,
        current_player=1,
        turn_number=10,
        houses_remaining=30,
        hotels_remaining=12,
    )


class TestGameStateInitialization:
    """Test GameState initialization."""

    def test_basic_initialization(self, basic_game_state: GameState) -> None:
        """Test basic GameState initialization."""
        assert len(basic_game_state.players) == 2
        assert basic_game_state.current_player == 0
        assert basic_game_state.turn_number == 0
        assert basic_game_state.houses_remaining == TOTAL_HOUSES
        assert basic_game_state.hotels_remaining == TOTAL_HOTELS
        assert basic_game_state.last_roll is None
        assert basic_game_state.doubles_count == 0
        assert basic_game_state.game_over is False
        assert basic_game_state.winner is None
        assert len(basic_game_state.event_log) == 0
        assert len(basic_game_state.pending_trades) == 0

    def test_card_decks_initialized(self, basic_game_state: GameState) -> None:
        """Test that card decks are properly initialized."""
        assert basic_game_state.chance_deck is not None
        assert basic_game_state.chest_deck is not None
        assert basic_game_state.chance_deck.total_cards() == len(CHANCE_CARDS)
        assert basic_game_state.chest_deck.total_cards() == len(COMMUNITY_CHEST_CARDS)

    def test_property_manager_initialized(self, basic_game_state: GameState) -> None:
        """Test that property manager is properly initialized."""
        assert basic_game_state.property_manager is not None
        assert len(basic_game_state.property_manager.properties) == 28


class TestGameStateSerialization:
    """Test GameState serialization and deserialization."""

    def test_to_dict_basic(self, basic_game_state: GameState) -> None:
        """Test serialization of basic game state."""
        data = basic_game_state.to_dict()

        assert "players" in data
        assert "properties" in data
        assert "current_player" in data
        assert "turn_number" in data
        assert "houses_remaining" in data
        assert "hotels_remaining" in data
        assert "chance_deck" in data
        assert "chest_deck" in data
        assert "last_roll" in data
        assert "doubles_count" in data
        assert "game_over" in data
        assert "winner" in data
        assert "event_log" in data
        assert "pending_trades" in data

        assert len(data["players"]) == 2
        assert data["current_player"] == 0
        assert data["turn_number"] == 0
        assert data["houses_remaining"] == TOTAL_HOUSES
        assert data["hotels_remaining"] == TOTAL_HOTELS

    def test_to_dict_with_properties(
        self, game_state_with_properties: GameState
    ) -> None:
        """Test serialization with owned properties."""
        data = game_state_with_properties.to_dict()

        # Check player data
        assert len(data["players"]) == 2
        assert data["players"][0]["money"] == 1200
        assert data["players"][1]["money"] == 800

        # Check property data
        properties = data["properties"]
        assert properties["1"]["owner"] == 0
        assert properties["1"]["houses"] == 2
        assert properties["1"]["mortgaged"] is False
        assert properties["6"]["owner"] == 0
        assert properties["6"]["mortgaged"] is True
        assert properties["5"]["owner"] == 1

        # Check game state
        assert data["current_player"] == 1
        assert data["turn_number"] == 10
        assert data["houses_remaining"] == 30

    def test_to_dict_with_last_roll(self, basic_game_state: GameState) -> None:
        """Test serialization with last roll data."""
        basic_game_state.last_roll = (3, 4)
        data = basic_game_state.to_dict()

        assert data["last_roll"] == (3, 4)

    def test_to_dict_event_log_limit(self, basic_game_state: GameState) -> None:
        """Test that event log is limited to last 50 events."""
        # Add 100 events
        for i in range(100):
            basic_game_state.event_log.append(f"Event {i}")

        data = basic_game_state.to_dict()

        # Should only have last 50
        assert len(data["event_log"]) == 50
        assert data["event_log"][0] == "Event 50"
        assert data["event_log"][-1] == "Event 99"

    def test_to_dict_with_pending_trades(self, basic_game_state: GameState) -> None:
        """Test serialization with pending trades."""
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1, 3],
            "give_money": 100,
            "want_properties": [5],
            "want_money": 0,
        }
        basic_game_state.pending_trades[0] = trade

        data = basic_game_state.to_dict()

        assert "0" in data["pending_trades"]
        assert data["pending_trades"]["0"]["from_player"] == 0
        assert data["pending_trades"]["0"]["to_player"] == 1

    def test_from_dict_basic(self, basic_game_state: GameState) -> None:
        """Test deserialization of basic game state."""
        data = basic_game_state.to_dict()
        restored = GameState.from_dict(data)

        assert len(restored.players) == 2
        assert restored.players[0].name == "Alice"
        assert restored.players[1].name == "Bob"
        assert restored.current_player == 0
        assert restored.turn_number == 0
        assert restored.houses_remaining == TOTAL_HOUSES
        assert restored.hotels_remaining == TOTAL_HOTELS

    def test_from_dict_with_properties(
        self, game_state_with_properties: GameState
    ) -> None:
        """Test deserialization with owned properties."""
        data = game_state_with_properties.to_dict()
        restored = GameState.from_dict(data)

        # Check players
        assert len(restored.players) == 2
        assert restored.players[0].money == 1200
        assert restored.players[1].money == 800

        # Check properties
        assert restored.property_manager.properties[1].owner == 0
        assert restored.property_manager.properties[1].houses == 2
        assert restored.property_manager.properties[6].owner == 0
        assert restored.property_manager.properties[6].mortgaged is True
        assert restored.property_manager.properties[5].owner == 1

        # Check game state
        assert restored.current_player == 1
        assert restored.turn_number == 10
        assert restored.houses_remaining == 30

    def test_from_dict_with_card_decks(
        self, game_state_with_properties: GameState
    ) -> None:
        """Test that card decks are properly restored."""
        data = game_state_with_properties.to_dict()
        restored = GameState.from_dict(data)

        assert restored.chance_deck is not None
        assert restored.chest_deck is not None
        assert restored.chance_deck.total_cards() == len(CHANCE_CARDS)
        assert restored.chest_deck.total_cards() == len(COMMUNITY_CHEST_CARDS)

    def test_from_dict_backwards_compatible(self) -> None:
        """Test that from_dict handles missing fields gracefully."""
        # Minimal data with only required fields
        data = {
            "players": [
                {
                    "id": 0,
                    "name": "Alice",
                    "money": 1500,
                    "position": 0,
                    "jail_cards": 0,
                    "in_jail": False,
                    "jail_turns": 0,
                    "bankrupt": False,
                }
            ],
            "properties": {},
        }

        restored = GameState.from_dict(data)

        # Should have defaults for missing fields
        assert restored.current_player == 0
        assert restored.turn_number == 0
        assert restored.houses_remaining == TOTAL_HOUSES
        assert restored.hotels_remaining == TOTAL_HOTELS
        assert restored.doubles_count == 0
        assert restored.game_over is False
        assert restored.winner is None
        assert len(restored.event_log) == 0
        assert len(restored.pending_trades) == 0

    def test_roundtrip_serialization(
        self, game_state_with_properties: GameState
    ) -> None:
        """Test that serialization and deserialization are inverses."""
        original = game_state_with_properties
        data = original.to_dict()
        restored = GameState.from_dict(data)

        # Check key fields match
        assert len(restored.players) == len(original.players)
        assert restored.current_player == original.current_player
        assert restored.turn_number == original.turn_number
        assert restored.houses_remaining == original.houses_remaining
        assert restored.hotels_remaining == original.hotels_remaining


class TestObservableState:
    """Test observable state filtering."""

    def test_get_observable_state_basic(self, basic_game_state: GameState) -> None:
        """Test getting observable state for a player."""
        observable = basic_game_state.get_observable_state(0)

        assert "players" in observable
        assert "properties" in observable
        assert "current_player" in observable
        assert "turn_number" in observable

    def test_get_observable_state_hides_deck_order(
        self, basic_game_state: GameState
    ) -> None:
        """Test that observable state hides exact deck order."""
        observable = basic_game_state.get_observable_state(0)

        # Should have deck info but not exact order
        assert "chance_deck" in observable
        assert "chest_deck" in observable
        assert "cards_remaining" in observable["chance_deck"]
        assert "total_cards" in observable["chance_deck"]
        assert "draw_pile" not in observable["chance_deck"]
        assert "discard_pile" not in observable["chance_deck"]


class TestGameStateHelpers:
    """Test helper methods on GameState."""

    def test_log_event(self, basic_game_state: GameState) -> None:
        """Test logging events."""
        assert len(basic_game_state.event_log) == 0

        basic_game_state.log_event("Player 1 rolled a 7")
        assert len(basic_game_state.event_log) == 1
        assert basic_game_state.event_log[0] == "Player 1 rolled a 7"

        basic_game_state.log_event("Player 1 bought Mediterranean Ave")
        assert len(basic_game_state.event_log) == 2

    def test_get_current_player(self, basic_game_state: GameState) -> None:
        """Test getting current player."""
        current = basic_game_state.get_current_player()
        assert current.id == 0
        assert current.name == "Alice"

        basic_game_state.current_player = 1
        current = basic_game_state.get_current_player()
        assert current.id == 1
        assert current.name == "Bob"

    def test_get_player(self, basic_game_state: GameState) -> None:
        """Test getting player by ID."""
        player = basic_game_state.get_player(0)
        assert player is not None
        assert player.name == "Alice"

        player = basic_game_state.get_player(1)
        assert player is not None
        assert player.name == "Bob"

        player = basic_game_state.get_player(99)
        assert player is None

    def test_get_active_players(self, basic_game_state: GameState) -> None:
        """Test getting active players."""
        active = basic_game_state.get_active_players()
        assert len(active) == 2

        # Bankrupt one player
        basic_game_state.players[1].declare_bankrupt()
        active = basic_game_state.get_active_players()
        assert len(active) == 1
        assert active[0].id == 0

    def test_count_active_players(self, basic_game_state: GameState) -> None:
        """Test counting active players."""
        assert basic_game_state.count_active_players() == 2

        basic_game_state.players[0].declare_bankrupt()
        assert basic_game_state.count_active_players() == 1

        basic_game_state.players[1].declare_bankrupt()
        assert basic_game_state.count_active_players() == 0

    def test_next_player(self, basic_game_state: GameState) -> None:
        """Test advancing to next player."""
        assert basic_game_state.current_player == 0

        next_player = basic_game_state.next_player()
        assert next_player.id == 1
        assert basic_game_state.current_player == 1

        next_player = basic_game_state.next_player()
        assert next_player.id == 0
        assert basic_game_state.current_player == 0

    def test_next_player_skips_bankrupt(self) -> None:
        """Test that next_player skips bankrupt players."""
        players = [
            Player(id=0, name="Alice"),
            Player(id=1, name="Bob"),
            Player(id=2, name="Charlie"),
            Player(id=3, name="Diana"),
        ]
        state = GameState(players=players, property_manager=PropertyManager())

        # Bankrupt player 1 and 2
        players[1].declare_bankrupt()
        players[2].declare_bankrupt()

        # Start at player 0, next should be player 3
        next_player = state.next_player()
        assert next_player.id == 3
        assert state.current_player == 3

        # From player 3, next should wrap to player 0
        next_player = state.next_player()
        assert next_player.id == 0
        assert state.current_player == 0

    def test_next_player_no_active_players_raises(self) -> None:
        """Test that next_player raises if no active players."""
        players = [
            Player(id=0, name="Alice"),
            Player(id=1, name="Bob"),
        ]
        state = GameState(players=players, property_manager=PropertyManager())

        # Bankrupt all players
        players[0].declare_bankrupt()
        players[1].declare_bankrupt()

        with pytest.raises(RuntimeError, match="No active players remaining"):
            state.next_player()

    def test_check_winner_no_winner(self, basic_game_state: GameState) -> None:
        """Test check_winner when no winner yet."""
        winner = basic_game_state.check_winner()
        assert winner is None

    def test_check_winner_single_player_remaining(
        self, basic_game_state: GameState
    ) -> None:
        """Test check_winner when one player remains."""
        basic_game_state.players[1].declare_bankrupt()

        winner = basic_game_state.check_winner()
        assert winner == 0

    def test_check_winner_no_players_remaining(
        self, basic_game_state: GameState
    ) -> None:
        """Test check_winner when no players remain (edge case)."""
        basic_game_state.players[0].declare_bankrupt()
        basic_game_state.players[1].declare_bankrupt()

        winner = basic_game_state.check_winner()
        assert winner is None

    def test_str_representation(self, basic_game_state: GameState) -> None:
        """Test string representation of GameState."""
        s = str(basic_game_state)
        assert "GameState" in s
        assert "turn=0" in s
        assert "current_player=Alice" in s
        assert "active_players=2/2" in s
        assert "game_over=False" in s


class TestGameStateWithMultiplePlayers:
    """Test GameState with more than 2 players."""

    @pytest.fixture
    def four_player_state(self) -> GameState:
        """Create a game state with 4 players."""
        players = [
            Player(id=0, name="Alice"),
            Player(id=1, name="Bob"),
            Player(id=2, name="Charlie"),
            Player(id=3, name="Diana"),
        ]
        return GameState(players=players, property_manager=PropertyManager())

    def test_four_player_initialization(self, four_player_state: GameState) -> None:
        """Test initialization with 4 players."""
        assert len(four_player_state.players) == 4
        assert four_player_state.count_active_players() == 4

    def test_four_player_next_player_cycle(
        self, four_player_state: GameState
    ) -> None:
        """Test cycling through 4 players."""
        assert four_player_state.current_player == 0

        four_player_state.next_player()
        assert four_player_state.current_player == 1

        four_player_state.next_player()
        assert four_player_state.current_player == 2

        four_player_state.next_player()
        assert four_player_state.current_player == 3

        four_player_state.next_player()
        assert four_player_state.current_player == 0

    def test_four_player_serialization(self, four_player_state: GameState) -> None:
        """Test serialization with 4 players."""
        data = four_player_state.to_dict()
        assert len(data["players"]) == 4

        restored = GameState.from_dict(data)
        assert len(restored.players) == 4
        assert restored.players[2].name == "Charlie"


class TestGameStateEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_current_player_invalid_index(self) -> None:
        """Test get_current_player with invalid index."""
        players = [Player(id=0, name="Alice")]
        state = GameState(players=players, property_manager=PropertyManager())
        state.current_player = 99  # Invalid index

        with pytest.raises(IndexError):
            state.get_current_player()

    def test_game_over_state(self, basic_game_state: GameState) -> None:
        """Test game over state."""
        basic_game_state.game_over = True
        basic_game_state.winner = 0

        data = basic_game_state.to_dict()
        assert data["game_over"] is True
        assert data["winner"] == 0

        restored = GameState.from_dict(data)
        assert restored.game_over is True
        assert restored.winner == 0

    def test_last_roll_doubles(self, basic_game_state: GameState) -> None:
        """Test tracking doubles rolls."""
        basic_game_state.last_roll = (3, 3)
        basic_game_state.doubles_count = 1

        data = basic_game_state.to_dict()
        assert data["last_roll"] == (3, 3)
        assert data["doubles_count"] == 1

        restored = GameState.from_dict(data)
        assert restored.last_roll == (3, 3)
        assert restored.doubles_count == 1
