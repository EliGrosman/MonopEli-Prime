"""Tests for the MonopolyGame class.

This module tests the main game orchestrator that ties everything together.
"""

import pytest

from monopoly_engine import (
    MonopolyGame,
    Player,
    InvalidPlayerError,
    InvalidPropertyError,
)


class TestMonopolyGameInitialization:
    """Test game initialization."""

    def test_default_initialization(self) -> None:
        """Test creating a game with default settings."""
        game = MonopolyGame(num_players=2)

        assert len(game.players) == 2
        assert game.current_player == 0
        assert game.houses_remaining == 32
        assert game.hotels_remaining == 12
        assert not game.game_over
        assert game.winner is None

    def test_custom_player_names(self) -> None:
        """Test creating a game with custom player names."""
        names = ["Alice", "Bob", "Charlie"]
        game = MonopolyGame(num_players=3, player_names=names)

        assert len(game.players) == 3
        assert game.players[0].name == "Alice"
        assert game.players[1].name == "Bob"
        assert game.players[2].name == "Charlie"

    def test_seeded_game_is_deterministic(self) -> None:
        """Test that seeded games produce same results."""
        game1 = MonopolyGame(num_players=2, seed=42)
        game2 = MonopolyGame(num_players=2, seed=42)

        # Roll dice should produce same results
        roll1 = game1.roll_dice()
        roll2 = game2.roll_dice()

        assert roll1 == roll2

    def test_invalid_num_players(self) -> None:
        """Test that invalid number of players raises error."""
        with pytest.raises(ValueError, match="Number of players must be between 2 and 8"):
            MonopolyGame(num_players=1)

        with pytest.raises(ValueError, match="Number of players must be between 2 and 8"):
            MonopolyGame(num_players=9)

    def test_mismatched_player_names(self) -> None:
        """Test that mismatched player names raises error."""
        with pytest.raises(ValueError, match="Expected 3 player names, got 2"):
            MonopolyGame(num_players=3, player_names=["Alice", "Bob"])


class TestMonopolyGameMovement:
    """Test player movement orchestration."""

    def test_move_player_forward(self) -> None:
        """Test moving a player forward."""
        game = MonopolyGame(num_players=2)
        player = game.players[0]

        new_pos, passed_go = game.move_player(0, 5)

        assert new_pos == 5
        assert not passed_go
        assert player.position == 5

    def test_move_player_passes_go(self) -> None:
        """Test that passing GO collects $200."""
        game = MonopolyGame(num_players=2)
        player = game.players[0]
        starting_money = player.money

        # Move to position 38
        player.position = 38

        # Move 5 spaces (wraps to position 3)
        new_pos, passed_go = game.move_player(0, 5)

        assert new_pos == 3
        assert passed_go
        assert player.money == starting_money + 200

    def test_move_player_to(self) -> None:
        """Test moving a player to a specific position."""
        game = MonopolyGame(num_players=2)
        player = game.players[0]

        # Move to position 20 (doesn't pass GO from position 0)
        passed_go = game.move_player_to(0, 20)

        assert player.position == 20
        assert not passed_go

        # Now move from 30 to 5 (passes GO)
        player.position = 30
        passed_go = game.move_player_to(0, 5)

        assert player.position == 5
        assert passed_go

    def test_send_to_jail(self) -> None:
        """Test sending a player to jail."""
        game = MonopolyGame(num_players=2)
        player = game.players[0]

        game.send_to_jail(0)

        assert player.position == 10  # Jail position
        assert player.in_jail
        assert game.doubles_count == 0


class TestMonopolyGameHouseHotelManagement:
    """Test house and hotel inventory management."""

    def test_allocate_house(self) -> None:
        """Test allocating a house from the bank."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0

        starting_houses = game.houses_remaining

        success = game.allocate_house(1)

        assert success
        assert prop.houses == 1
        assert game.houses_remaining == starting_houses - 1

    def test_allocate_house_when_none_remaining(self) -> None:
        """Test that allocating house fails when none remain."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0

        game.houses_remaining = 0

        success = game.allocate_house(1)

        assert not success
        assert prop.houses == 0

    def test_return_house(self) -> None:
        """Test returning a house to the bank."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0
        prop.houses = 2

        game.houses_remaining = 30

        game.return_house(1)

        assert prop.houses == 1
        assert game.houses_remaining == 31

    def test_allocate_hotel(self) -> None:
        """Test allocating a hotel."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0
        prop.houses = 4

        game.houses_remaining = 30
        game.hotels_remaining = 12

        success = game.allocate_hotel(1)

        assert success
        assert prop.houses == 5  # Hotel
        assert game.houses_remaining == 34  # Returned 4 houses
        assert game.hotels_remaining == 11

    def test_return_hotel(self) -> None:
        """Test returning a hotel to the bank."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0
        prop.houses = 5  # Hotel

        game.houses_remaining = 28
        game.hotels_remaining = 11

        success = game.return_hotel(1)

        assert success
        assert prop.houses == 4  # Back to 4 houses
        assert game.houses_remaining == 24  # Used 4 houses
        assert game.hotels_remaining == 12


class TestMonopolyGamePropertyTransfer:
    """Test property ownership transfer."""

    def test_transfer_property(self) -> None:
        """Test transferring property between players."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0

        game.transfer_property(1, 0, 1)

        assert prop.owner == 1

    def test_reset_property(self) -> None:
        """Test resetting property to unowned state."""
        game = MonopolyGame(num_players=2)
        prop = game.property_manager.get(1)
        assert prop is not None
        prop.owner = 0
        prop.houses = 2
        prop.mortgaged = True

        game.reset_property(1)

        assert prop.owner is None
        assert prop.houses == 0
        assert not prop.mortgaged


class TestMonopolyGameTrading:
    """Test trade management."""

    def test_propose_trade(self) -> None:
        """Test proposing a trade."""
        game = MonopolyGame(num_players=2)

        trade_id = game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            give_money=100,
            want_properties=[3],
            want_money=50,
        )

        assert trade_id == 0
        assert len(game.state.pending_trades) == 1

    def test_accept_trade(self) -> None:
        """Test accepting a trade."""
        game = MonopolyGame(num_players=2)

        # Set up properties
        game.property_manager.get(1).owner = 0  # type: ignore
        game.property_manager.get(3).owner = 1  # type: ignore

        # Propose trade
        trade_id = game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            give_money=100,
            want_properties=[3],
            want_money=50,
        )

        # Accept trade
        game.accept_trade(trade_id)

        # Check ownership transferred
        assert game.property_manager.get(1).owner == 1  # type: ignore
        assert game.property_manager.get(3).owner == 0  # type: ignore

        # Check money transferred
        assert game.players[0].money == 1500 - 100 + 50  # Gave 100, got 50
        assert game.players[1].money == 1500 + 100 - 50  # Got 100, gave 50

        # Check trade removed
        assert len(game.state.pending_trades) == 0

    def test_reject_trade(self) -> None:
        """Test rejecting a trade."""
        game = MonopolyGame(num_players=2)

        trade_id = game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            give_money=100,
            want_properties=[3],
            want_money=50,
        )

        game.reject_trade(trade_id)

        assert len(game.state.pending_trades) == 0


class TestMonopolyGameBankruptcy:
    """Test bankruptcy handling."""

    def test_handle_bankruptcy_to_bank(self) -> None:
        """Test bankruptcy to the bank."""
        game = MonopolyGame(num_players=2)

        # Give player some properties
        game.property_manager.get(1).owner = 0  # type: ignore
        game.property_manager.get(3).owner = 0  # type: ignore

        game.handle_bankruptcy(0)

        # Check player is bankrupt
        assert game.players[0].bankrupt
        assert game.players[0].money == 0

        # Check properties returned to bank
        assert game.property_manager.get(1).owner is None  # type: ignore
        assert game.property_manager.get(3).owner is None  # type: ignore

    def test_handle_bankruptcy_declares_winner(self) -> None:
        """Test that bankruptcy with one player left declares winner."""
        game = MonopolyGame(num_players=2)

        game.handle_bankruptcy(0)

        assert game.game_over
        assert game.winner == 1


class TestMonopolyGameTurnManagement:
    """Test turn management."""

    def test_end_turn(self) -> None:
        """Test ending a turn."""
        game = MonopolyGame(num_players=2)

        game.end_turn()

        assert game.current_player == 1
        assert game.turn_number == 1

    def test_next_player(self) -> None:
        """Test advancing to next player."""
        game = MonopolyGame(num_players=3)

        next_p = game.next_player()

        assert next_p.id == 1
        assert game.current_player == 1


class TestMonopolyGameSerialization:
    """Test game serialization."""

    def test_to_dict(self) -> None:
        """Test serializing game to dict."""
        game = MonopolyGame(num_players=2, seed=42)

        data = game.to_dict()

        assert "players" in data
        assert "properties" in data
        assert len(data["players"]) == 2

    def test_from_dict(self) -> None:
        """Test deserializing game from dict."""
        game1 = MonopolyGame(num_players=2, seed=42)
        game1.players[0].position = 10
        game1.players[0].money = 1000

        data = game1.to_dict()
        game2 = MonopolyGame.from_dict(data)

        assert len(game2.players) == 2
        assert game2.players[0].position == 10
        assert game2.players[0].money == 1000


class TestMonopolyGameHelpers:
    """Test helper methods."""

    def test_get_player(self) -> None:
        """Test getting a player by ID."""
        game = MonopolyGame(num_players=2)

        player = game._get_player(0)

        assert player.id == 0

    def test_get_player_invalid_raises(self) -> None:
        """Test getting invalid player raises error."""
        game = MonopolyGame(num_players=2)

        with pytest.raises(InvalidPlayerError):
            game._get_player(5)

    def test_get_property(self) -> None:
        """Test getting a property."""
        game = MonopolyGame(num_players=2)

        prop = game._get_property(1)

        assert prop.position == 1

    def test_get_property_invalid_raises(self) -> None:
        """Test getting invalid property raises error."""
        game = MonopolyGame(num_players=2)

        with pytest.raises(InvalidPropertyError):
            game._get_property(0)  # GO is not a property


class TestMonopolyGameLogging:
    """Test event logging."""

    def test_log_event(self) -> None:
        """Test logging an event."""
        game = MonopolyGame(num_players=2)

        game.log_event("Test event")

        assert "Test event" in game.state.event_log

    def test_initialization_logged(self) -> None:
        """Test that game initialization is logged."""
        game = MonopolyGame(num_players=2)

        assert len(game.state.event_log) > 0
        assert "initialized" in game.state.event_log[0].lower()
