"""Game state container with serialization for the Monopoly game engine.

This module provides the GameState dataclass that holds all mutable game state
and provides serialization/deserialization for saving and loading games.

State is PASSIVE - no business logic, just data storage and serialization.
All game logic and state mutations happen through game.py.
"""

from dataclasses import dataclass, field
from typing import Any

from .player import Player
from .property import PropertyManager
from .cards import CardDeck, CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from .types import (
    TradeOfferData,
    TOTAL_HOUSES,
    TOTAL_HOTELS,
)


@dataclass
class GameState:
    """Complete game state container.

    This is a passive data container that holds all mutable game state.
    All state mutations should be done through game.py orchestration methods,
    not by directly modifying fields.

    Attributes:
        players: List of all players in the game
        property_manager: Manages all property ownership and state
        current_player: Index of the player whose turn it is
        turn_number: Total number of turns elapsed
        houses_remaining: Number of houses available in the bank
        hotels_remaining: Number of hotels available in the bank
        chance_deck: Shuffled Chance card deck
        chest_deck: Shuffled Community Chest card deck
        last_roll: Last dice roll (die1, die2) or None if no roll yet
        doubles_count: Number of consecutive doubles rolled by current player
        game_over: Whether the game has ended
        winner: Player ID of the winner, or None if game not over
        event_log: List of game events for debugging/replay
        pending_trades: Dict mapping trade ID to TradeOfferData
    """

    players: list[Player]
    property_manager: PropertyManager
    current_player: int = 0
    turn_number: int = 0
    houses_remaining: int = TOTAL_HOUSES
    hotels_remaining: int = TOTAL_HOTELS
    chance_deck: CardDeck = field(default_factory=lambda: CardDeck.create_chance_deck())
    chest_deck: CardDeck = field(
        default_factory=lambda: CardDeck.create_community_chest_deck()
    )
    last_roll: tuple[int, int] | None = None
    doubles_count: int = 0
    game_over: bool = False
    winner: int | None = None
    event_log: list[str] = field(default_factory=list)
    pending_trades: dict[int, TradeOfferData] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete game state to JSON-compatible dict.

        Returns a dictionary containing all game state that can be
        serialized to JSON for saving games or sending to clients.

        The event_log is limited to the last 50 events to avoid
        excessive data size.

        Returns:
            Dictionary with all game state fields
        """
        return {
            "players": [p.to_dict() for p in self.players],
            "properties": self.property_manager.to_dict(),
            "current_player": self.current_player,
            "turn_number": self.turn_number,
            "houses_remaining": self.houses_remaining,
            "hotels_remaining": self.hotels_remaining,
            "chance_deck": self.chance_deck.to_dict(),
            "chest_deck": self.chest_deck.to_dict(),
            "last_roll": self.last_roll,
            "doubles_count": self.doubles_count,
            "game_over": self.game_over,
            "winner": self.winner,
            "event_log": self.event_log[-50:],  # Last 50 events only
            "pending_trades": {
                str(trade_id): trade_data
                for trade_id, trade_data in self.pending_trades.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        """Deserialize game state from JSON-compatible dict.

        Reconstructs a complete GameState from a serialized dictionary.
        Handles backwards compatibility by providing defaults for missing fields.

        Args:
            data: Dictionary containing serialized game state

        Returns:
            Reconstructed GameState instance

        Raises:
            KeyError: If required fields are missing from data
            ValueError: If data is malformed or invalid
        """
        # Reconstruct players
        players = [
            Player.from_dict(player_data) for player_data in data["players"]
        ]

        # Reconstruct property manager
        property_manager = PropertyManager.from_dict(data["properties"])

        # Reconstruct card decks with deterministic order
        chance_deck = CardDeck.from_dict(
            data.get("chance_deck", {"draw_pile": [], "discard_pile": []}),
            CHANCE_CARDS,
        )

        chest_deck = CardDeck.from_dict(
            data.get("chest_deck", {"draw_pile": [], "discard_pile": []}),
            COMMUNITY_CHEST_CARDS,
        )

        # Reconstruct pending trades (backwards compatible)
        pending_trades = {}
        if "pending_trades" in data:
            pending_trades = {
                int(trade_id): trade_data
                for trade_id, trade_data in data["pending_trades"].items()
            }

        # Create GameState with all fields
        state = cls(
            players=players,
            property_manager=property_manager,
            current_player=data.get("current_player", 0),
            turn_number=data.get("turn_number", 0),
            houses_remaining=data.get("houses_remaining", TOTAL_HOUSES),
            hotels_remaining=data.get("hotels_remaining", TOTAL_HOTELS),
            chance_deck=chance_deck,
            chest_deck=chest_deck,
            last_roll=data.get("last_roll"),
            doubles_count=data.get("doubles_count", 0),
            game_over=data.get("game_over", False),
            winner=data.get("winner"),
            event_log=data.get("event_log", []),
            pending_trades=pending_trades,
        )

        return state

    def get_observable_state(self, player_id: int) -> dict[str, Any]:
        """Get game state visible to a specific player.

        Returns a filtered view of the game state that hides information
        that should not be visible to the specified player. This is useful
        for implementing fog-of-war in multiplayer games.

        Currently returns full state, but can be extended to hide:
        - Opponent's Get Out of Jail Free cards (in some variants)
        - Exact order of cards in decks
        - Other players' private information

        Args:
            player_id: ID of the player requesting the state (unused in current implementation)

        Returns:
            Dictionary with observable game state
        """
        _ = player_id  # Reserved for future use in filtering state
        state = self.to_dict()

        # Hide exact deck order (only show counts)
        state["chance_deck"] = {
            "cards_remaining": len(self.chance_deck.cards),
            "total_cards": self.chance_deck.total_cards(),
        }
        state["chest_deck"] = {
            "cards_remaining": len(self.chest_deck.cards),
            "total_cards": self.chest_deck.total_cards(),
        }

        # In tournament mode, could hide opponent jail cards:
        # for p_data in state["players"]:
        #     if p_data["id"] != player_id:
        #         p_data["jail_cards"] = None

        return state

    def log_event(self, event: str) -> None:
        """Add an event to the event log.

        Args:
            event: Description of the game event
        """
        self.event_log.append(event)

    def get_current_player(self) -> Player:
        """Get the current player object.

        Returns:
            The Player whose turn it is

        Raises:
            IndexError: If current_player index is invalid
        """
        return self.players[self.current_player]

    def get_player(self, player_id: int) -> Player | None:
        """Get a player by their ID.

        Args:
            player_id: The player's ID

        Returns:
            The Player with the given ID, or None if not found
        """
        for player in self.players:
            if player.id == player_id:
                return player
        return None

    def get_active_players(self) -> list[Player]:
        """Get all players who are still active (not bankrupt).

        Returns:
            List of active players
        """
        return [p for p in self.players if p.is_active()]

    def count_active_players(self) -> int:
        """Count number of active (non-bankrupt) players.

        Returns:
            Number of active players
        """
        return sum(1 for p in self.players if p.is_active())

    def next_player(self) -> Player:
        """Advance to the next player and return them.

        Skips bankrupt players.

        Returns:
            The next active player

        Raises:
            RuntimeError: If no active players remain
        """
        active_count = self.count_active_players()
        if active_count == 0:
            raise RuntimeError("No active players remaining")

        # Find next active player
        attempts = 0
        max_attempts = len(self.players)

        while attempts < max_attempts:
            self.current_player = (self.current_player + 1) % len(self.players)
            if self.players[self.current_player].is_active():
                return self.players[self.current_player]
            attempts += 1

        raise RuntimeError("Could not find next active player")

    def check_winner(self) -> int | None:
        """Check if there is a winner and return their player ID.

        A player wins if they are the only active player remaining.

        Returns:
            The winner's player ID, or None if no winner yet
        """
        active = self.get_active_players()
        if len(active) == 1:
            return active[0].id
        return None

    def __str__(self) -> str:
        """Human-readable string representation."""
        current = self.get_current_player()
        return (
            f"GameState(turn={self.turn_number}, "
            f"current_player={current.name}, "
            f"active_players={self.count_active_players()}/{len(self.players)}, "
            f"game_over={self.game_over})"
        )
