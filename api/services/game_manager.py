"""
Game session management service.

Manages active games in memory, handles action execution,
and coordinates state broadcasts.
"""

import asyncio
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from monopoly_engine import MonopolyGame

from ..config import get_settings
from ..models.game import GameInfo, GameState, PlayerSlot
from ..models.websocket import WSMessage, WSMessageType

if TYPE_CHECKING:
    from monopoly_engine.actions import Action

    from .ai_manager import AIManager
    from .broadcast import ConnectionManager


@dataclass
class ActiveGame:
    """Container for an active game session."""

    id: str
    game: MonopolyGame
    created_at: datetime
    started_at: datetime | None = None
    player_slots: dict[int, PlayerSlot] = field(default_factory=dict)
    spectator_count: int = 0
    root_seed: int | None = None

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if game has exceeded timeout."""
        if self.started_at is None:
            # Not started - use created_at
            elapsed = datetime.now(UTC) - self.created_at
        else:
            elapsed = datetime.now(UTC) - self.started_at

        return elapsed > timedelta(minutes=timeout_minutes)


class GameManager:
    """Manages active game sessions.

    Thread-safe access to game instances with automatic cleanup.
    """

    def __init__(self) -> None:
        """Initialize the game manager."""
        self.games: dict[str, ActiveGame] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._connection_manager: "ConnectionManager | None" = None
        self._ai_manager: "AIManager | None" = None

    def set_connection_manager(self, manager: "ConnectionManager") -> None:
        """Set the WebSocket connection manager for broadcasts.

        Args:
            manager: The ConnectionManager instance
        """
        self._connection_manager = manager

    def set_ai_manager(self, manager: "AIManager") -> None:
        """Set the AI manager for AI opponent turns.

        Args:
            manager: The AIManager instance
        """
        self._ai_manager = manager

    async def create_game(
        self,
        num_players: int = 4,
        player_names: list[str] | None = None,
        seed: int | None = None,
    ) -> str:
        """Create a new game session.

        Args:
            num_players: Number of player slots (2-8)
            player_names: Optional player names
            seed: Random seed for determinism

        Returns:
            Game ID (UUID string)

        Raises:
            ValueError: If parameters are invalid or max games reached
        """
        settings = get_settings()

        async with self._lock:
            if len(self.games) >= settings.max_concurrent_games:
                raise ValueError("Maximum concurrent games reached")

            game_id = str(uuid.uuid4())

            # Generate player names if not provided
            if player_names is None:
                player_names = [f"Player {i + 1}" for i in range(num_players)]
            elif len(player_names) != num_players:
                raise ValueError(f"Expected {num_players} player names, got {len(player_names)}")

            seed = secrets.randbits(64) if seed is None else seed

            # Create game engine instance
            game = MonopolyGame(
                num_players=num_players,
                player_names=player_names,
                seed=seed,
            )

            # Create player slots (initially empty - no sessions assigned)
            player_slots = {
                i: PlayerSlot(
                    player_id=i,
                    name=player_names[i],
                    session_id=None,
                    is_ai=False,
                    is_ready=False,
                )
                for i in range(num_players)
            }

            active_game = ActiveGame(
                id=game_id,
                root_seed=seed,
                game=game,
                created_at=datetime.now(UTC),
                player_slots=player_slots,
            )

            self.games[game_id] = active_game

            return game_id

    async def get_game(self, game_id: str) -> ActiveGame | None:
        """Get an active game by ID.

        Args:
            game_id: The game ID to look up

        Returns:
            ActiveGame if found, None otherwise
        """
        return self.games.get(game_id)

    async def get_game_state(self, game_id: str) -> GameState | None:
        """Get serialized game state.

        Args:
            game_id: The game ID to get state for

        Returns:
            GameState if game found, None otherwise
        """
        game = await self.get_game(game_id)
        if game is None:
            return None

        return GameState.from_engine(game.game, game.player_slots)

    async def execute_action(
        self,
        game_id: str,
        action: "Action",
        broadcast: bool = True,
        exclude_session: str | None = None,
    ) -> tuple[bool, str]:
        """Execute a game action.

        Args:
            game_id: The game to execute action on
            action: The action to execute
            broadcast: Whether to broadcast state update (default True)
            exclude_session: Session to exclude from broadcast (action sender)

        Returns:
            Tuple of (success, message)
        """
        async with self._lock:
            active_game = self.games.get(game_id)
            if active_game is None:
                return False, "Game not found"

            # Validate action
            is_valid, error = action.validate(active_game.game)
            if not is_valid:
                return False, error

            # Execute action
            try:
                active_game.game.apply_action(action.player_id, action)
            except Exception as e:
                return False, f"Action execution failed: {e}"

        # Broadcast state update to other players (outside lock)
        if broadcast and self._connection_manager:
            state = await self.get_game_state(game_id)
            if state:
                await self._connection_manager.broadcast_to_game(
                    game_id,
                    WSMessage(
                        type=WSMessageType.STATE_UPDATE,
                        data=state.model_dump(),
                    ),
                    exclude_session=exclude_session,
                )

        # Check if we should trigger AI turns
        await self._maybe_process_ai_turns(game_id)

        return True, ""

    async def _maybe_process_ai_turns(self, game_id: str) -> None:
        """Process AI turns if the current player is AI.

        This is called after action execution to check if the next player
        is an AI and should take their turn automatically.

        Args:
            game_id: The game ID
        """
        if self._ai_manager is None:
            return

        active_game = self.games.get(game_id)
        if active_game is None or active_game.game.game_over:
            return

        current_player = active_game.game.decision_player
        if self._ai_manager.is_ai_player(game_id, current_player):
            # Schedule AI turn processing (don't block the response)
            import asyncio

            asyncio.create_task(self._ai_manager.process_ai_turns_for_game(self, game_id))

    async def claim_player_slot(
        self,
        game_id: str,
        player_id: int,
        session_id: str,
        player_name: str | None = None,
    ) -> tuple[bool, str, bool]:
        """Claim a player slot for a session.

        Supports reconnection within the reconnect window.

        Args:
            game_id: The game ID
            player_id: The player slot to claim
            session_id: The session claiming the slot
            player_name: Optional name override

        Returns:
            Tuple of (success, message, is_reconnect)
        """
        settings = get_settings()
        async with self._lock:
            active_game = self.games.get(game_id)
            if active_game is None:
                return False, "Game not found", False

            if player_id not in active_game.player_slots:
                return False, f"Invalid player ID: {player_id}", False

            slot = active_game.player_slots[player_id]

            # Check if this is a reconnection (same session_id returning)
            if slot.session_id == session_id:
                # Clear disconnection status - this is a reconnect
                is_reconnect = slot.disconnected_at is not None
                slot.disconnected_at = None
                if player_name:
                    slot.name = player_name
                return True, "", is_reconnect

            # Check if slot is claimed by someone else
            if slot.session_id is not None:
                # Check if previous owner is within reconnect window
                if slot.disconnected_at is not None:
                    elapsed = datetime.now(UTC) - slot.disconnected_at
                    if elapsed <= timedelta(seconds=settings.ws_reconnect_window):
                        return (
                            False,
                            f"Player slot {player_id} is reserved for reconnection",
                            False,
                        )
                    # Reconnect window expired - allow claiming
                else:
                    # Slot is actively held by another session
                    return False, f"Player slot {player_id} is already claimed", False

            # Claim the slot
            slot.session_id = session_id
            slot.disconnected_at = None
            if player_name:
                slot.name = player_name

            return True, "", False

    async def get_player_id_by_session(
        self,
        game_id: str,
        session_id: str,
    ) -> int | None:
        """Get player ID for a session in a game.

        Args:
            game_id: The game ID
            session_id: The session ID to look up

        Returns:
            The player ID if found, None otherwise
        """
        async with self._lock:
            active_game = self.games.get(game_id)
            if active_game is None:
                return None

            for player_id, slot in active_game.player_slots.items():
                if slot.session_id == session_id:
                    return player_id

            return None

    async def release_player_slot(
        self,
        game_id: str,
        player_id: int,
        session_id: str,
        permanent: bool = False,
    ) -> bool:
        """Release a player slot.

        By default, marks the slot as disconnected for reconnection support.
        Set permanent=True to fully release the slot (e.g., player left game).

        Args:
            game_id: The game ID
            player_id: The player slot to release
            session_id: The session releasing the slot (must match)
            permanent: If True, fully release slot. If False, mark disconnected.

        Returns:
            True if released/marked, False otherwise
        """
        async with self._lock:
            active_game = self.games.get(game_id)
            if active_game is None:
                return False

            if player_id not in active_game.player_slots:
                return False

            slot = active_game.player_slots[player_id]

            # Only release if session matches
            if slot.session_id == session_id:
                if permanent:
                    # Fully release the slot
                    slot.session_id = None
                    slot.disconnected_at = None
                else:
                    # Mark as disconnected for potential reconnection
                    slot.disconnected_at = datetime.now(UTC)
                return True

            return False

    async def get_player_slot_by_session(
        self,
        game_id: str,
        session_id: str,
    ) -> int | None:
        """Get player ID for a session in a game.

        Args:
            game_id: The game ID
            session_id: The session to look up

        Returns:
            Player ID if found, None otherwise
        """
        active_game = self.games.get(game_id)
        if active_game is None:
            return None

        for player_id, slot in active_game.player_slots.items():
            if slot.session_id == session_id:
                return player_id

        return None

    async def delete_game(self, game_id: str) -> bool:
        """Delete a game session.

        Args:
            game_id: The game ID to delete

        Returns:
            True if game was deleted, False if not found
        """
        async with self._lock:
            if game_id in self.games:
                del self.games[game_id]
                return True
            return False

    async def list_games(self) -> list[GameInfo]:
        """List all active games.

        Returns:
            List of GameInfo summaries
        """
        return [
            GameInfo(
                id=g.id,
                num_players=len(g.player_slots),
                players_joined=sum(1 for p in g.player_slots.values() if p.session_id is not None),
                started=g.started_at is not None,
                game_over=g.game.game_over,
                created_at=g.created_at,
            )
            for g in self.games.values()
        ]

    async def start_cleanup_task(self) -> None:
        """Start background task to clean up expired games."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_games())

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.games.clear()

    async def _cleanup_expired_games(self) -> None:
        """Background task to clean up expired games."""
        settings = get_settings()
        while True:
            await asyncio.sleep(60)  # Check every minute

            async with self._lock:
                expired = [
                    gid
                    for gid, g in self.games.items()
                    if g.is_expired(settings.game_timeout_minutes)
                ]
                for gid in expired:
                    del self.games[gid]

    def game_count(self) -> int:
        """Get number of active games.

        Returns:
            Count of active games
        """
        return len(self.games)
