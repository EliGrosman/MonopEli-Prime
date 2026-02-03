"""
Lobby management service.

Manages lobby lifecycle: creation, joining, ready states, and game start.
"""

import asyncio
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .broadcast import ConnectionManager
    from .game_manager import GameManager

from ..models.lobby import (
    LobbyInfo,
    LobbyPlayer,
    LobbySettings,
    LobbyState,
    LobbyStatus,
    LobbyWSMessageType,
)
from ..models.websocket import WSMessage, WSMessageType


@dataclass
class ActiveLobby:
    """Container for an active lobby."""

    id: str
    name: str
    host_session_id: str
    status: LobbyStatus
    settings: LobbySettings
    players: dict[int, LobbyPlayer]  # slot_id -> player
    created_at: datetime
    invite_code: str | None = None
    game_id: str | None = None
    spectator_sessions: set[str] = field(default_factory=set)

    def get_next_slot(self) -> int | None:
        """Get next available player slot."""
        for i in range(self.settings.max_players):
            if i not in self.players:
                return i
        return None

    def player_count(self) -> int:
        """Get number of players in lobby."""
        return len(self.players)

    def all_ready(self) -> bool:
        """Check if all players are ready (host is always ready)."""
        if len(self.players) < self.settings.min_players:
            return False
        return all(p.is_ready or p.is_host for p in self.players.values())

    def to_state(self) -> LobbyState:
        """Convert to LobbyState model."""
        return LobbyState(
            id=self.id,
            name=self.name,
            host_session_id=self.host_session_id,
            status=self.status,
            settings=self.settings,
            players=list(self.players.values()),
            spectator_count=len(self.spectator_sessions),
            created_at=self.created_at,
            game_id=self.game_id,
            invite_code=self.invite_code,
        )


class LobbyManager:
    """Manages lobby sessions."""

    def __init__(self) -> None:
        """Initialize lobby manager."""
        self.lobbies: dict[str, ActiveLobby] = {}
        self._lock = asyncio.Lock()
        self._connection_manager: "ConnectionManager | None" = None
        self._game_manager: "GameManager | None" = None

    def set_connection_manager(self, manager: "ConnectionManager") -> None:
        """Set connection manager for broadcasts."""
        self._connection_manager = manager

    def set_game_manager(self, manager: "GameManager") -> None:
        """Set game manager for starting games."""
        self._game_manager = manager

    async def create_lobby(
        self,
        name: str,
        host_name: str,
        host_session_id: str,
        settings: LobbySettings | None = None,
    ) -> tuple[str, str | None]:
        """Create a new lobby.

        Args:
            name: Lobby name
            host_name: Host player name
            host_session_id: Host's session ID
            settings: Optional lobby settings

        Returns:
            Tuple of (lobby_id, invite_code or None)
        """
        if settings is None:
            settings = LobbySettings()

        async with self._lock:
            lobby_id = str(uuid.uuid4())
            invite_code = secrets.token_urlsafe(8) if settings.private else None

            host_player = LobbyPlayer(
                slot_id=0,
                session_id=host_session_id,
                name=host_name,
                is_host=True,
                is_ready=True,  # Host is always ready
                joined_at=datetime.now(UTC),
            )

            lobby = ActiveLobby(
                id=lobby_id,
                name=name,
                host_session_id=host_session_id,
                status=LobbyStatus.WAITING,
                settings=settings,
                players={0: host_player},
                created_at=datetime.now(UTC),
                invite_code=invite_code,
            )

            self.lobbies[lobby_id] = lobby
            return lobby_id, invite_code

    async def get_lobby(self, lobby_id: str) -> ActiveLobby | None:
        """Get a lobby by ID."""
        return self.lobbies.get(lobby_id)

    async def get_lobby_state(self, lobby_id: str) -> LobbyState | None:
        """Get lobby state."""
        lobby = await self.get_lobby(lobby_id)
        if lobby is None:
            return None
        return lobby.to_state()

    async def join_lobby(
        self,
        lobby_id: str,
        player_name: str,
        session_id: str,
        invite_code: str | None = None,
    ) -> tuple[bool, str, int]:
        """Join a lobby.

        Args:
            lobby_id: Lobby to join
            player_name: Player's display name
            session_id: Player's session ID
            invite_code: Required for private lobbies

        Returns:
            Tuple of (success, message, slot_id)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found", -1

            if lobby.status != LobbyStatus.WAITING:
                return False, "Lobby is not accepting players", -1

            if lobby.settings.private and lobby.invite_code != invite_code:
                return False, "Invalid invite code", -1

            # Check if already in lobby
            for player in lobby.players.values():
                if player.session_id == session_id:
                    return True, "Already in lobby", player.slot_id

            # Get next slot
            slot_id = lobby.get_next_slot()
            if slot_id is None:
                return False, "Lobby is full", -1

            player = LobbyPlayer(
                slot_id=slot_id,
                session_id=session_id,
                name=player_name,
                is_host=False,
                is_ready=False,
                joined_at=datetime.now(UTC),
            )

            lobby.players[slot_id] = player

        # Broadcast player joined
        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.PLAYER_JOINED,
            {"slot_id": slot_id, "name": player_name},
        )

        return True, "", slot_id

    async def leave_lobby(
        self,
        lobby_id: str,
        session_id: str,
    ) -> tuple[bool, str]:
        """Leave a lobby.

        Args:
            lobby_id: Lobby to leave
            session_id: Player's session ID

        Returns:
            Tuple of (success, message)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found"

            # Find player
            player_slot = None
            player_name = ""
            for slot_id, player in lobby.players.items():
                if player.session_id == session_id:
                    player_slot = slot_id
                    player_name = player.name
                    break

            if player_slot is None:
                # Check if spectator
                if session_id in lobby.spectator_sessions:
                    lobby.spectator_sessions.discard(session_id)
                    return True, ""
                return False, "Not in lobby"

            # If host leaves, close the lobby
            if lobby.players[player_slot].is_host:
                lobby.status = LobbyStatus.CLOSED
                del self.lobbies[lobby_id]

                await self._broadcast_lobby_event(
                    lobby_id,
                    LobbyWSMessageType.LOBBY_CLOSED,
                    {"reason": "Host left"},
                )
                return True, "Lobby closed"

            # Remove player
            del lobby.players[player_slot]

        # Broadcast player left
        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.PLAYER_LEFT,
            {"slot_id": player_slot, "name": player_name},
        )

        return True, ""

    async def set_ready(
        self,
        lobby_id: str,
        session_id: str,
        ready: bool,
    ) -> tuple[bool, str]:
        """Set player ready status.

        Args:
            lobby_id: Lobby ID
            session_id: Player's session ID
            ready: Ready status

        Returns:
            Tuple of (success, message)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found"

            # Find player
            for player in lobby.players.values():
                if player.session_id == session_id:
                    player.is_ready = ready
                    msg_type = (
                        LobbyWSMessageType.PLAYER_READY
                        if ready
                        else LobbyWSMessageType.PLAYER_UNREADY
                    )

                    await self._broadcast_lobby_event(
                        lobby_id,
                        msg_type,
                        {"slot_id": player.slot_id, "name": player.name},
                    )
                    return True, ""

            return False, "Not in lobby"

    async def add_ai_player(
        self,
        lobby_id: str,
        host_session_id: str,
        ai_type: str = "rule_based",
        name: str | None = None,
    ) -> tuple[bool, str, int]:
        """Add an AI player to the lobby.

        Args:
            lobby_id: Lobby ID
            host_session_id: Must be host to add AI
            ai_type: Type of AI
            name: Optional AI name

        Returns:
            Tuple of (success, message, slot_id)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found", -1

            if lobby.host_session_id != host_session_id:
                return False, "Only host can add AI players", -1

            slot_id = lobby.get_next_slot()
            if slot_id is None:
                return False, "Lobby is full", -1

            if name is None:
                name = f"AI {slot_id + 1} ({ai_type})"

            ai_player = LobbyPlayer(
                slot_id=slot_id,
                session_id=f"ai-{uuid.uuid4()}",
                name=name,
                is_host=False,
                is_ready=True,  # AI is always ready
                is_ai=True,
                ai_type=ai_type,
                joined_at=datetime.now(UTC),
            )

            lobby.players[slot_id] = ai_player

        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.AI_ADDED,
            {"slot_id": slot_id, "name": name, "ai_type": ai_type},
        )

        return True, "", slot_id

    async def remove_ai_player(
        self,
        lobby_id: str,
        host_session_id: str,
        slot_id: int,
    ) -> tuple[bool, str]:
        """Remove an AI player from the lobby.

        Args:
            lobby_id: Lobby ID
            host_session_id: Must be host
            slot_id: AI's slot to remove

        Returns:
            Tuple of (success, message)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found"

            if lobby.host_session_id != host_session_id:
                return False, "Only host can remove AI players"

            if slot_id not in lobby.players:
                return False, "No player in that slot"

            player = lobby.players[slot_id]
            if not player.is_ai:
                return False, "That slot is not an AI player"

            del lobby.players[slot_id]

        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.AI_REMOVED,
            {"slot_id": slot_id},
        )

        return True, ""

    async def kick_player(
        self,
        lobby_id: str,
        host_session_id: str,
        slot_id: int,
    ) -> tuple[bool, str]:
        """Kick a player from the lobby.

        Args:
            lobby_id: Lobby ID
            host_session_id: Must be host
            slot_id: Player's slot to kick

        Returns:
            Tuple of (success, message)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found"

            if lobby.host_session_id != host_session_id:
                return False, "Only host can kick players"

            if slot_id not in lobby.players:
                return False, "No player in that slot"

            player = lobby.players[slot_id]
            if player.is_host:
                return False, "Cannot kick the host"

            player_name = player.name
            del lobby.players[slot_id]

        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.PLAYER_KICKED,
            {"slot_id": slot_id, "name": player_name},
        )

        return True, ""

    async def update_settings(
        self,
        lobby_id: str,
        host_session_id: str,
        settings: LobbySettings,
    ) -> tuple[bool, str]:
        """Update lobby settings.

        Args:
            lobby_id: Lobby ID
            host_session_id: Must be host
            settings: New settings

        Returns:
            Tuple of (success, message)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found"

            if lobby.host_session_id != host_session_id:
                return False, "Only host can update settings"

            if lobby.status != LobbyStatus.WAITING:
                return False, "Cannot change settings after game starts"

            # Remove players if max_players reduced
            if settings.max_players < len(lobby.players):
                return False, "Cannot reduce max players below current count"

            lobby.settings = settings

        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.SETTINGS_UPDATED,
            settings.model_dump(),
        )

        return True, ""

    async def start_game(
        self,
        lobby_id: str,
        host_session_id: str,
    ) -> tuple[bool, str, str | None]:
        """Start the game from lobby.

        Args:
            lobby_id: Lobby ID
            host_session_id: Must be host

        Returns:
            Tuple of (success, message, game_id or None)
        """
        async with self._lock:
            lobby = self.lobbies.get(lobby_id)
            if lobby is None:
                return False, "Lobby not found", None

            if lobby.host_session_id != host_session_id:
                return False, "Only host can start the game", None

            if lobby.status != LobbyStatus.WAITING:
                return False, "Game already starting or started", None

            if len(lobby.players) < lobby.settings.min_players:
                return (
                    False,
                    f"Need at least {lobby.settings.min_players} players",
                    None,
                )

            if not lobby.all_ready():
                return False, "Not all players are ready", None

            lobby.status = LobbyStatus.STARTING

        # Notify clients game is starting
        await self._broadcast_lobby_event(
            lobby_id,
            LobbyWSMessageType.GAME_STARTING,
            {},
        )

        # Create the game
        if self._game_manager is None:
            async with self._lock:
                lobby.status = LobbyStatus.WAITING
            return False, "Game manager not available", None

        try:
            # Prepare player names in slot order
            player_names = [
                lobby.players[i].name
                for i in sorted(lobby.players.keys())
            ]

            game_id = await self._game_manager.create_game(
                num_players=len(player_names),
                player_names=player_names,
            )

            async with self._lock:
                lobby.status = LobbyStatus.STARTED
                lobby.game_id = game_id

            # Notify clients game started
            await self._broadcast_lobby_event(
                lobby_id,
                LobbyWSMessageType.GAME_STARTED,
                {"game_id": game_id},
            )

            return True, "", game_id

        except Exception as e:
            async with self._lock:
                lobby.status = LobbyStatus.WAITING
            return False, f"Failed to create game: {e}", None

    async def list_lobbies(self, include_private: bool = False) -> list[LobbyInfo]:
        """List public lobbies.

        Args:
            include_private: Whether to include private lobbies

        Returns:
            List of LobbyInfo
        """
        result = []
        for lobby in self.lobbies.values():
            if lobby.status != LobbyStatus.WAITING:
                continue
            if lobby.settings.private and not include_private:
                continue

            # Find host name
            host_name = "Unknown"
            for player in lobby.players.values():
                if player.is_host:
                    host_name = player.name
                    break

            result.append(
                LobbyInfo(
                    id=lobby.id,
                    name=lobby.name,
                    host_name=host_name,
                    status=lobby.status,
                    current_players=len(lobby.players),
                    max_players=lobby.settings.max_players,
                    created_at=lobby.created_at,
                )
            )

        return result

    async def delete_lobby(self, lobby_id: str) -> bool:
        """Delete a lobby.

        Args:
            lobby_id: Lobby to delete

        Returns:
            True if deleted
        """
        async with self._lock:
            if lobby_id in self.lobbies:
                del self.lobbies[lobby_id]
                return True
            return False

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        self.lobbies.clear()

    def lobby_count(self) -> int:
        """Get number of active lobbies."""
        return len(self.lobbies)

    async def _broadcast_lobby_event(
        self,
        lobby_id: str,
        event_type: LobbyWSMessageType,
        data: dict[str, Any],
    ) -> None:
        """Broadcast event to all lobby participants.

        Args:
            lobby_id: Target lobby
            event_type: Type of event
            data: Event data
        """
        if self._connection_manager is None:
            return

        # Use the lobby_id as the "game_id" for the connection manager
        # Lobby connections use the same broadcast mechanism
        await self._connection_manager.broadcast_to_game(
            f"lobby:{lobby_id}",
            WSMessage(
                type=WSMessageType.STATE_UPDATE,  # Reuse state_update type
                data={"lobby_event": event_type.value, **data},
            ),
        )


