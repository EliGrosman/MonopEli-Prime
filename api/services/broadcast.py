"""
WebSocket connection management and broadcasting.

Handles:
- Connection lifecycle (connect, disconnect, reconnect)
- Message broadcasting to game participants
- Player and spectator tracking
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import WebSocket

from ..models.websocket import WSMessage


@dataclass
class Connection:
    """A WebSocket connection."""

    websocket: WebSocket
    session_id: str
    game_id: str
    player_id: int | None  # None = spectator
    player_name: str = "Unknown"
    connected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = datetime.now(UTC)


class ConnectionManager:
    """Manages WebSocket connections for all games.

    Thread-safe management of WebSocket connections with support for:
    - Multiple connections per game
    - Player and spectator connections
    - Broadcast to all game participants
    - Targeted messages to specific players
    """

    def __init__(self) -> None:
        # game_id -> list of connections
        self._connections: dict[str, list[Connection]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        game_id: str,
        player_id: int | None = None,
        player_name: str = "Unknown",
    ) -> Connection:
        """Register a new WebSocket connection.

        Args:
            websocket: The WebSocket instance
            session_id: Client session ID
            game_id: Game to connect to
            player_id: Player slot (None for spectators)
            player_name: Display name for the player

        Returns:
            Connection object
        """
        await websocket.accept()

        conn = Connection(
            websocket=websocket,
            session_id=session_id,
            game_id=game_id,
            player_id=player_id,
            player_name=player_name,
        )

        async with self._lock:
            if game_id not in self._connections:
                self._connections[game_id] = []
            self._connections[game_id].append(conn)

        return conn

    async def disconnect(self, connection: Connection) -> None:
        """Remove a WebSocket connection.

        Args:
            connection: The connection to remove
        """
        async with self._lock:
            game_conns = self._connections.get(connection.game_id, [])
            if connection in game_conns:
                game_conns.remove(connection)

            # Cleanup empty game lists
            if not game_conns and connection.game_id in self._connections:
                del self._connections[connection.game_id]

    async def broadcast_to_game(
        self,
        game_id: str,
        message: WSMessage,
        exclude_session: str | None = None,
        exclude_player: int | None = None,
    ) -> int:
        """Broadcast message to all connections in a game.

        Args:
            game_id: Target game
            message: Message to send
            exclude_session: Optional session ID to exclude
            exclude_player: Optional player ID to exclude

        Returns:
            Number of connections message was sent to
        """
        async with self._lock:
            connections = list(self._connections.get(game_id, []))

        # Send to all connections (outside lock to avoid blocking)
        data = message.model_dump_json()
        sent_count = 0

        for conn in connections:
            if exclude_session is not None and conn.session_id == exclude_session:
                continue
            if exclude_player is not None and conn.player_id == exclude_player:
                continue

            try:
                await conn.websocket.send_text(data)
                sent_count += 1
            except Exception:
                # Connection may be closed - will be cleaned up
                pass

        return sent_count

    async def send_to_player(
        self,
        game_id: str,
        player_id: int,
        message: WSMessage,
    ) -> bool:
        """Send message to a specific player.

        Args:
            game_id: Target game
            player_id: Target player
            message: Message to send

        Returns:
            True if message was sent to at least one connection
        """
        async with self._lock:
            connections = list(self._connections.get(game_id, []))

        data = message.model_dump_json()
        sent = False

        for conn in connections:
            if conn.player_id == player_id:
                try:
                    await conn.websocket.send_text(data)
                    sent = True
                except Exception:
                    pass

        return sent

    async def send_to_session(
        self,
        session_id: str,
        message: WSMessage,
    ) -> bool:
        """Send message to a specific session across all games.

        Args:
            session_id: Target session
            message: Message to send

        Returns:
            True if message was sent
        """
        async with self._lock:
            all_connections = [
                conn
                for conns in self._connections.values()
                for conn in conns
                if conn.session_id == session_id
            ]

        data = message.model_dump_json()
        sent = False

        for conn in all_connections:
            try:
                await conn.websocket.send_text(data)
                sent = True
            except Exception:
                pass

        return sent

    def get_connection_count(self, game_id: str) -> int:
        """Get total number of connections for a game.

        Args:
            game_id: The game to check

        Returns:
            Number of connections
        """
        return len(self._connections.get(game_id, []))

    def get_player_count(self, game_id: str) -> int:
        """Get number of player connections (non-spectators) for a game.

        Args:
            game_id: The game to check

        Returns:
            Number of player connections
        """
        return sum(
            1
            for conn in self._connections.get(game_id, [])
            if conn.player_id is not None
        )

    def get_spectator_count(self, game_id: str) -> int:
        """Get number of spectators for a game.

        Args:
            game_id: The game to check

        Returns:
            Number of spectator connections
        """
        return sum(
            1
            for conn in self._connections.get(game_id, [])
            if conn.player_id is None
        )

    def is_player_connected(self, game_id: str, player_id: int) -> bool:
        """Check if a player has an active connection.

        Args:
            game_id: The game to check
            player_id: The player to check

        Returns:
            True if player has at least one active connection
        """
        return any(
            conn.player_id == player_id
            for conn in self._connections.get(game_id, [])
        )

    def get_connected_players(self, game_id: str) -> list[int]:
        """Get list of connected player IDs for a game.

        Args:
            game_id: The game to check

        Returns:
            List of player IDs with active connections
        """
        return list(
            {
                conn.player_id
                for conn in self._connections.get(game_id, [])
                if conn.player_id is not None
            }
        )

    async def close_game_connections(self, game_id: str, reason: str = "") -> int:
        """Close all connections for a game.

        Args:
            game_id: The game to close connections for
            reason: Optional reason for closing

        Returns:
            Number of connections closed
        """
        async with self._lock:
            connections = self._connections.pop(game_id, [])

        closed = 0
        for conn in connections:
            try:
                await conn.websocket.close(code=1000, reason=reason)
                closed += 1
            except Exception:
                pass

        return closed

    def total_connections(self) -> int:
        """Get total number of connections across all games.

        Returns:
            Total connection count
        """
        return sum(len(conns) for conns in self._connections.values())

    def total_games_with_connections(self) -> int:
        """Get number of games with at least one connection.

        Returns:
            Number of games with connections
        """
        return len(self._connections)
