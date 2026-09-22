"""
Player session management service.

Handles player session lifecycle including creation, lookup, expiration,
and tracking of which games/lobbies players are in.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class PlayerSession:
    """A player session representing a connected client.

    Sessions track player identity and current game participation.
    They persist across disconnections within the reconnect window.
    """

    id: str
    display_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Current game/lobby info (at most one of each)
    current_game_id: str | None = None
    current_player_id: int | None = None
    current_lobby_id: str | None = None

    # Disconnection tracking for reconnection support
    disconnected_at: datetime | None = None

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now(UTC)
        self.disconnected_at = None

    def mark_disconnected(self) -> None:
        """Mark session as disconnected (for reconnection tracking)."""
        self.disconnected_at = datetime.now(UTC)

    def is_expired(self, expire_hours: int) -> bool:
        """Check if session has expired.

        Args:
            expire_hours: Hours of inactivity before expiration

        Returns:
            True if session is expired
        """
        elapsed = datetime.now(UTC) - self.last_active
        return elapsed > timedelta(hours=expire_hours)

    def can_reconnect(self, window_seconds: int) -> bool:
        """Check if session is within reconnection window.

        Args:
            window_seconds: Seconds allowed for reconnection

        Returns:
            True if reconnection is allowed
        """
        if self.disconnected_at is None:
            return True

        elapsed = datetime.now(UTC) - self.disconnected_at
        return elapsed <= timedelta(seconds=window_seconds)


class SessionManager:
    """Manages player sessions with automatic cleanup.

    Thread-safe session management with support for:
    - Session creation and lookup
    - Game/lobby assignment tracking
    - Automatic expiration of inactive sessions
    - Reconnection within time window
    """

    def __init__(
        self,
        expire_hours: int = 24,
        reconnect_window_seconds: int = 60,
    ) -> None:
        """Initialize session manager.

        Args:
            expire_hours: Hours of inactivity before session expires
            reconnect_window_seconds: Seconds allowed for reconnection
        """
        self.sessions: dict[str, PlayerSession] = {}
        self.expire_hours = expire_hours
        self.reconnect_window = reconnect_window_seconds
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def create_session(self, display_name: str = "Player") -> PlayerSession:
        """Create a new player session.

        Args:
            display_name: Display name for the player

        Returns:
            The created session
        """
        async with self._lock:
            session_id = str(uuid.uuid4())
            session = PlayerSession(id=session_id, display_name=display_name)
            self.sessions[session_id] = session
            return session

    async def get_session(self, session_id: str) -> PlayerSession | None:
        """Get session by ID, updating last_active.

        Automatically removes expired sessions.

        Args:
            session_id: The session ID to look up

        Returns:
            Session if found and not expired, None otherwise
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None

            if session.is_expired(self.expire_hours):
                del self.sessions[session_id]
                return None

            session.touch()
            return session

    async def get_session_no_touch(self, session_id: str) -> PlayerSession | None:
        """Get session by ID without updating last_active.

        Args:
            session_id: The session ID to look up

        Returns:
            Session if found and not expired, None otherwise
        """
        session = self.sessions.get(session_id)
        if session is None:
            return None

        if session.is_expired(self.expire_hours):
            return None

        return session

    async def update_display_name(
        self,
        session_id: str,
        display_name: str,
    ) -> bool:
        """Update a session's display name.

        Args:
            session_id: The session ID
            display_name: New display name

        Returns:
            True if updated, False if session not found
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None or session.is_expired(self.expire_hours):
                return False

            session.display_name = display_name
            session.touch()
            return True

    async def set_current_game(
        self,
        session_id: str,
        game_id: str | None,
        player_id: int | None = None,
    ) -> bool:
        """Set the current game for a session.

        Args:
            session_id: The session ID
            game_id: The game ID (None to clear)
            player_id: The player slot in the game

        Returns:
            True if updated, False if session not found
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None or session.is_expired(self.expire_hours):
                return False

            session.current_game_id = game_id
            session.current_player_id = player_id
            session.touch()
            return True

    async def set_current_lobby(
        self,
        session_id: str,
        lobby_id: str | None,
    ) -> bool:
        """Set the current lobby for a session.

        Args:
            session_id: The session ID
            lobby_id: The lobby ID (None to clear)

        Returns:
            True if updated, False if session not found
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None or session.is_expired(self.expire_hours):
                return False

            session.current_lobby_id = lobby_id
            session.touch()
            return True

    async def mark_disconnected(self, session_id: str) -> bool:
        """Mark a session as disconnected.

        Args:
            session_id: The session ID

        Returns:
            True if marked, False if session not found
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return False

            session.mark_disconnected()
            return True

    async def can_reconnect(self, session_id: str) -> bool:
        """Check if a session can reconnect.

        Args:
            session_id: The session ID

        Returns:
            True if reconnection is allowed
        """
        session = self.sessions.get(session_id)
        if session is None:
            return False

        return session.can_reconnect(self.reconnect_window)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

    async def get_sessions_in_game(self, game_id: str) -> list[PlayerSession]:
        """Get all sessions currently in a game.

        Args:
            game_id: The game ID

        Returns:
            List of sessions in the game
        """
        return [
            s
            for s in self.sessions.values()
            if s.current_game_id == game_id and not s.is_expired(self.expire_hours)
        ]

    async def get_sessions_in_lobby(self, lobby_id: str) -> list[PlayerSession]:
        """Get all sessions currently in a lobby.

        Args:
            lobby_id: The lobby ID

        Returns:
            List of sessions in the lobby
        """
        return [
            s
            for s in self.sessions.values()
            if s.current_lobby_id == lobby_id and not s.is_expired(self.expire_hours)
        ]

    async def cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        async with self._lock:
            expired = [sid for sid, s in self.sessions.items() if s.is_expired(self.expire_hours)]
            for sid in expired:
                del self.sessions[sid]
            return len(expired)

    async def start_cleanup_task(self) -> None:
        """Start background task to clean up expired sessions."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.sessions.clear()

    async def _cleanup_loop(self) -> None:
        """Background task to periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            await self.cleanup_expired()

    def session_count(self) -> int:
        """Get total number of active sessions.

        Returns:
            Count of sessions (including expired until cleanup)
        """
        return len(self.sessions)
