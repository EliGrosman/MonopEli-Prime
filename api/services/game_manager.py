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
from typing import TYPE_CHECKING, Any, Callable

from monopoly_engine import MonopolyGame

from ..config import get_settings
from ..models.game import GameActivity, GameInfo, GameState, PlayerSlot
from ..models.websocket import WSMessage, WSMessageType
from .activity import describe_activity

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
    owner_session_id: str | None = None
    processed_request_ids: set[str] = field(default_factory=set)
    agent_action_log: list[dict[str, Any]] = field(default_factory=list)

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

    def __init__(self, *, background_ai_scheduling: bool = True) -> None:
        """Initialize the game manager."""
        self.games: dict[str, ActiveGame] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._connection_manager: "ConnectionManager | None" = None
        self._ai_manager: "AIManager | None" = None
        self._background_ai_scheduling = background_ai_scheduling

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

    @staticmethod
    def _public_agent_activity(active_game: ActiveGame) -> list[dict[str, Any]]:
        return [
            {
                "actor": row["actor"],
                "before_revision": row["before_revision"],
                "after_revision": row["after_revision"],
                "source": row.get("source", "human"),
                "summary": row.get("summary", row["action"]["type"]),
                "action_type": row["action"]["type"],
                "fallback_reason": row.get("fallback_reason"),
            }
            for row in active_game.agent_action_log[-20:]
            if row.get("source") in {"jev", "forced", "fallback"}
        ]

    async def create_game(
        self,
        num_players: int = 4,
        player_names: list[str] | None = None,
        seed: int | None = None,
        rules_id: str = "foundation-v1",
        owner_session_id: str | None = None,
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
                rules_id=rules_id,
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
                owner_session_id=owner_session_id,
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
        async with self._lock:
            game = self.games.get(game_id)
            if game is None:
                return None
            inspections = self._ai_manager.public_inspections(game_id) if self._ai_manager else {}
            return GameState.from_engine(
                game.game,
                game.player_slots,
                agent_inspections=inspections,
                agent_activity=self._public_agent_activity(game),
                game_activity=[
                    row["public_activity"]
                    for row in game.agent_action_log[-100:]
                    if "public_activity" in row
                ],
            )

    async def capture_decision(self, game_id: str, player_id: int):
        """Capture one detached public decision under the game lock."""
        async with self._lock:
            active_game = self.games.get(game_id)
            if (
                active_game is None
                or active_game.game.game_over
                or active_game.game.decision_player != player_id
            ):
                return None
            return active_game.game.decision_view(player_id)

    async def capture_public_view(self, game_id: str):
        """Capture the latest detached public view for observation/inspection."""
        async with self._lock:
            active_game = self.games.get(game_id)
            return active_game.game.decision_view(None) if active_game is not None else None

    async def execute_action(
        self,
        game_id: str,
        action: "Action",
        broadcast: bool = True,
        exclude_session: str | None = None,
        expected_revision: int | None = None,
        request_id: str | None = None,
        actor_session_id: str | None = None,
        commit_guard: Callable[[], bool] | None = None,
        action_metadata: dict[str, Any] | None = None,
        post_commit: Callable[[int, int], None] | None = None,
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
            if request_id is not None and request_id in active_game.processed_request_ids:
                return False, "Duplicate request ID"
            if commit_guard is not None and not commit_guard():
                return False, "AI agent was replaced or removed"
            if actor_session_id is not None:
                slot = active_game.player_slots.get(action.player_id)
                if slot is None or slot.session_id != actor_session_id:
                    return False, "Session no longer owns this player slot"
            if (
                expected_revision is not None
                and expected_revision != active_game.game.state.revision
            ):
                return False, (
                    f"Stale decision revision: expected {expected_revision}, "
                    f"current {active_game.game.state.revision}"
                )

            # Validate action
            is_valid, error = action.validate(active_game.game)
            if not is_valid:
                return False, error

            # Execute action
            try:
                before_view = active_game.game.decision_view(None)
                before_revision = active_game.game.state.revision
                result = active_game.game.apply_action(
                    action.player_id, action, expected_revision=expected_revision
                )
                if request_id is not None:
                    active_game.processed_request_ids.add(request_id)
                after_view = active_game.game.decision_view(None)
                if post_commit is not None:
                    post_commit(result.revision, active_game.game.state.turn_number)
                metadata = dict(action_metadata or {})
                try:
                    public_activity = describe_activity(
                        game_id, action, before_view, after_view, result
                    )
                except Exception:
                    # Activity is a public projection of a committed action. A
                    # description failure must never make that action appear rejected.
                    public_activity = GameActivity(
                        id=f"{game_id}:{result.revision}",
                        actor=action.player_id,
                        revision=result.revision,
                        turn_number=before_view.turn_number,
                        action_type=type(action).__name__,
                        summary=f"{before_view.players[action.player_id].name} completed an action",
                        occurred_at=datetime.now(UTC),
                    )
                active_game.agent_action_log.append(
                    {
                        "actor": action.player_id,
                        "request_id": request_id,
                        "before_revision": before_revision,
                        "after_revision": result.revision,
                        "action": action.to_dict(),
                        "events": list(result.events),
                        "structured_events": list(result.structured_events),
                        **metadata,
                        "public_activity": public_activity.model_dump(mode="json"),
                    }
                )
                active_game.agent_action_log = active_game.agent_action_log[-5000:]
                inspections = (
                    self._ai_manager.public_inspections(game_id) if self._ai_manager else {}
                )
                captured_state = GameState.from_engine(
                    active_game.game,
                    active_game.player_slots,
                    agent_inspections=inspections,
                    agent_activity=self._public_agent_activity(active_game),
                    game_activity=[
                        row["public_activity"]
                        for row in active_game.agent_action_log[-100:]
                        if "public_activity" in row
                    ],
                )
            except Exception as e:
                return False, f"Action execution failed: {e}"

        if self._ai_manager is not None:
            self._ai_manager.observe_transition(
                game_id,
                before_view,
                after_view,
                list(result.structured_events),
            )

        # Broadcast state update to other players (outside lock)
        if broadcast and self._connection_manager:
            await self._connection_manager.broadcast_to_game(
                game_id,
                WSMessage(
                    type=WSMessageType.STATE_UPDATE,
                    data=captured_state.model_dump(),
                ),
                exclude_session=exclude_session,
            )

        # Check if we should trigger AI turns
        await self._maybe_process_ai_turns(game_id)

        return True, ""

    async def broadcast_agent_update(self, game_id: str, inspection: dict[str, Any]) -> None:
        """Broadcast non-authoritative agent status without changing game revision."""
        if self._connection_manager is not None:
            await self._connection_manager.broadcast_to_game(
                game_id,
                WSMessage(type=WSMessageType.AGENT_UPDATE, data=inspection),
            )

    async def _maybe_process_ai_turns(self, game_id: str) -> None:
        """Process AI turns if the current player is AI.

        This is called after action execution to check if the next player
        is an AI and should take their turn automatically.

        Args:
            game_id: The game ID
        """
        if self._ai_manager is None:
            return
        if not self._background_ai_scheduling:
            return

        active_game = self.games.get(game_id)
        if active_game is None or active_game.game.game_over:
            return

        current_player = active_game.game.decision_player
        if self._ai_manager.is_ai_player(game_id, current_player):
            self._ai_manager.schedule_ai_turns(self, game_id)

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
            found = self.games.pop(game_id, None) is not None
        if found and self._ai_manager is not None:
            self._ai_manager.remove_game_agents(game_id)
        return found

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
        if self._ai_manager is not None:
            await self._ai_manager.shutdown()
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
            if self._ai_manager is not None:
                for gid in expired:
                    self._ai_manager.remove_game_agents(gid)

    def game_count(self) -> int:
        """Get number of active games.

        Returns:
            Count of active games
        """
        return len(self.games)
