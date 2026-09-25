"""
Pydantic models for lobby/matchmaking system.

Lobbies are waiting rooms where players gather before a game starts.
Players can join, set ready status, and the host can start the game.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class LobbyStatus(str, Enum):
    """Lobby lifecycle status."""

    WAITING = "waiting"  # Waiting for players
    STARTING = "starting"  # Game is being created
    STARTED = "started"  # Game has started
    CLOSED = "closed"  # Lobby closed without starting


class LobbyPlayer(BaseModel):
    """A player in a lobby."""

    slot_id: int  # 0-indexed player slot
    session_id: str
    name: str
    is_host: bool = False
    is_ready: bool = False
    is_ai: bool = False
    ai_type: str | None = None
    joined_at: datetime


class LobbySettings(BaseModel):
    """Configurable settings for a lobby/game."""

    max_players: int = Field(ge=2, le=8, default=4)
    min_players: int = Field(ge=2, le=8, default=2)
    starting_money: int = Field(ge=500, le=5000, default=1500)
    go_salary: int = Field(ge=100, le=500, default=200)
    allow_spectators: bool = True
    private: bool = False  # If true, requires invite code
    rules_id: str = Field(default="foundation-v1", pattern="^(foundation-v1|foundation-trade-v1)$")


class LobbyState(BaseModel):
    """Full lobby state."""

    id: str
    name: str
    host_session_id: str
    status: LobbyStatus
    settings: LobbySettings
    players: list[LobbyPlayer]
    spectator_count: int = 0
    created_at: datetime
    game_id: str | None = None  # Set when game starts
    invite_code: str | None = None  # For private lobbies
    jev_available: bool = False


class LobbyInfo(BaseModel):
    """Summary info for lobby listing (public lobbies)."""

    id: str
    name: str
    host_name: str
    status: LobbyStatus
    current_players: int
    max_players: int
    created_at: datetime


class CreateLobbyRequest(BaseModel):
    """Request to create a new lobby."""

    name: str = Field(min_length=1, max_length=50)
    host_name: str = Field(min_length=1, max_length=20)
    settings: LobbySettings | None = None


class CreateLobbyResponse(BaseModel):
    """Response after creating a lobby."""

    id: str
    name: str
    session_id: str  # Host's session ID
    invite_code: str | None  # For private lobbies
    websocket_url: str
    created_at: datetime


class JoinLobbyRequest(BaseModel):
    """Request to join a lobby."""

    player_name: str = Field(min_length=1, max_length=20)
    invite_code: str | None = None  # Required for private lobbies


class JoinLobbyResponse(BaseModel):
    """Response after joining a lobby."""

    lobby_id: str
    session_id: str
    slot_id: int
    websocket_url: str


class AddAIRequest(BaseModel):
    """Request to add an AI player to the lobby."""

    ai_type: str = "rule_based"  # random, rule_based, aggressive, conservative, jev
    name: str | None = None  # Auto-generated if not provided


class LobbyMessage(BaseModel):
    """Chat message in lobby."""

    session_id: str
    player_name: str
    message: str
    timestamp: datetime


# WebSocket message types for lobby
class LobbyWSMessageType(str, Enum):
    """WebSocket message types for lobby communication."""

    # Client → Server
    READY = "ready"
    UNREADY = "unready"
    CHAT = "chat"
    KICK = "kick"  # Host only
    ADD_AI = "add_ai"  # Host only
    REMOVE_AI = "remove_ai"  # Host only
    UPDATE_SETTINGS = "update_settings"  # Host only
    START_GAME = "start_game"  # Host only

    # Server → Client (must match frontend WSMessageType)
    LOBBY_STATE = "lobby_update"
    PLAYER_JOINED = "lobby_player_joined"
    PLAYER_LEFT = "lobby_player_left"
    PLAYER_READY = "lobby_player_ready"
    PLAYER_UNREADY = "lobby_player_unready"
    PLAYER_KICKED = "lobby_player_kicked"
    AI_ADDED = "lobby_ai_added"
    AI_REMOVED = "lobby_ai_removed"
    SETTINGS_UPDATED = "lobby_settings_changed"
    CHAT_MESSAGE = "chat_message"
    GAME_STARTING = "lobby_game_starting"
    GAME_STARTED = "lobby_game_started"
    ERROR = "error"
    LOBBY_CLOSED = "lobby_closed"
