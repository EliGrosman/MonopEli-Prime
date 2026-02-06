"""
Pydantic models for WebSocket messages.

Protocol:
    Client → Server:
        {"type": "action", "data": {"action_type": "buy_property", ...}}
        {"type": "heartbeat"}
        {"type": "chat", "data": {"message": "Hello!"}}

    Server → Client:
        {"type": "state_update", "data": {...}}
        {"type": "action_result", "data": {"success": true, "message": ""}}
        {"type": "error", "data": {"message": "..."}}
        {"type": "player_joined", "data": {"player_id": 0, "name": "..."}}
        {"type": "game_over", "data": {"winner": 0}}
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class WSMessageType(str, Enum):
    """WebSocket message types."""

    # Client → Server
    ACTION = "action"
    HEARTBEAT = "heartbeat"
    CHAT = "chat"

    # Server → Client
    STATE_UPDATE = "state_update"
    ACTION_RESULT = "action_result"
    ERROR = "error"
    HEARTBEAT_ACK = "heartbeat_ack"
    IDENTITY = "identity"  # Tells client their player_id
    PLAYER_JOINED = "player_joined"
    PLAYER_LEFT = "player_left"
    PLAYER_DISCONNECTED = "player_disconnected"
    PLAYER_RECONNECTED = "player_reconnected"
    GAME_STARTED = "game_started"
    GAME_OVER = "game_over"


class WSMessage(BaseModel):
    """WebSocket message wrapper."""

    type: WSMessageType
    data: dict[str, Any] = {}


class WSActionRequest(BaseModel):
    """Action request from client via WebSocket."""

    action_type: str
    property_position: int | None = None
    trade_id: int | None = None
    trade_data: dict[str, Any] | None = None


class WSActionResult(BaseModel):
    """Result of an action execution."""

    success: bool
    message: str = ""


class WSError(BaseModel):
    """Error message."""

    message: str
    code: str | None = None


class WSPlayerEvent(BaseModel):
    """Player join/leave/disconnect event."""

    player_id: int
    name: str | None = None


class WSGameOver(BaseModel):
    """Game over event."""

    winner: int
    winner_name: str


class WSChatMessage(BaseModel):
    """Chat message."""

    player_id: int | None  # None for spectators
    player_name: str
    message: str


class WSIdentity(BaseModel):
    """Identity message telling client their player_id."""

    player_id: int | None  # None for spectators
    player_name: str
