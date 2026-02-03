"""
Pydantic models for the MonopEli API.
"""

from .action import ActionRequest, ActionResponse
from .game import (
    CreateGameRequest,
    CreateGameResponse,
    GameInfo,
    GameState,
    PlayerSlot,
    PlayerState,
    PropertyState,
)
from .websocket import (
    WSActionRequest,
    WSActionResult,
    WSChatMessage,
    WSError,
    WSGameOver,
    WSMessage,
    WSMessageType,
    WSPlayerEvent,
)

__all__ = [
    # Game models
    "CreateGameRequest",
    "CreateGameResponse",
    "GameInfo",
    "GameState",
    "PlayerSlot",
    "PlayerState",
    "PropertyState",
    # Action models
    "ActionRequest",
    "ActionResponse",
    # WebSocket models
    "WSActionRequest",
    "WSActionResult",
    "WSChatMessage",
    "WSError",
    "WSGameOver",
    "WSMessage",
    "WSMessageType",
    "WSPlayerEvent",
]
