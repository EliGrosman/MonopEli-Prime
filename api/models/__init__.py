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
from .lobby import (
    AddAIRequest,
    CreateLobbyRequest,
    CreateLobbyResponse,
    JoinLobbyRequest,
    JoinLobbyResponse,
    LobbyInfo,
    LobbyMessage,
    LobbyPlayer,
    LobbySettings,
    LobbyState,
    LobbyStatus,
    LobbyWSMessageType,
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
    # Lobby models
    "AddAIRequest",
    "CreateLobbyRequest",
    "CreateLobbyResponse",
    "JoinLobbyRequest",
    "JoinLobbyResponse",
    "LobbyInfo",
    "LobbyMessage",
    "LobbyPlayer",
    "LobbySettings",
    "LobbyState",
    "LobbyStatus",
    "LobbyWSMessageType",
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
