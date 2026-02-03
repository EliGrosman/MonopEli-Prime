"""
FastAPI dependency injection for the MonopEli API.

This module provides dependency functions that can be used with FastAPI's
Depends() mechanism to inject shared services into route handlers.
"""

from fastapi import Request, WebSocket

from .services.broadcast import ConnectionManager
from .services.game_manager import GameManager
from .services.lobby_manager import LobbyManager


def get_game_manager(request: Request) -> GameManager:
    """Get the game manager from application state (for HTTP endpoints).

    Args:
        request: The FastAPI request object

    Returns:
        The GameManager instance
    """
    game_manager: GameManager = request.app.state.game_manager
    return game_manager


def get_connection_manager(request: Request) -> ConnectionManager:
    """Get the connection manager from application state (for HTTP endpoints).

    Args:
        request: The FastAPI request object

    Returns:
        The ConnectionManager instance
    """
    conn_manager: ConnectionManager = request.app.state.connection_manager
    return conn_manager


def get_game_manager_ws(websocket: WebSocket) -> GameManager:
    """Get the game manager from application state (for WebSocket endpoints).

    Args:
        websocket: The FastAPI WebSocket object

    Returns:
        The GameManager instance
    """
    game_manager: GameManager = websocket.app.state.game_manager
    return game_manager


def get_connection_manager_ws(websocket: WebSocket) -> ConnectionManager:
    """Get the connection manager from application state (for WebSocket endpoints).

    Args:
        websocket: The FastAPI WebSocket object

    Returns:
        The ConnectionManager instance
    """
    conn_manager: ConnectionManager = websocket.app.state.connection_manager
    return conn_manager


def get_lobby_manager(request: Request) -> LobbyManager:
    """Get the lobby manager from application state (for HTTP endpoints).

    Args:
        request: The FastAPI request object

    Returns:
        The LobbyManager instance
    """
    lobby_manager: LobbyManager = request.app.state.lobby_manager
    return lobby_manager


def get_lobby_manager_ws(websocket: WebSocket) -> LobbyManager:
    """Get the lobby manager from application state (for WebSocket endpoints).

    Args:
        websocket: The FastAPI WebSocket object

    Returns:
        The LobbyManager instance
    """
    lobby_manager: LobbyManager = websocket.app.state.lobby_manager
    return lobby_manager
