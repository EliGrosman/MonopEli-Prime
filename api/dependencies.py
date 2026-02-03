"""
FastAPI dependency injection for the MonopEli API.

This module provides dependency functions that can be used with FastAPI's
Depends() mechanism to inject shared services into route handlers.
"""

from fastapi import Request

from .services.game_manager import GameManager


def get_game_manager(request: Request) -> GameManager:
    """Get the game manager from application state.

    Args:
        request: The FastAPI request object

    Returns:
        The GameManager instance
    """
    game_manager: GameManager = request.app.state.game_manager
    return game_manager
