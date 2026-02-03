"""
FastAPI routers for the MonopEli API.
"""

from . import games, lobbies, players, websocket

__all__ = ["games", "lobbies", "players", "websocket"]
