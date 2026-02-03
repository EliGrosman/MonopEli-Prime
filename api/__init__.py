"""
MonopEli FastAPI Web Backend.

This package provides the HTTP/WebSocket API for the Monopoly game engine.
"""

from .main import create_app

__all__ = ["create_app"]
