"""
Pydantic models for player sessions and authentication.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request to create a new player session."""

    display_name: str = Field(
        default="Player",
        min_length=1,
        max_length=32,
        description="Display name for the player",
    )


class SessionResponse(BaseModel):
    """Response containing session information."""

    session_id: str
    display_name: str
    created_at: datetime
    current_game_id: str | None = None
    current_player_id: int | None = None
    current_lobby_id: str | None = None


class UpdatePlayerRequest(BaseModel):
    """Request to update player settings."""

    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description="New display name for the player",
    )


class PlayerInfo(BaseModel):
    """Public player information."""

    session_id: str
    display_name: str
    is_connected: bool = True
