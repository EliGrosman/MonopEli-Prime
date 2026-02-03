"""
Player session endpoints.

Provides REST endpoints for player session management including
session creation, retrieval, and updates.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ..dependencies import get_session_manager
from ..models.player import (
    CreateSessionRequest,
    SessionResponse,
    UpdatePlayerRequest,
)
from ..services.session_manager import SessionManager

router = APIRouter()


@router.post("/session", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    sessions: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Create a new player session.

    This is a simple "login" - no password required.
    Returns a session ID to use for authentication in subsequent requests.

    Args:
        request: Session creation request with display name
        sessions: Session manager dependency

    Returns:
        SessionResponse with the new session details
    """
    session = await sessions.create_session(display_name=request.display_name)
    return SessionResponse(
        session_id=session.id,
        display_name=session.display_name,
        created_at=session.created_at,
        current_game_id=session.current_game_id,
        current_player_id=session.current_player_id,
        current_lobby_id=session.current_lobby_id,
    )


@router.get("/me", response_model=SessionResponse)
async def get_current_player(
    x_session_id: str = Header(..., description="Session ID from create_session"),
    sessions: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Get current player information.

    Requires X-Session-Id header with a valid session ID.

    Args:
        x_session_id: Session ID from header
        sessions: Session manager dependency

    Returns:
        SessionResponse with current session details

    Raises:
        HTTPException 401: If session is invalid or expired
    """
    session = await sessions.get_session(x_session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )

    return SessionResponse(
        session_id=session.id,
        display_name=session.display_name,
        created_at=session.created_at,
        current_game_id=session.current_game_id,
        current_player_id=session.current_player_id,
        current_lobby_id=session.current_lobby_id,
    )


@router.put("/me", response_model=SessionResponse)
async def update_current_player(
    request: UpdatePlayerRequest,
    x_session_id: str = Header(..., description="Session ID from create_session"),
    sessions: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Update current player settings.

    Requires X-Session-Id header with a valid session ID.

    Args:
        request: Update request with new settings
        x_session_id: Session ID from header
        sessions: Session manager dependency

    Returns:
        SessionResponse with updated session details

    Raises:
        HTTPException 401: If session is invalid or expired
    """
    session = await sessions.get_session(x_session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )

    # Update display name if provided
    if request.display_name is not None:
        await sessions.update_display_name(x_session_id, request.display_name)
        session = await sessions.get_session(x_session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired during update",
            )

    return SessionResponse(
        session_id=session.id,
        display_name=session.display_name,
        created_at=session.created_at,
        current_game_id=session.current_game_id,
        current_player_id=session.current_player_id,
        current_lobby_id=session.current_lobby_id,
    )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    x_session_id: str = Header(..., description="Session ID from create_session"),
    sessions: SessionManager = Depends(get_session_manager),
) -> None:
    """Delete the current session (logout).

    Args:
        x_session_id: Session ID from header
        sessions: Session manager dependency

    Raises:
        HTTPException 401: If session is invalid
    """
    deleted = await sessions.delete_session(x_session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )
