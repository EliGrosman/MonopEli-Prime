"""
Lobby REST endpoints and WebSocket handler.

Provides endpoints for creating, joining, and managing game lobbies.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from ..dependencies import get_lobby_manager, get_lobby_manager_ws
from ..models.lobby import (
    AddAIRequest,
    CreateLobbyRequest,
    CreateLobbyResponse,
    JoinLobbyRequest,
    JoinLobbyResponse,
    LobbyInfo,
    LobbySettings,
    LobbyState,
)
from ..services.lobby_manager import LobbyManager

router = APIRouter()


# ============================================================================
# REST Endpoints
# ============================================================================


@router.post("", response_model=CreateLobbyResponse, status_code=201)
async def create_lobby(
    request: CreateLobbyRequest,
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> CreateLobbyResponse:
    """Create a new game lobby.

    The creator becomes the host and is automatically added as player 0.
    """
    # Generate a session ID for the host
    host_session_id = str(uuid.uuid4())

    lobby_id, invite_code = await lobby_manager.create_lobby(
        name=request.name,
        host_name=request.host_name,
        host_session_id=host_session_id,
        settings=request.settings,
    )

    return CreateLobbyResponse(
        id=lobby_id,
        name=request.name,
        session_id=host_session_id,
        invite_code=invite_code,
        websocket_url=f"/ws/lobbies/{lobby_id}",
        created_at=datetime.now(UTC),
    )


@router.get("", response_model=list[LobbyInfo])
async def list_lobbies(
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> list[LobbyInfo]:
    """List all public lobbies that are waiting for players."""
    return await lobby_manager.list_lobbies(include_private=False)


@router.get("/{lobby_id}", response_model=LobbyState)
async def get_lobby(
    lobby_id: str,
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> LobbyState:
    """Get lobby state."""
    state = await lobby_manager.get_lobby_state(lobby_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Lobby not found")
    return state


@router.post("/{lobby_id}/join", response_model=JoinLobbyResponse)
async def join_lobby(
    lobby_id: str,
    request: JoinLobbyRequest,
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> JoinLobbyResponse:
    """Join an existing lobby."""
    session_id = str(uuid.uuid4())

    success, message, slot_id = await lobby_manager.join_lobby(
        lobby_id=lobby_id,
        player_name=request.player_name,
        session_id=session_id,
        invite_code=request.invite_code,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return JoinLobbyResponse(
        lobby_id=lobby_id,
        session_id=session_id,
        slot_id=slot_id,
        websocket_url=f"/ws/lobbies/{lobby_id}",
    )


@router.post("/{lobby_id}/leave")
async def leave_lobby(
    lobby_id: str,
    session_id: str = Query(...),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, str]:
    """Leave a lobby."""
    success, message = await lobby_manager.leave_lobby(lobby_id, session_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"status": "left", "message": message}


@router.post("/{lobby_id}/ready")
async def set_ready(
    lobby_id: str,
    session_id: str = Query(...),
    ready: bool = Query(True),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, bool]:
    """Set player ready status."""
    success, message = await lobby_manager.set_ready(lobby_id, session_id, ready)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"ready": ready}


@router.post("/{lobby_id}/ai", status_code=201)
async def add_ai_player(
    lobby_id: str,
    request: AddAIRequest,
    session_id: str = Query(..., description="Host session ID"),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, Any]:
    """Add an AI player to the lobby (host only)."""
    success, message, slot_id = await lobby_manager.add_ai_player(
        lobby_id=lobby_id,
        host_session_id=session_id,
        ai_type=request.ai_type,
        name=request.name,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"slot_id": slot_id, "ai_type": request.ai_type}


@router.delete("/{lobby_id}/ai/{slot_id}")
async def remove_ai_player(
    lobby_id: str,
    slot_id: int,
    session_id: str = Query(..., description="Host session ID"),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, str]:
    """Remove an AI player from the lobby (host only)."""
    success, message = await lobby_manager.remove_ai_player(
        lobby_id=lobby_id,
        host_session_id=session_id,
        slot_id=slot_id,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"status": "removed"}


@router.post("/{lobby_id}/kick/{slot_id}")
async def kick_player(
    lobby_id: str,
    slot_id: int,
    session_id: str = Query(..., description="Host session ID"),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, str]:
    """Kick a player from the lobby (host only)."""
    success, message = await lobby_manager.kick_player(
        lobby_id=lobby_id,
        host_session_id=session_id,
        slot_id=slot_id,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"status": "kicked"}


@router.put("/{lobby_id}/settings")
async def update_settings(
    lobby_id: str,
    settings: LobbySettings,
    session_id: str = Query(..., description="Host session ID"),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, str]:
    """Update lobby settings (host only)."""
    success, message = await lobby_manager.update_settings(
        lobby_id=lobby_id,
        host_session_id=session_id,
        settings=settings,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"status": "updated"}


@router.post("/{lobby_id}/start")
async def start_game(
    lobby_id: str,
    session_id: str = Query(..., description="Host session ID"),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, Any]:
    """Start the game from the lobby (host only)."""
    success, message, game_id = await lobby_manager.start_game(
        lobby_id=lobby_id,
        host_session_id=session_id,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "started",
        "game_id": game_id,
        "websocket_url": f"/ws/games/{game_id}",
    }


@router.delete("/{lobby_id}")
async def delete_lobby(
    lobby_id: str,
    session_id: str = Query(..., description="Host session ID"),
    lobby_manager: LobbyManager = Depends(get_lobby_manager),
) -> dict[str, str]:
    """Delete/close a lobby (host only)."""
    lobby = await lobby_manager.get_lobby(lobby_id)
    if lobby is None:
        raise HTTPException(status_code=404, detail="Lobby not found")

    if lobby.host_session_id != session_id:
        raise HTTPException(status_code=403, detail="Only host can delete lobby")

    await lobby_manager.delete_lobby(lobby_id)
    return {"status": "deleted"}


# ============================================================================
# WebSocket Endpoint
# ============================================================================


async def _send_message(websocket: WebSocket, data: dict[str, Any]) -> bool:
    """Send a WebSocket message."""
    try:
        await websocket.send_json(data)
        return True
    except Exception:
        return False


async def _send_error(websocket: WebSocket, message: str) -> bool:
    """Send error message."""
    return await _send_message(
        websocket,
        {"type": "error", "data": {"message": message}},
    )


@router.websocket("/ws/{lobby_id}")
async def lobby_websocket(
    websocket: WebSocket,
    lobby_id: str,
    session_id: str = Query(...),
    lobby_manager: LobbyManager = Depends(get_lobby_manager_ws),
) -> None:
    """WebSocket endpoint for lobby real-time updates.

    Connect to receive lobby state updates and send lobby actions.

    Query params:
        session_id: Your session ID (from join response)

    Messages from client:
        - ready/unready: Toggle ready status
        - chat: Send chat message
        - kick: Kick a player (host only)
        - add_ai/remove_ai: Manage AI players (host only)
        - update_settings: Change settings (host only)
        - start_game: Start the game (host only)

    Messages to client:
        - lobby_state: Full lobby state
        - player_joined/left: Player events
        - player_ready/unready: Ready status changes
        - game_starting/started: Game lifecycle
        - error: Error messages
    """
    await websocket.accept()

    # Verify lobby exists
    lobby = await lobby_manager.get_lobby(lobby_id)
    if lobby is None:
        await _send_error(websocket, "Lobby not found")
        await websocket.close(code=4004)
        return

    # Verify session is in lobby
    player_slot = None
    is_host = False
    for slot_id, player in lobby.players.items():
        if player.session_id == session_id:
            player_slot = slot_id
            is_host = player.is_host
            break

    if player_slot is None:
        await _send_error(websocket, "Not a member of this lobby")
        await websocket.close(code=4003)
        return

    # Send initial state
    state = await lobby_manager.get_lobby_state(lobby_id)
    if state:
        await _send_message(
            websocket,
            {"type": "lobby_state", "data": state.model_dump(mode="json")},
        )

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON")
                continue

            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            # Handle message types
            if msg_type == "ready":
                success, error = await lobby_manager.set_ready(
                    lobby_id, session_id, True
                )
                if not success:
                    await _send_error(websocket, error)

            elif msg_type == "unready":
                success, error = await lobby_manager.set_ready(
                    lobby_id, session_id, False
                )
                if not success:
                    await _send_error(websocket, error)

            elif msg_type == "chat":
                # Broadcast chat message (simplified - no persistence)
                # Note: Would need broadcast implementation for full feature
                _ = {
                    "type": "chat_message",
                    "data": {
                        "slot_id": player_slot,
                        "message": data.get("message", "")[:500],
                    },
                }

            elif msg_type == "kick":
                if not is_host:
                    await _send_error(websocket, "Only host can kick players")
                    continue
                slot = data.get("slot_id")
                if slot is not None:
                    success, error = await lobby_manager.kick_player(
                        lobby_id, session_id, slot
                    )
                    if not success:
                        await _send_error(websocket, error)

            elif msg_type == "add_ai":
                if not is_host:
                    await _send_error(websocket, "Only host can add AI")
                    continue
                success, error, _ = await lobby_manager.add_ai_player(
                    lobby_id,
                    session_id,
                    ai_type=data.get("ai_type", "rule_based"),
                    name=data.get("name"),
                )
                if not success:
                    await _send_error(websocket, error)

            elif msg_type == "remove_ai":
                if not is_host:
                    await _send_error(websocket, "Only host can remove AI")
                    continue
                slot = data.get("slot_id")
                if slot is not None:
                    success, error = await lobby_manager.remove_ai_player(
                        lobby_id, session_id, slot
                    )
                    if not success:
                        await _send_error(websocket, error)

            elif msg_type == "update_settings":
                if not is_host:
                    await _send_error(websocket, "Only host can update settings")
                    continue
                try:
                    settings = LobbySettings(**data)
                    success, error = await lobby_manager.update_settings(
                        lobby_id, session_id, settings
                    )
                    if not success:
                        await _send_error(websocket, error)
                except Exception as e:
                    await _send_error(websocket, f"Invalid settings: {e}")

            elif msg_type == "start_game":
                if not is_host:
                    await _send_error(websocket, "Only host can start the game")
                    continue
                success, error, game_id = await lobby_manager.start_game(
                    lobby_id, session_id
                )
                if not success:
                    await _send_error(websocket, error)
                else:
                    # Send game started message directly
                    await _send_message(
                        websocket,
                        {
                            "type": "game_started",
                            "data": {
                                "game_id": game_id,
                                "websocket_url": f"/ws/games/{game_id}",
                            },
                        },
                    )

            else:
                await _send_error(websocket, f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        pass
    finally:
        # Leave lobby on disconnect
        await lobby_manager.leave_lobby(lobby_id, session_id)
