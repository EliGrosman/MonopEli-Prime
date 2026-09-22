"""
WebSocket endpoint for real-time game communication.

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

import json
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from ..dependencies import get_connection_manager_ws, get_game_manager_ws
from ..models.action import ActionRequest
from ..models.websocket import (
    WSActionResult,
    WSChatMessage,
    WSError,
    WSIdentity,
    WSMessage,
    WSMessageType,
    WSPlayerEvent,
)
from ..services.broadcast import Connection, ConnectionManager
from ..services.game_manager import GameManager

router = APIRouter()


async def _send_message(websocket: WebSocket, message: WSMessage) -> bool:
    """Send a WebSocket message, handling errors.

    Args:
        websocket: The WebSocket connection
        message: The message to send

    Returns:
        True if sent successfully, False otherwise
    """
    try:
        await websocket.send_text(message.model_dump_json())
        return True
    except Exception:
        return False


async def _send_error(websocket: WebSocket, message: str, code: str | None = None) -> bool:
    """Send an error message.

    Args:
        websocket: The WebSocket connection
        message: The error message
        code: Optional error code

    Returns:
        True if sent successfully
    """
    return await _send_message(
        websocket,
        WSMessage(
            type=WSMessageType.ERROR,
            data=WSError(message=message, code=code).model_dump(),
        ),
    )


async def _handle_action(
    websocket: WebSocket,
    connection: Connection,
    data: dict[str, Any],
    game_manager: GameManager,
) -> None:
    """Handle an action message from the client.

    Args:
        websocket: The WebSocket connection
        connection: The connection object
        data: The action data
        game_manager: The game manager
    """
    # Spectators cannot perform actions
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
        request_id = None
    if connection.player_id is None:
        await _send_message(
            websocket,
            WSMessage(
                type=WSMessageType.ACTION_RESULT,
                data=WSActionResult(
                    success=False,
                    message="Spectators cannot perform actions",
                    request_id=request_id,
                    error_code="SPECTATOR_ACTION",
                ).model_dump(),
            ),
        )
        return

    # Parse action request
    try:
        action_req = ActionRequest(**data)
        action = action_req.to_engine_action(connection.player_id)
    except (ValueError, TypeError) as e:
        await _send_message(
            websocket,
            WSMessage(
                type=WSMessageType.ACTION_RESULT,
                data=WSActionResult(
                    success=False,
                    message=f"Invalid action: {e}",
                    request_id=request_id,
                    error_code="INVALID_ACTION",
                ).model_dump(),
            ),
        )
        return

    active_game = await game_manager.get_game(connection.game_id)
    if active_game is None:
        await _send_error(websocket, "Game not found", code="GAME_NOT_FOUND")
        return
    if active_game.game.rules_id == "foundation-trade-v1" and (
        action_req.contract_version != "decision-contract-v1"
        or action_req.expected_revision is None
        or action_req.request_id is None
    ):
        await _send_message(
            websocket,
            WSMessage(
                type=WSMessageType.ACTION_RESULT,
                data=WSActionResult(
                    success=False,
                    message="Trading games require contract version, revision and request ID",
                    request_id=action_req.request_id,
                    revision=active_game.game.state.revision,
                    error_code="CONTRACT_REQUIRED",
                ).model_dump(),
            ),
        )
        return

    # Execute action (broadcasts state update to OTHER players, not sender)
    success, message = await game_manager.execute_action(
        connection.game_id,
        action,
        broadcast=True,
        exclude_session=connection.session_id,
        expected_revision=action_req.expected_revision,
        request_id=action_req.request_id,
        actor_session_id=connection.session_id,
    )

    # Send result to the client that performed the action FIRST
    await _send_message(
        websocket,
        WSMessage(
            type=WSMessageType.ACTION_RESULT,
            data=WSActionResult(
                success=success,
                message=message,
                request_id=action_req.request_id,
                revision=active_game.game.state.revision,
                error_code=(
                    "STALE_REVISION"
                    if "Stale decision revision" in message
                    else "DUPLICATE_REQUEST"
                    if "Duplicate request" in message
                    else "INVALID_ACTION"
                    if not success
                    else None
                ),
            ).model_dump(),
        ),
    )

    # Then send state update to the sender (so they get action_result before state)
    if success:
        state = await game_manager.get_game_state(connection.game_id)
        if state:
            await _send_message(
                websocket,
                WSMessage(
                    type=WSMessageType.STATE_UPDATE,
                    data=state.model_dump(),
                ),
            )
    elif active_game.game.rules_id == "foundation-trade-v1":
        state = await game_manager.get_game_state(connection.game_id)
        if state:
            await _send_message(
                websocket,
                WSMessage(type=WSMessageType.STATE_UPDATE, data=state.model_dump()),
            )

    # Note: Game over is communicated via state_update


async def _handle_chat(
    connection: Connection,
    data: dict[str, Any],
    conn_manager: ConnectionManager,
) -> None:
    """Handle a chat message from the client.

    Args:
        connection: The connection object
        data: The chat data
        conn_manager: The connection manager
    """
    message_text = data.get("message", "")
    if not message_text:
        return

    # Truncate long messages
    if len(message_text) > 500:
        message_text = message_text[:500] + "..."

    chat_msg = WSChatMessage(
        player_id=connection.player_id,
        player_name=connection.player_name,
        message=message_text,
    )

    await conn_manager.broadcast_to_game(
        connection.game_id,
        WSMessage(
            type=WSMessageType.CHAT,
            data=chat_msg.model_dump(),
        ),
    )


async def _handle_heartbeat(websocket: WebSocket, connection: Connection) -> None:
    """Handle a heartbeat message.

    Args:
        websocket: The WebSocket connection
        connection: The connection object
    """
    connection.update_heartbeat()
    await _send_message(
        websocket,
        WSMessage(type=WSMessageType.HEARTBEAT_ACK, data={}),
    )


@router.websocket("/ws/games/{game_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    game_id: str,
    session_id: str = Query(..., description="Client session identifier"),
    player_id: int | None = Query(None, description="Player slot to claim (None = spectator)"),
    player_name: str = Query("Unknown", description="Display name"),
    game_manager: GameManager = Depends(get_game_manager_ws),
    conn_manager: ConnectionManager = Depends(get_connection_manager_ws),
) -> None:
    """WebSocket endpoint for real-time game communication.

    Connect to receive game state updates and send game actions.

    Query params:
        session_id: Required. Client session identifier for reconnection.
        player_id: Optional. Player slot to claim. If None, connects as spectator.
        player_name: Optional. Display name for the player.

    Messages from client:
        - action: Execute a game action
        - heartbeat: Keep connection alive
        - chat: Send chat message

    Messages to client:
        - state_update: Full game state
        - action_result: Result of action execution
        - error: Error message
        - player_joined/left/disconnected: Player events
        - game_over: Game ended
    """
    # Accept the connection first
    await websocket.accept()

    # Verify game exists
    game = await game_manager.get_game(game_id)
    if game is None:
        await _send_error(websocket, "Game not found", code="GAME_NOT_FOUND")
        await websocket.close(code=4004, reason="Game not found")
        return

    # Track if this is a reconnection
    is_reconnect = False

    # If player_id is not provided, look it up by session_id
    # (the lobby manager pre-claims slots for players)
    if player_id is None:
        player_id = await game_manager.get_player_id_by_session(game_id, session_id)
        # If still None, this session isn't a player - connect as spectator
        if player_id is not None:
            # Found the player - mark as reconnect since slot is already claimed
            is_reconnect = True

    # Validate player_id if provided
    if player_id is not None:
        if player_id < 0 or player_id >= len(game.player_slots):
            await _send_error(websocket, "Invalid player ID", code="INVALID_PLAYER")
            await websocket.close(code=4001, reason="Invalid player ID")
            return

        # Try to claim the slot (or reconnect to it)
        success, error, is_reconnect = await game_manager.claim_player_slot(
            game_id, player_id, session_id, player_name
        )
        if not success:
            await _send_error(websocket, error, code="SLOT_CLAIMED")
            await websocket.close(code=4003, reason=error)
            return

    # Create connection object (websocket already accepted)
    connection = Connection(
        websocket=websocket,
        session_id=session_id,
        game_id=game_id,
        player_id=player_id,
        player_name=player_name,
    )

    # Register with connection manager
    async with conn_manager._lock:
        if game_id not in conn_manager._connections:
            conn_manager._connections[game_id] = []
        conn_manager._connections[game_id].append(connection)

    # Send identity message first (tells client their player_id)
    await _send_message(
        websocket,
        WSMessage(
            type=WSMessageType.IDENTITY,
            data=WSIdentity(
                player_id=player_id,
                player_name=player_name,
            ).model_dump(),
        ),
    )

    # Send initial state
    state = await game_manager.get_game_state(game_id)
    if state:
        await _send_message(
            websocket,
            WSMessage(
                type=WSMessageType.STATE_UPDATE,
                data=state.model_dump(),
            ),
        )

    # Notify others of player join or reconnection (if player, not spectator)
    if player_id is not None:
        event_type = (
            WSMessageType.PLAYER_RECONNECTED if is_reconnect else WSMessageType.PLAYER_JOINED
        )
        await conn_manager.broadcast_to_game(
            game_id,
            WSMessage(
                type=event_type,
                data=WSPlayerEvent(
                    player_id=player_id,
                    name=player_name,
                ).model_dump(),
            ),
            exclude_session=session_id,
        )

    try:
        while True:
            # Receive message
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON")
                continue

            msg_type = msg.get("type")
            data = msg.get("data", {})

            if msg_type == "heartbeat":
                await _handle_heartbeat(websocket, connection)

            elif msg_type == "action":
                await _handle_action(websocket, connection, data, game_manager)

            elif msg_type == "chat":
                await _handle_chat(connection, data, conn_manager)

            else:
                await _send_error(websocket, f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        pass
    finally:
        # Disconnect from connection manager
        await conn_manager.disconnect(connection)

        # Release player slot
        if player_id is not None:
            await game_manager.release_player_slot(game_id, player_id, session_id)

            # Notify others of disconnect
            await conn_manager.broadcast_to_game(
                game_id,
                WSMessage(
                    type=WSMessageType.PLAYER_DISCONNECTED,
                    data=WSPlayerEvent(
                        player_id=player_id,
                        name=player_name,
                    ).model_dump(),
                ),
            )
