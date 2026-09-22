"""
Tests for WebSocket endpoint and related services.

Tests cover:
- WebSocket connection lifecycle
- Message handling (actions, heartbeat, chat)
- Broadcasting
- Connection manager
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services.broadcast import ConnectionManager
from api.services.game_manager import GameManager

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client with lifespan."""
    with TestClient(app, backend_options={"use_uvloop": True}) as client:
        yield client


@pytest.fixture
def game_manager():
    """Create standalone game manager for unit tests."""
    return GameManager()


@pytest.fixture
def connection_manager():
    """Create standalone connection manager for unit tests."""
    return ConnectionManager()


def create_game(client: TestClient, num_players: int = 2) -> str:
    """Helper to create a game and return its ID."""
    response = client.post("/api/games", json={"num_players": num_players})
    return response.json()["id"]


# ============================================================================
# WebSocket Connection Tests
# ============================================================================


class TestWebSocketConnection:
    """Tests for WebSocket connection lifecycle."""

    def test_connect_as_player(self, client: TestClient):
        """Test connecting as a player."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test-session&player_id=0&player_name=Alice"
        ) as websocket:
            # Should receive identity first
            identity = websocket.receive_json()
            assert identity["type"] == "identity"

            # Then initial state
            data = websocket.receive_json()
            assert data["type"] == "state_update"
            assert "players" in data["data"]

    def test_connect_as_spectator(self, client: TestClient):
        """Test connecting as a spectator (no player_id)."""
        game_id = create_game(client)

        with client.websocket_connect(f"/ws/games/{game_id}?session_id=test-session") as websocket:
            # Should receive identity first
            identity = websocket.receive_json()
            assert identity["type"] == "identity"

            # Then initial state
            data = websocket.receive_json()
            assert data["type"] == "state_update"

    def test_connect_to_nonexistent_game(self, client: TestClient):
        """Test connecting to a game that doesn't exist."""
        with client.websocket_connect("/ws/games/nonexistent-game-id?session_id=test") as websocket:
            # Should receive error then close
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "not found" in data["data"]["message"].lower()

    def test_connect_with_invalid_player_id(self, client: TestClient):
        """Test connecting with an invalid player_id."""
        game_id = create_game(client, num_players=2)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test&player_id=99"
        ) as websocket:
            # Should receive error then close
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "invalid" in data["data"]["message"].lower()

    def test_multiple_connections_same_game(self, client: TestClient):
        """Test multiple connections to the same game."""
        game_id = create_game(client, num_players=4)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=session1&player_id=0"
        ) as ws1:
            ws1.receive_json()  # Skip identity
            data1 = ws1.receive_json()
            assert data1["type"] == "state_update"

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=session2&player_id=1"
            ) as ws2:
                # Second connection receives identity + state
                ws2.receive_json()  # Skip identity
                data2 = ws2.receive_json()
                assert data2["type"] == "state_update"

                # First connection receives player_joined notification
                join_msg = ws1.receive_json()
                assert join_msg["type"] == "player_joined"
                assert join_msg["data"]["player_id"] == 1


# ============================================================================
# Heartbeat Tests
# ============================================================================


class TestHeartbeat:
    """Tests for heartbeat/keepalive messages."""

    def test_heartbeat_response(self, client: TestClient):
        """Test that heartbeat receives acknowledgment."""
        game_id = create_game(client)

        with client.websocket_connect(f"/ws/games/{game_id}?session_id=test") as websocket:
            # Skip identity and initial state
            websocket.receive_json()  # identity
            websocket.receive_json()  # state

            # Send heartbeat
            websocket.send_json({"type": "heartbeat"})

            # Should receive ack
            data = websocket.receive_json()
            assert data["type"] == "heartbeat_ack"


# ============================================================================
# Action Tests
# ============================================================================


class TestWebSocketActions:
    """Tests for game actions via WebSocket."""

    def test_roll_dice_action(self, client: TestClient):
        """Test executing a roll dice action."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test&player_id=0"
        ) as websocket:
            # Skip identity and initial state
            websocket.receive_json()  # identity
            websocket.receive_json()  # state

            # Send roll dice action
            websocket.send_json({"type": "action", "data": {"action_type": "roll_dice"}})

            # Should receive action result
            result = websocket.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is True

            # Should also receive state update (broadcast)
            state = websocket.receive_json()
            assert state["type"] == "state_update"

    def test_spectator_cannot_perform_actions(self, client: TestClient):
        """Test that spectators cannot perform actions."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test"  # No player_id = spectator
        ) as websocket:
            # Skip identity and initial state
            websocket.receive_json()  # identity
            websocket.receive_json()  # state

            # Try to send action
            websocket.send_json({"type": "action", "data": {"action_type": "roll_dice"}})

            # Should receive failure
            result = websocket.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is False
            assert "spectator" in result["data"]["message"].lower()

    def test_invalid_action_type(self, client: TestClient):
        """Test sending an invalid action type."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test&player_id=0"
        ) as websocket:
            # Skip identity and initial state
            websocket.receive_json()  # identity
            websocket.receive_json()  # state

            # Send invalid action
            websocket.send_json({"type": "action", "data": {"action_type": "invalid_action"}})

            # Should receive failure
            result = websocket.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is False

    def test_action_validation_failure(self, client: TestClient):
        """Test action that fails validation."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test&player_id=1"  # Player 1, but turn is player 0
        ) as websocket:
            # Skip identity and initial state
            websocket.receive_json()  # identity
            websocket.receive_json()  # state

            # Try to roll dice when it's not our turn (player 0's turn, we're player 1)
            websocket.send_json({"type": "action", "data": {"action_type": "roll_dice"}})

            # Should receive validation failure
            result = websocket.receive_json()
            assert result["type"] == "action_result"
            assert result["data"]["success"] is False
            assert "not your turn" in result["data"]["message"].lower()


# ============================================================================
# Chat Tests
# ============================================================================


class TestWebSocketChat:
    """Tests for chat messages via WebSocket."""

    def test_chat_message(self, client: TestClient):
        """Test sending a chat message."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=session1&player_id=0&player_name=Alice"
        ) as ws1:
            ws1.receive_json()  # Skip identity
            ws1.receive_json()  # Skip state

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=session2&player_id=1&player_name=Bob"
            ) as ws2:
                ws2.receive_json()  # Skip identity
                ws2.receive_json()  # Skip state
                ws1.receive_json()  # Skip player_joined

                # Send chat from player 1
                ws2.send_json({"type": "chat", "data": {"message": "Hello everyone!"}})

                # Both should receive the chat message
                chat1 = ws1.receive_json()
                assert chat1["type"] == "chat"
                assert chat1["data"]["message"] == "Hello everyone!"
                assert chat1["data"]["player_id"] == 1
                assert chat1["data"]["player_name"] == "Bob"

                chat2 = ws2.receive_json()
                assert chat2["type"] == "chat"
                assert chat2["data"]["message"] == "Hello everyone!"

    def test_chat_message_truncation(self, client: TestClient):
        """Test that long chat messages are truncated."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=test&player_id=0&player_name=Test"
        ) as websocket:
            websocket.receive_json()  # Skip identity
            websocket.receive_json()  # Skip state

            # Send very long message
            long_message = "x" * 1000
            websocket.send_json({"type": "chat", "data": {"message": long_message}})

            # Should receive truncated message
            chat = websocket.receive_json()
            assert chat["type"] == "chat"
            assert len(chat["data"]["message"]) <= 503  # 500 + "..."


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestWebSocketErrors:
    """Tests for WebSocket error handling."""

    def test_invalid_json(self, client: TestClient):
        """Test sending invalid JSON."""
        game_id = create_game(client)

        with client.websocket_connect(f"/ws/games/{game_id}?session_id=test") as websocket:
            websocket.receive_json()  # Skip identity
            websocket.receive_json()  # Skip state

            # Send invalid JSON
            websocket.send_text("not valid json")

            # Should receive error
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert "json" in error["data"]["message"].lower()

    def test_unknown_message_type(self, client: TestClient):
        """Test sending unknown message type."""
        game_id = create_game(client)

        with client.websocket_connect(f"/ws/games/{game_id}?session_id=test") as websocket:
            websocket.receive_json()  # Skip identity
            websocket.receive_json()  # Skip state

            # Send unknown type
            websocket.send_json({"type": "unknown_type", "data": {}})

            # Should receive error
            error = websocket.receive_json()
            assert error["type"] == "error"


# ============================================================================
# Connection Manager Unit Tests
# ============================================================================


class TestConnectionManager:
    """Unit tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connection_count(self, connection_manager: ConnectionManager):
        """Test connection counting."""
        assert connection_manager.get_connection_count("game1") == 0
        assert connection_manager.total_connections() == 0

    @pytest.mark.asyncio
    async def test_player_count(self, connection_manager: ConnectionManager):
        """Test player vs spectator counting."""
        assert connection_manager.get_player_count("game1") == 0
        assert connection_manager.get_spectator_count("game1") == 0

    @pytest.mark.asyncio
    async def test_is_player_connected(self, connection_manager: ConnectionManager):
        """Test checking if player is connected."""
        assert connection_manager.is_player_connected("game1", 0) is False

    @pytest.mark.asyncio
    async def test_get_connected_players(self, connection_manager: ConnectionManager):
        """Test getting list of connected players."""
        players = connection_manager.get_connected_players("game1")
        assert players == []

    @pytest.mark.asyncio
    async def test_total_games_with_connections(self, connection_manager: ConnectionManager):
        """Test counting games with connections."""
        assert connection_manager.total_games_with_connections() == 0


# ============================================================================
# Game Manager WebSocket Integration Tests
# ============================================================================


class TestGameManagerWebSocketIntegration:
    """Tests for GameManager with WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_claim_player_slot(self, game_manager: GameManager):
        """Test claiming a player slot."""
        game_id = await game_manager.create_game(num_players=2)

        success, msg, is_reconnect = await game_manager.claim_player_slot(
            game_id, player_id=0, session_id="session1", player_name="Alice"
        )
        assert success is True
        assert is_reconnect is False

        # Verify slot is claimed
        game = await game_manager.get_game(game_id)
        assert game is not None
        assert game.player_slots[0].session_id == "session1"
        assert game.player_slots[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_claim_already_claimed_slot(self, game_manager: GameManager):
        """Test claiming an already claimed slot."""
        game_id = await game_manager.create_game(num_players=2)

        # First claim
        await game_manager.claim_player_slot(game_id, player_id=0, session_id="session1")

        # Second claim by different session
        success, msg, _ = await game_manager.claim_player_slot(
            game_id, player_id=0, session_id="session2"
        )
        assert success is False
        assert "claimed" in msg.lower()

    @pytest.mark.asyncio
    async def test_reclaim_own_slot(self, game_manager: GameManager):
        """Test reclaiming own slot (reconnection)."""
        game_id = await game_manager.create_game(num_players=2)

        # First claim
        await game_manager.claim_player_slot(game_id, player_id=0, session_id="session1")

        # Reclaim same session (not a reconnect since no disconnect)
        success, msg, is_reconnect = await game_manager.claim_player_slot(
            game_id, player_id=0, session_id="session1"
        )
        assert success is True
        assert is_reconnect is False  # Not a reconnect since wasn't marked disconnected

    @pytest.mark.asyncio
    async def test_claim_invalid_player_id(self, game_manager: GameManager):
        """Test claiming an invalid player slot."""
        game_id = await game_manager.create_game(num_players=2)

        success, msg, _ = await game_manager.claim_player_slot(
            game_id, player_id=99, session_id="session1"
        )
        assert success is False
        assert "invalid" in msg.lower()

    @pytest.mark.asyncio
    async def test_claim_slot_nonexistent_game(self, game_manager: GameManager):
        """Test claiming slot in nonexistent game."""
        success, msg, _ = await game_manager.claim_player_slot(
            "nonexistent", player_id=0, session_id="session1"
        )
        assert success is False
        assert "not found" in msg.lower()

    @pytest.mark.asyncio
    async def test_release_player_slot(self, game_manager: GameManager):
        """Test releasing a player slot permanently."""
        game_id = await game_manager.create_game(num_players=2)

        await game_manager.claim_player_slot(game_id, player_id=0, session_id="session1")

        # Permanent release fully clears the slot
        released = await game_manager.release_player_slot(
            game_id, player_id=0, session_id="session1", permanent=True
        )
        assert released is True

        # Verify slot is released
        game = await game_manager.get_game(game_id)
        assert game is not None
        assert game.player_slots[0].session_id is None

    @pytest.mark.asyncio
    async def test_release_slot_wrong_session(self, game_manager: GameManager):
        """Test releasing slot with wrong session."""
        game_id = await game_manager.create_game(num_players=2)

        await game_manager.claim_player_slot(game_id, player_id=0, session_id="session1")

        # Try to release with different session
        released = await game_manager.release_player_slot(
            game_id, player_id=0, session_id="session2"
        )
        assert released is False

    @pytest.mark.asyncio
    async def test_get_player_slot_by_session(self, game_manager: GameManager):
        """Test finding player slot by session."""
        game_id = await game_manager.create_game(num_players=4)

        await game_manager.claim_player_slot(game_id, player_id=2, session_id="session1")

        player_id = await game_manager.get_player_slot_by_session(game_id, "session1")
        assert player_id == 2

    @pytest.mark.asyncio
    async def test_get_player_slot_not_found(self, game_manager: GameManager):
        """Test finding player slot for unknown session."""
        game_id = await game_manager.create_game(num_players=2)

        player_id = await game_manager.get_player_slot_by_session(game_id, "unknown-session")
        assert player_id is None

    @pytest.mark.asyncio
    async def test_set_connection_manager(self, game_manager: GameManager):
        """Test setting connection manager."""
        conn_manager = ConnectionManager()
        game_manager.set_connection_manager(conn_manager)
        assert game_manager._connection_manager is conn_manager


# ============================================================================
# Broadcast Tests
# ============================================================================


class TestBroadcast:
    """Tests for state broadcasting after actions."""

    def test_action_broadcast_to_other_players(self, client: TestClient):
        """Test that actions broadcast state to other players."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=session1&player_id=0"
        ) as ws1:
            ws1.receive_json()  # Skip identity
            ws1.receive_json()  # Skip initial state

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=session2&player_id=1"
            ) as ws2:
                ws2.receive_json()  # Skip identity
                ws2.receive_json()  # Skip initial state
                ws1.receive_json()  # Skip player_joined

                # Player 0 rolls dice
                ws1.send_json({"type": "action", "data": {"action_type": "roll_dice"}})

                # Player 0 receives action result
                result = ws1.receive_json()
                assert result["type"] == "action_result"

                # Both players receive state update
                state1 = ws1.receive_json()
                assert state1["type"] == "state_update"

                state2 = ws2.receive_json()
                assert state2["type"] == "state_update"

    def test_spectator_receives_broadcasts(self, client: TestClient):
        """Test that spectators receive state broadcasts."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=player&player_id=0"
        ) as player_ws:
            player_ws.receive_json()  # Skip identity
            player_ws.receive_json()  # Skip state

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=spectator"
            ) as spectator_ws:
                spectator_ws.receive_json()  # Skip identity
                spectator_ws.receive_json()  # Skip state

                # Player performs action
                player_ws.send_json({"type": "action", "data": {"action_type": "roll_dice"}})

                # Player gets result
                player_ws.receive_json()

                # Both get state update
                player_state = player_ws.receive_json()
                assert player_state["type"] == "state_update"

                spectator_state = spectator_ws.receive_json()
                assert spectator_state["type"] == "state_update"


# ============================================================================
# Disconnect Tests
# ============================================================================


class TestDisconnect:
    """Tests for player disconnection handling."""

    def test_player_disconnect_notification(self, client: TestClient):
        """Test that other players are notified of disconnection."""
        game_id = create_game(client)

        with client.websocket_connect(
            f"/ws/games/{game_id}?session_id=session1&player_id=0"
        ) as ws1:
            ws1.receive_json()  # Skip identity
            ws1.receive_json()  # Skip state

            with client.websocket_connect(
                f"/ws/games/{game_id}?session_id=session2&player_id=1&player_name=Bob"
            ) as ws2:
                ws2.receive_json()  # Skip identity
                ws2.receive_json()  # Skip state
                ws1.receive_json()  # Skip player_joined

            # ws2 closed - ws1 should receive disconnect notification
            disconnect_msg = ws1.receive_json()
            assert disconnect_msg["type"] == "player_disconnected"
            assert disconnect_msg["data"]["player_id"] == 1

    def test_player_slot_released_on_disconnect(self, client: TestClient):
        """Test that player slot is marked disconnected (not fully released)."""
        game_id = create_game(client)

        with client.websocket_connect(f"/ws/games/{game_id}?session_id=session1&player_id=0") as ws:
            ws.receive_json()  # Skip identity
            ws.receive_json()  # Skip state

        # After disconnect, slot should be marked as disconnected (for reconnection)
        # The session_id is kept but disconnected_at is set
        # Check via API
        response = client.get(f"/api/games/{game_id}/players")
        players = response.json()

        # Find player 0
        player_0 = next(p for p in players if p["player_id"] == 0)
        # Session is still held (for reconnection window)
        assert player_0["session_id"] == "session1"
        # But marked as disconnected
        assert player_0["disconnected_at"] is not None
