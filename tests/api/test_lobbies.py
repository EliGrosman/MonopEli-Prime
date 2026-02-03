"""
Tests for lobby system REST endpoints and WebSocket handler.

Covers:
- Lobby creation and deletion
- Joining and leaving lobbies
- Ready status management
- AI player management
- Lobby settings updates
- Game starting from lobby
- WebSocket real-time updates
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.models.lobby import LobbySettings, LobbyStatus


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client with lifespan."""
    with TestClient(app) as client:
        yield client


# ============================================================================
# Lobby Creation Tests
# ============================================================================


class TestCreateLobby:
    """Tests for POST /api/lobbies (create lobby)."""

    def test_create_lobby_basic(self, client: TestClient) -> None:
        """Create a basic lobby with default settings."""
        response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Lobby"
        assert "id" in data
        assert "session_id" in data
        assert data["websocket_url"].startswith("/ws/lobbies/")
        assert "created_at" in data

    def test_create_lobby_with_settings(self, client: TestClient) -> None:
        """Create a lobby with custom settings."""
        response = client.post(
            "/api/lobbies",
            json={
                "name": "Custom Lobby",
                "host_name": "Bob",
                "settings": {
                    "max_players": 6,
                    "starting_money": 2000,
                    "private": True,
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Custom Lobby"
        assert data["invite_code"] is not None  # Private lobbies get invite codes

    def test_create_lobby_name_too_long(self, client: TestClient) -> None:
        """Lobby name must be under 50 characters."""
        response = client.post(
            "/api/lobbies",
            json={"name": "A" * 51, "host_name": "Alice"},
        )
        assert response.status_code == 422

    def test_create_lobby_name_empty(self, client: TestClient) -> None:
        """Lobby name cannot be empty."""
        response = client.post(
            "/api/lobbies",
            json={"name": "", "host_name": "Alice"},
        )
        assert response.status_code == 422

    def test_create_lobby_host_name_too_long(self, client: TestClient) -> None:
        """Host name must be under 20 characters."""
        response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "A" * 21},
        )
        assert response.status_code == 422


# ============================================================================
# Lobby Listing Tests
# ============================================================================


class TestListLobbies:
    """Tests for GET /api/lobbies (list lobbies)."""

    def test_list_empty(self, client: TestClient) -> None:
        """List returns empty when no lobbies exist."""
        response = client.get("/api/lobbies")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_public_lobbies(self, client: TestClient) -> None:
        """List returns public lobbies."""
        # Create two public lobbies
        client.post("/api/lobbies", json={"name": "Lobby 1", "host_name": "Alice"})
        client.post("/api/lobbies", json={"name": "Lobby 2", "host_name": "Bob"})

        response = client.get("/api/lobbies")
        assert response.status_code == 200
        lobbies = response.json()
        assert len(lobbies) == 2
        names = {lobby["name"] for lobby in lobbies}
        assert names == {"Lobby 1", "Lobby 2"}

    def test_list_excludes_private_lobbies(self, client: TestClient) -> None:
        """Private lobbies are not listed."""
        # Create one public, one private
        client.post("/api/lobbies", json={"name": "Public", "host_name": "Alice"})
        client.post(
            "/api/lobbies",
            json={
                "name": "Private",
                "host_name": "Bob",
                "settings": {"private": True},
            },
        )

        response = client.get("/api/lobbies")
        lobbies = response.json()
        assert len(lobbies) == 1
        assert lobbies[0]["name"] == "Public"


# ============================================================================
# Get Lobby State Tests
# ============================================================================


class TestGetLobby:
    """Tests for GET /api/lobbies/{lobby_id}."""

    def test_get_lobby_state(self, client: TestClient) -> None:
        """Get full lobby state."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        response = client.get(f"/api/lobbies/{lobby_id}")
        assert response.status_code == 200
        state = response.json()
        assert state["id"] == lobby_id
        assert state["name"] == "Test Lobby"
        assert state["status"] == "waiting"
        assert len(state["players"]) == 1
        assert state["players"][0]["name"] == "Alice"
        assert state["players"][0]["is_host"] is True

    def test_get_lobby_not_found(self, client: TestClient) -> None:
        """404 for non-existent lobby."""
        response = client.get("/api/lobbies/nonexistent-id")
        assert response.status_code == 404


# ============================================================================
# Join Lobby Tests
# ============================================================================


class TestJoinLobby:
    """Tests for POST /api/lobbies/{lobby_id}/join."""

    def test_join_lobby_success(self, client: TestClient) -> None:
        """Successfully join an existing lobby."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        assert join_response.status_code == 200
        data = join_response.json()
        assert data["lobby_id"] == lobby_id
        assert "session_id" in data
        assert data["slot_id"] == 1  # Second player

    def test_join_private_lobby_with_code(self, client: TestClient) -> None:
        """Join private lobby with invite code."""
        create_response = client.post(
            "/api/lobbies",
            json={
                "name": "Private Lobby",
                "host_name": "Alice",
                "settings": {"private": True},
            },
        )
        lobby_id = create_response.json()["id"]
        invite_code = create_response.json()["invite_code"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob", "invite_code": invite_code},
        )
        assert join_response.status_code == 200

    def test_join_private_lobby_without_code(self, client: TestClient) -> None:
        """Cannot join private lobby without invite code."""
        create_response = client.post(
            "/api/lobbies",
            json={
                "name": "Private Lobby",
                "host_name": "Alice",
                "settings": {"private": True},
            },
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        assert join_response.status_code == 400
        # New error format from ErrorHandlingMiddleware
        error = join_response.json()["error"]
        assert "invite code" in error["message"].lower()

    def test_join_lobby_full(self, client: TestClient) -> None:
        """Cannot join a full lobby."""
        create_response = client.post(
            "/api/lobbies",
            json={
                "name": "Small Lobby",
                "host_name": "Alice",
                "settings": {"max_players": 2},
            },
        )
        lobby_id = create_response.json()["id"]

        # Second player joins
        client.post(f"/api/lobbies/{lobby_id}/join", json={"player_name": "Bob"})

        # Third player cannot join
        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Charlie"},
        )
        assert join_response.status_code == 400
        # New error format from ErrorHandlingMiddleware
        error = join_response.json()["error"]
        assert "full" in error["message"].lower()

    def test_join_nonexistent_lobby(self, client: TestClient) -> None:
        """Cannot join non-existent lobby."""
        join_response = client.post(
            "/api/lobbies/nonexistent/join",
            json={"player_name": "Bob"},
        )
        assert join_response.status_code == 400


# ============================================================================
# Leave Lobby Tests
# ============================================================================


class TestLeaveLobby:
    """Tests for POST /api/lobbies/{lobby_id}/leave."""

    def test_leave_lobby_success(self, client: TestClient) -> None:
        """Successfully leave a lobby."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        leave_response = client.post(
            f"/api/lobbies/{lobby_id}/leave",
            params={"session_id": bob_session},
        )
        assert leave_response.status_code == 200
        assert leave_response.json()["status"] == "left"

    def test_leave_lobby_not_member(self, client: TestClient) -> None:
        """Cannot leave lobby you're not a member of."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        leave_response = client.post(
            f"/api/lobbies/{lobby_id}/leave",
            params={"session_id": "not-a-member"},
        )
        assert leave_response.status_code == 400


# ============================================================================
# Ready Status Tests
# ============================================================================


class TestReadyStatus:
    """Tests for POST /api/lobbies/{lobby_id}/ready."""

    def test_set_ready(self, client: TestClient) -> None:
        """Set player as ready."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        ready_response = client.post(
            f"/api/lobbies/{lobby_id}/ready",
            params={"session_id": session_id, "ready": True},
        )
        assert ready_response.status_code == 200
        assert ready_response.json()["ready"] is True

    def test_set_unready(self, client: TestClient) -> None:
        """Set player as not ready."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        # First set ready
        client.post(
            f"/api/lobbies/{lobby_id}/ready",
            params={"session_id": session_id, "ready": True},
        )

        # Then unready
        ready_response = client.post(
            f"/api/lobbies/{lobby_id}/ready",
            params={"session_id": session_id, "ready": False},
        )
        assert ready_response.status_code == 200
        assert ready_response.json()["ready"] is False


# ============================================================================
# AI Player Tests
# ============================================================================


class TestAIPlayers:
    """Tests for AI player management."""

    def test_add_ai_player(self, client: TestClient) -> None:
        """Host can add an AI player."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        ai_response = client.post(
            f"/api/lobbies/{lobby_id}/ai",
            json={"ai_type": "rule_based", "name": "Bot"},
            params={"session_id": session_id},
        )
        assert ai_response.status_code == 201
        assert ai_response.json()["ai_type"] == "rule_based"
        assert "slot_id" in ai_response.json()

    def test_add_ai_player_non_host(self, client: TestClient) -> None:
        """Non-host cannot add AI players."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        ai_response = client.post(
            f"/api/lobbies/{lobby_id}/ai",
            json={"ai_type": "rule_based"},
            params={"session_id": bob_session},
        )
        assert ai_response.status_code == 400
        # New error format from ErrorHandlingMiddleware
        error = ai_response.json()["error"]
        assert "host" in error["message"].lower()

    def test_remove_ai_player(self, client: TestClient) -> None:
        """Host can remove an AI player."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        # Add AI
        ai_response = client.post(
            f"/api/lobbies/{lobby_id}/ai",
            json={"ai_type": "rule_based"},
            params={"session_id": session_id},
        )
        slot_id = ai_response.json()["slot_id"]

        # Remove AI
        remove_response = client.delete(
            f"/api/lobbies/{lobby_id}/ai/{slot_id}",
            params={"session_id": session_id},
        )
        assert remove_response.status_code == 200
        assert remove_response.json()["status"] == "removed"


# ============================================================================
# Kick Player Tests
# ============================================================================


class TestKickPlayer:
    """Tests for POST /api/lobbies/{lobby_id}/kick/{slot_id}."""

    def test_kick_player(self, client: TestClient) -> None:
        """Host can kick a player."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        host_session = create_response.json()["session_id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_slot = join_response.json()["slot_id"]

        kick_response = client.post(
            f"/api/lobbies/{lobby_id}/kick/{bob_slot}",
            params={"session_id": host_session},
        )
        assert kick_response.status_code == 200
        assert kick_response.json()["status"] == "kicked"

    def test_kick_player_non_host(self, client: TestClient) -> None:
        """Non-host cannot kick players."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        kick_response = client.post(
            f"/api/lobbies/{lobby_id}/kick/0",  # Try to kick host
            params={"session_id": bob_session},
        )
        assert kick_response.status_code == 400


# ============================================================================
# Settings Update Tests
# ============================================================================


class TestUpdateSettings:
    """Tests for PUT /api/lobbies/{lobby_id}/settings."""

    def test_update_settings(self, client: TestClient) -> None:
        """Host can update lobby settings."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        update_response = client.put(
            f"/api/lobbies/{lobby_id}/settings",
            json={"max_players": 6, "starting_money": 2000},
            params={"session_id": session_id},
        )
        assert update_response.status_code == 200

        # Verify settings updated
        state = client.get(f"/api/lobbies/{lobby_id}").json()
        assert state["settings"]["max_players"] == 6
        assert state["settings"]["starting_money"] == 2000

    def test_update_settings_non_host(self, client: TestClient) -> None:
        """Non-host cannot update settings."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        update_response = client.put(
            f"/api/lobbies/{lobby_id}/settings",
            json={"max_players": 6},
            params={"session_id": bob_session},
        )
        assert update_response.status_code == 400


# ============================================================================
# Start Game Tests
# ============================================================================


class TestStartGame:
    """Tests for POST /api/lobbies/{lobby_id}/start."""

    def test_start_game_success(self, client: TestClient) -> None:
        """Host can start game when enough players are ready."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        host_session = create_response.json()["session_id"]

        # Add an AI player to meet minimum
        client.post(
            f"/api/lobbies/{lobby_id}/ai",
            json={"ai_type": "rule_based"},
            params={"session_id": host_session},
        )

        # Host sets ready
        client.post(
            f"/api/lobbies/{lobby_id}/ready",
            params={"session_id": host_session, "ready": True},
        )

        # Start game
        start_response = client.post(
            f"/api/lobbies/{lobby_id}/start",
            params={"session_id": host_session},
        )
        assert start_response.status_code == 200
        data = start_response.json()
        assert data["status"] == "started"
        assert "game_id" in data
        assert data["websocket_url"].startswith("/ws/games/")

    def test_start_game_not_enough_players(self, client: TestClient) -> None:
        """Cannot start game without minimum players."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        host_session = create_response.json()["session_id"]

        start_response = client.post(
            f"/api/lobbies/{lobby_id}/start",
            params={"session_id": host_session},
        )
        assert start_response.status_code == 400
        # New error format from ErrorHandlingMiddleware
        error = start_response.json()["error"]
        assert "player" in error["message"].lower()

    def test_start_game_non_host(self, client: TestClient) -> None:
        """Non-host cannot start game."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        start_response = client.post(
            f"/api/lobbies/{lobby_id}/start",
            params={"session_id": bob_session},
        )
        assert start_response.status_code == 400
        # New error format from ErrorHandlingMiddleware
        error = start_response.json()["error"]
        assert "host" in error["message"].lower()


# ============================================================================
# Delete Lobby Tests
# ============================================================================


class TestDeleteLobby:
    """Tests for DELETE /api/lobbies/{lobby_id}."""

    def test_delete_lobby(self, client: TestClient) -> None:
        """Host can delete lobby."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        delete_response = client.delete(
            f"/api/lobbies/{lobby_id}",
            params={"session_id": session_id},
        )
        assert delete_response.status_code == 200
        assert delete_response.json()["status"] == "deleted"

        # Verify lobby no longer exists
        get_response = client.get(f"/api/lobbies/{lobby_id}")
        assert get_response.status_code == 404

    def test_delete_lobby_non_host(self, client: TestClient) -> None:
        """Non-host cannot delete lobby."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        delete_response = client.delete(
            f"/api/lobbies/{lobby_id}",
            params={"session_id": bob_session},
        )
        assert delete_response.status_code == 403


# ============================================================================
# WebSocket Tests
# ============================================================================


class TestLobbyWebSocket:
    """Tests for WebSocket /ws/lobbies/{lobby_id}."""

    def test_websocket_connect_and_receive_state(self, client: TestClient) -> None:
        """Connect to lobby WebSocket and receive initial state."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        with client.websocket_connect(
            f"/api/lobbies/ws/{lobby_id}?session_id={session_id}"
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "lobby_state"
            assert msg["data"]["id"] == lobby_id
            assert msg["data"]["name"] == "Test Lobby"

    def test_websocket_connect_invalid_lobby(self, client: TestClient) -> None:
        """WebSocket connection returns error for invalid lobby."""
        with client.websocket_connect(
            "/api/lobbies/ws/nonexistent?session_id=test"
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not found" in msg["data"]["message"].lower()

    def test_websocket_connect_not_member(self, client: TestClient) -> None:
        """WebSocket connection returns error if not a lobby member."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        with client.websocket_connect(
            f"/api/lobbies/ws/{lobby_id}?session_id=not-a-member"
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not a member" in msg["data"]["message"].lower()

    def test_websocket_ready_message(self, client: TestClient) -> None:
        """Send ready message via WebSocket."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        session_id = create_response.json()["session_id"]

        with client.websocket_connect(
            f"/api/lobbies/ws/{lobby_id}?session_id={session_id}"
        ) as ws:
            # Receive initial state
            ws.receive_json()

            # Send ready message
            ws.send_json({"type": "ready"})

            # Should receive updated state or acknowledgment
            # (actual behavior depends on implementation)

    def test_websocket_host_only_actions_non_host(self, client: TestClient) -> None:
        """Non-host WebSocket cannot perform host-only actions."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]

        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        bob_session = join_response.json()["session_id"]

        with client.websocket_connect(
            f"/api/lobbies/ws/{lobby_id}?session_id={bob_session}"
        ) as ws:
            # Receive initial state
            ws.receive_json()

            # Try to add AI (host-only)
            ws.send_json({"type": "add_ai", "data": {"ai_type": "rule_based"}})

            # Should receive error
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "host" in msg["data"]["message"].lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestLobbyIntegration:
    """End-to-end lobby flow tests."""

    def test_full_lobby_flow(self, client: TestClient) -> None:
        """Complete flow: create -> join -> ready -> start."""
        # 1. Create lobby
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Game Night", "host_name": "Alice"},
        )
        assert create_response.status_code == 201
        lobby_id = create_response.json()["id"]
        host_session = create_response.json()["session_id"]

        # 2. Another player joins
        join_response = client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )
        assert join_response.status_code == 200
        bob_session = join_response.json()["session_id"]

        # 3. Verify lobby state
        state = client.get(f"/api/lobbies/{lobby_id}").json()
        assert len(state["players"]) == 2

        # 4. Both players ready up
        client.post(
            f"/api/lobbies/{lobby_id}/ready",
            params={"session_id": host_session, "ready": True},
        )
        client.post(
            f"/api/lobbies/{lobby_id}/ready",
            params={"session_id": bob_session, "ready": True},
        )

        # 5. Host starts game
        start_response = client.post(
            f"/api/lobbies/{lobby_id}/start",
            params={"session_id": host_session},
        )
        assert start_response.status_code == 200
        game_id = start_response.json()["game_id"]

        # 6. Verify game exists
        game_response = client.get(f"/api/games/{game_id}")
        assert game_response.status_code == 200

        # 7. Verify lobby status changed
        lobby_state = client.get(f"/api/lobbies/{lobby_id}").json()
        assert lobby_state["status"] == "started"
        assert lobby_state["game_id"] == game_id

    def test_host_leaves_closes_lobby(self, client: TestClient) -> None:
        """When host leaves, the lobby is closed."""
        # Create lobby
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        host_session = create_response.json()["session_id"]

        # Another player joins
        client.post(
            f"/api/lobbies/{lobby_id}/join",
            json={"player_name": "Bob"},
        )

        # Host leaves
        leave_response = client.post(
            f"/api/lobbies/{lobby_id}/leave",
            params={"session_id": host_session},
        )
        assert leave_response.status_code == 200
        assert "closed" in leave_response.json()["message"].lower()

        # Verify lobby is closed/deleted
        state_response = client.get(f"/api/lobbies/{lobby_id}")
        assert state_response.status_code == 404

    def test_all_players_leave_closes_lobby(self, client: TestClient) -> None:
        """Lobby closes when all players leave."""
        create_response = client.post(
            "/api/lobbies",
            json={"name": "Test Lobby", "host_name": "Alice"},
        )
        lobby_id = create_response.json()["id"]
        host_session = create_response.json()["session_id"]

        # Host leaves
        client.post(
            f"/api/lobbies/{lobby_id}/leave",
            params={"session_id": host_session},
        )

        # Lobby should be closed/deleted
        response = client.get(f"/api/lobbies/{lobby_id}")
        # Either 404 (deleted) or status=closed
        assert response.status_code == 404 or response.json()["status"] == "closed"
