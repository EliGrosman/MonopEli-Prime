"""
Tests for player session endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.services.session_manager import SessionManager


# client fixture comes from conftest.py


@pytest.fixture
def session_manager() -> SessionManager:
    """Create a session manager for unit tests."""
    return SessionManager()


# ============================================================================
# Session Manager Unit Tests
# ============================================================================


class TestSessionManagerUnit:
    """Unit tests for SessionManager."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager: SessionManager):
        """Test creating a new session."""
        session = await session_manager.create_session(display_name="Alice")

        assert session.id is not None
        assert session.display_name == "Alice"
        assert session.current_game_id is None
        assert session.current_lobby_id is None

    @pytest.mark.asyncio
    async def test_create_session_default_name(self, session_manager: SessionManager):
        """Test creating session with default name."""
        session = await session_manager.create_session()

        assert session.display_name == "Player"

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager: SessionManager):
        """Test getting a session by ID."""
        created = await session_manager.create_session(display_name="Bob")

        retrieved = await session_manager.get_session(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.display_name == "Bob"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, session_manager: SessionManager):
        """Test getting a non-existent session."""
        result = await session_manager.get_session("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_display_name(self, session_manager: SessionManager):
        """Test updating display name."""
        session = await session_manager.create_session(display_name="Old Name")

        success = await session_manager.update_display_name(session.id, "New Name")

        assert success is True
        updated = await session_manager.get_session(session.id)
        assert updated is not None
        assert updated.display_name == "New Name"

    @pytest.mark.asyncio
    async def test_set_current_game(self, session_manager: SessionManager):
        """Test setting current game for session."""
        session = await session_manager.create_session()

        success = await session_manager.set_current_game(
            session.id, "game-123", player_id=2
        )

        assert success is True
        updated = await session_manager.get_session(session.id)
        assert updated is not None
        assert updated.current_game_id == "game-123"
        assert updated.current_player_id == 2

    @pytest.mark.asyncio
    async def test_set_current_lobby(self, session_manager: SessionManager):
        """Test setting current lobby for session."""
        session = await session_manager.create_session()

        success = await session_manager.set_current_lobby(session.id, "lobby-456")

        assert success is True
        updated = await session_manager.get_session(session.id)
        assert updated is not None
        assert updated.current_lobby_id == "lobby-456"

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager: SessionManager):
        """Test deleting a session."""
        session = await session_manager.create_session()

        deleted = await session_manager.delete_session(session.id)

        assert deleted is True
        retrieved = await session_manager.get_session(session.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, session_manager: SessionManager):
        """Test deleting a non-existent session."""
        deleted = await session_manager.delete_session("nonexistent")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_session_count(self, session_manager: SessionManager):
        """Test session count."""
        assert session_manager.session_count() == 0

        await session_manager.create_session()
        assert session_manager.session_count() == 1

        await session_manager.create_session()
        assert session_manager.session_count() == 2


# ============================================================================
# Players Router Tests
# ============================================================================


class TestCreateSession:
    """Tests for POST /api/players/session."""

    def test_create_session_success(self, client: TestClient):
        """Test creating a new session."""
        response = client.post(
            "/api/players/session", json={"display_name": "TestPlayer"}
        )

        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["display_name"] == "TestPlayer"
        assert "created_at" in data

    def test_create_session_default_name(self, client: TestClient):
        """Test creating session with default name."""
        response = client.post("/api/players/session", json={})

        assert response.status_code == 201
        data = response.json()
        assert data["display_name"] == "Player"

    def test_create_session_name_too_long(self, client: TestClient):
        """Test creating session with name that's too long."""
        response = client.post(
            "/api/players/session", json={"display_name": "A" * 50}
        )

        assert response.status_code == 422  # Validation error


class TestGetCurrentPlayer:
    """Tests for GET /api/players/me."""

    def test_get_current_player_success(self, client: TestClient):
        """Test getting current player info."""
        # First create a session
        create_resp = client.post(
            "/api/players/session", json={"display_name": "MyPlayer"}
        )
        session_id = create_resp.json()["session_id"]

        # Then get player info
        response = client.get(
            "/api/players/me", headers={"X-Session-Id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["display_name"] == "MyPlayer"

    def test_get_current_player_invalid_session(self, client: TestClient):
        """Test getting player with invalid session."""
        response = client.get(
            "/api/players/me", headers={"X-Session-Id": "invalid"}
        )

        assert response.status_code == 401

    def test_get_current_player_missing_header(self, client: TestClient):
        """Test getting player without session header."""
        response = client.get("/api/players/me")

        assert response.status_code == 422  # Missing required header


class TestUpdateCurrentPlayer:
    """Tests for PUT /api/players/me."""

    def test_update_display_name(self, client: TestClient):
        """Test updating display name."""
        # Create session
        create_resp = client.post(
            "/api/players/session", json={"display_name": "OldName"}
        )
        session_id = create_resp.json()["session_id"]

        # Update name
        response = client.put(
            "/api/players/me",
            headers={"X-Session-Id": session_id},
            json={"display_name": "NewName"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "NewName"

    def test_update_invalid_session(self, client: TestClient):
        """Test updating with invalid session."""
        response = client.put(
            "/api/players/me",
            headers={"X-Session-Id": "invalid"},
            json={"display_name": "Test"},
        )

        assert response.status_code == 401


class TestDeleteSession:
    """Tests for DELETE /api/players/me."""

    def test_delete_session_success(self, client: TestClient):
        """Test deleting a session."""
        # Create session
        create_resp = client.post("/api/players/session", json={})
        session_id = create_resp.json()["session_id"]

        # Delete it
        response = client.delete(
            "/api/players/me", headers={"X-Session-Id": session_id}
        )

        assert response.status_code == 204

        # Verify it's gone
        get_resp = client.get(
            "/api/players/me", headers={"X-Session-Id": session_id}
        )
        assert get_resp.status_code == 401

    def test_delete_invalid_session(self, client: TestClient):
        """Test deleting with invalid session."""
        response = client.delete(
            "/api/players/me", headers={"X-Session-Id": "invalid"}
        )

        assert response.status_code == 401
