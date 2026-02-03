"""
Tests for game management REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api.services.game_manager import GameManager


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Test GET /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_api_info(self, client: TestClient):
        """Test GET /api returns API info."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "MonopEli API"
        assert data["version"] == "1.0.0"
        assert "active_games" in data


class TestCreateGame:
    """Tests for POST /api/games endpoint."""

    def test_create_game_default(self, client: TestClient):
        """Test creating a game with default settings."""
        response = client.post("/api/games", json={})
        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["num_players"] == 4
        assert len(data["players"]) == 4
        assert "created_at" in data
        assert "websocket_url" in data

    def test_create_game_with_players(self, client: TestClient):
        """Test creating a game with custom player count."""
        response = client.post("/api/games", json={"num_players": 2})
        assert response.status_code == 201
        data = response.json()

        assert data["num_players"] == 2
        assert len(data["players"]) == 2

    def test_create_game_with_names(self, client: TestClient):
        """Test creating a game with custom player names."""
        names = ["Alice", "Bob", "Charlie"]
        response = client.post(
            "/api/games",
            json={"num_players": 3, "player_names": names},
        )
        assert response.status_code == 201
        data = response.json()

        assert len(data["players"]) == 3
        player_names = [p["name"] for p in data["players"]]
        assert player_names == names

    def test_create_game_with_seed(self, client: TestClient):
        """Test creating a game with a seed."""
        response = client.post("/api/games", json={"num_players": 2, "seed": 42})
        assert response.status_code == 201
        assert "id" in response.json()

    def test_create_game_invalid_player_count_low(self, client: TestClient):
        """Test creating a game with too few players."""
        response = client.post("/api/games", json={"num_players": 1})
        assert response.status_code == 422  # Validation error

    def test_create_game_invalid_player_count_high(self, client: TestClient):
        """Test creating a game with too many players."""
        response = client.post("/api/games", json={"num_players": 9})
        assert response.status_code == 422  # Validation error

    def test_create_game_mismatched_names(self, client: TestClient):
        """Test creating a game with mismatched player names count."""
        response = client.post(
            "/api/games",
            json={"num_players": 4, "player_names": ["Alice", "Bob"]},
        )
        assert response.status_code == 400
        assert "player names" in response.json()["detail"].lower()

    def test_create_game_websocket_url(self, client: TestClient):
        """Test that websocket URL is correctly formatted."""
        response = client.post("/api/games", json={})
        data = response.json()
        game_id = data["id"]
        assert data["websocket_url"] == f"/ws/games/{game_id}"


class TestListGames:
    """Tests for GET /api/games endpoint."""

    def test_list_games_empty(self, client: TestClient):
        """Test listing games when none exist."""
        response = client.get("/api/games")
        assert response.status_code == 200
        # May not be empty if other tests ran, but should be a list
        assert isinstance(response.json(), list)

    def test_list_games_after_create(self, client: TestClient):
        """Test listing games after creating one."""
        # Create a game
        create_resp = client.post("/api/games", json={"num_players": 2})
        game_id = create_resp.json()["id"]

        # List games
        response = client.get("/api/games")
        assert response.status_code == 200
        games = response.json()

        # Find our game
        game_ids = [g["id"] for g in games]
        assert game_id in game_ids

    def test_list_games_format(self, client: TestClient):
        """Test that listed games have correct format."""
        # Create a game
        client.post("/api/games", json={"num_players": 3})

        response = client.get("/api/games")
        games = response.json()
        assert len(games) >= 1

        game = games[-1]
        assert "id" in game
        assert "num_players" in game
        assert "players_joined" in game
        assert "started" in game
        assert "game_over" in game
        assert "created_at" in game


class TestGetGameState:
    """Tests for GET /api/games/{id} endpoint."""

    def test_get_game_state(self, client: TestClient):
        """Test getting game state."""
        # Create a game
        create_resp = client.post("/api/games", json={"num_players": 2})
        game_id = create_resp.json()["id"]

        # Get state
        response = client.get(f"/api/games/{game_id}")
        assert response.status_code == 200
        data = response.json()

        # Check state fields
        assert "players" in data
        assert "properties" in data
        assert "current_player" in data
        assert "turn_number" in data
        assert "houses_remaining" in data
        assert "hotels_remaining" in data
        assert "game_over" in data
        assert "event_log" in data

    def test_get_game_state_players(self, client: TestClient):
        """Test that game state contains correct player info."""
        names = ["Alice", "Bob"]
        create_resp = client.post(
            "/api/games",
            json={"num_players": 2, "player_names": names},
        )
        game_id = create_resp.json()["id"]

        response = client.get(f"/api/games/{game_id}")
        data = response.json()

        assert len(data["players"]) == 2
        for player in data["players"]:
            assert "id" in player
            assert "name" in player
            assert "money" in player
            assert "position" in player
            assert "in_jail" in player
            assert "bankrupt" in player

        # Check initial state
        assert data["players"][0]["money"] == 1500
        assert data["players"][0]["position"] == 0
        assert data["players"][0]["bankrupt"] is False

    def test_get_game_state_not_found(self, client: TestClient):
        """Test getting state for non-existent game."""
        response = client.get("/api/games/nonexistent-game-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_game_state_initial_values(self, client: TestClient):
        """Test initial game state values."""
        create_resp = client.post("/api/games", json={"num_players": 4})
        game_id = create_resp.json()["id"]

        response = client.get(f"/api/games/{game_id}")
        data = response.json()

        assert data["current_player"] == 0
        assert data["turn_number"] == 0
        assert data["houses_remaining"] == 32
        assert data["hotels_remaining"] == 12
        assert data["game_over"] is False
        assert data["winner"] is None


class TestDeleteGame:
    """Tests for DELETE /api/games/{id} endpoint."""

    def test_delete_game(self, client: TestClient):
        """Test deleting a game."""
        # Create a game
        create_resp = client.post("/api/games", json={"num_players": 2})
        game_id = create_resp.json()["id"]

        # Delete it
        response = client.delete(f"/api/games/{game_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_resp = client.get(f"/api/games/{game_id}")
        assert get_resp.status_code == 404

    def test_delete_game_not_found(self, client: TestClient):
        """Test deleting non-existent game."""
        response = client.delete("/api/games/nonexistent-game-id")
        assert response.status_code == 404

    def test_delete_game_idempotent(self, client: TestClient):
        """Test that deleting twice returns 404 second time."""
        # Create and delete a game
        create_resp = client.post("/api/games", json={"num_players": 2})
        game_id = create_resp.json()["id"]
        client.delete(f"/api/games/{game_id}")

        # Try to delete again
        response = client.delete(f"/api/games/{game_id}")
        assert response.status_code == 404


class TestGetPlayers:
    """Tests for GET /api/games/{id}/players endpoint."""

    def test_get_players(self, client: TestClient):
        """Test getting player slots."""
        names = ["Alice", "Bob", "Charlie"]
        create_resp = client.post(
            "/api/games",
            json={"num_players": 3, "player_names": names},
        )
        game_id = create_resp.json()["id"]

        response = client.get(f"/api/games/{game_id}/players")
        assert response.status_code == 200

        players = response.json()
        assert len(players) == 3

        for i, player in enumerate(players):
            assert player["player_id"] == i
            assert player["name"] == names[i]
            assert player["is_ai"] is False
            assert player["is_ready"] is False

    def test_get_players_not_found(self, client: TestClient):
        """Test getting players for non-existent game."""
        response = client.get("/api/games/nonexistent/players")
        assert response.status_code == 404


class TestGameManagerUnit:
    """Unit tests for GameManager service."""

    @pytest.mark.asyncio
    async def test_create_game(self, game_manager: GameManager):
        """Test creating a game via game manager."""
        game_id = await game_manager.create_game(num_players=4)
        assert game_id is not None
        assert len(game_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_get_game(self, game_manager: GameManager):
        """Test getting a game via game manager."""
        game_id = await game_manager.create_game(num_players=2)
        game = await game_manager.get_game(game_id)

        assert game is not None
        assert game.id == game_id
        assert len(game.player_slots) == 2

    @pytest.mark.asyncio
    async def test_get_game_not_found(self, game_manager: GameManager):
        """Test getting non-existent game."""
        game = await game_manager.get_game("nonexistent")
        assert game is None

    @pytest.mark.asyncio
    async def test_get_game_state(self, game_manager: GameManager):
        """Test getting game state via game manager."""
        game_id = await game_manager.create_game(num_players=2)
        state = await game_manager.get_game_state(game_id)

        assert state is not None
        assert len(state.players) == 2
        assert state.turn_number == 0

    @pytest.mark.asyncio
    async def test_delete_game(self, game_manager: GameManager):
        """Test deleting a game via game manager."""
        game_id = await game_manager.create_game(num_players=2)
        deleted = await game_manager.delete_game(game_id)

        assert deleted is True
        assert await game_manager.get_game(game_id) is None

    @pytest.mark.asyncio
    async def test_list_games(self, game_manager: GameManager):
        """Test listing games via game manager."""
        # Create some games
        await game_manager.create_game(num_players=2)
        await game_manager.create_game(num_players=4)

        games = await game_manager.list_games()
        assert len(games) >= 2

    @pytest.mark.asyncio
    async def test_game_count(self, game_manager: GameManager):
        """Test game count."""
        assert game_manager.game_count() == 0

        await game_manager.create_game(num_players=2)
        assert game_manager.game_count() == 1

        await game_manager.create_game(num_players=2)
        assert game_manager.game_count() == 2

    @pytest.mark.asyncio
    async def test_execute_action(self, game_manager: GameManager):
        """Test executing an action."""
        from monopoly_engine.actions import RollDice

        game_id = await game_manager.create_game(num_players=2)
        action = RollDice(player_id=0)

        success, msg = await game_manager.execute_action(game_id, action)
        assert success is True

    @pytest.mark.asyncio
    async def test_execute_action_invalid(self, game_manager: GameManager):
        """Test executing an invalid action."""
        from monopoly_engine.actions import RollDice

        game_id = await game_manager.create_game(num_players=2)
        # Try to roll dice as wrong player
        action = RollDice(player_id=1)

        success, msg = await game_manager.execute_action(game_id, action)
        assert success is False
        assert msg != ""

    @pytest.mark.asyncio
    async def test_execute_action_game_not_found(self, game_manager: GameManager):
        """Test executing action on non-existent game."""
        from monopoly_engine.actions import RollDice

        action = RollDice(player_id=0)
        success, msg = await game_manager.execute_action("nonexistent", action)

        assert success is False
        assert "not found" in msg.lower()
