#!/usr/bin/env python3
"""
Load testing for MonopEli API using Locust.

Usage:
    # Install locust: uv add --dev locust

    # Run load test (web UI):
    uv run locust -f scripts/load_test.py --host http://localhost:8000

    # Run load test (headless):
    uv run locust -f scripts/load_test.py --host http://localhost:8000 \
        --headless -u 100 -r 10 -t 60s

    # Parameters:
    #   -u: Number of users to simulate
    #   -r: Spawn rate (users per second)
    #   -t: Test duration
"""

import json
import random
import uuid

from locust import HttpUser, TaskSet, between, task


class PlayerBehavior(TaskSet):
    """Simulates a player's behavior in the API."""

    session_id: str | None = None
    current_lobby_id: str | None = None
    current_game_id: str | None = None

    def on_start(self) -> None:
        """Called when a user starts. Create a session."""
        self._create_session()

    def _create_session(self) -> None:
        """Create a player session."""
        display_name = f"LoadTest-{uuid.uuid4().hex[:8]}"
        response = self.client.post(
            "/api/players/session",
            json={"display_name": display_name},
        )
        if response.status_code == 200:
            data = response.json()
            self.session_id = data.get("session_id")

    @task(5)
    def list_games(self) -> None:
        """List all active games (common operation)."""
        self.client.get("/api/games")

    @task(5)
    def list_lobbies(self) -> None:
        """List all open lobbies (common operation)."""
        self.client.get("/api/lobbies")

    @task(3)
    def check_health(self) -> None:
        """Check health endpoint (lightweight)."""
        self.client.get("/health")

    @task(2)
    def get_api_info(self) -> None:
        """Get API info."""
        self.client.get("/api")

    @task(2)
    def create_and_delete_game(self) -> None:
        """Create a game and then delete it."""
        # Create game
        response = self.client.post(
            "/api/games",
            json={"num_players": 2},
        )
        if response.status_code == 200:
            game_id = response.json().get("id")
            if game_id:
                # Get game state
                self.client.get(f"/api/games/{game_id}")
                # Delete game
                self.client.delete(f"/api/games/{game_id}")

    @task(2)
    def create_lobby_flow(self) -> None:
        """Create a lobby, update settings, then close it."""
        if not self.session_id:
            return

        # Create lobby
        response = self.client.post(
            "/api/lobbies",
            json={
                "name": f"LoadTest Lobby {uuid.uuid4().hex[:6]}",
                "max_players": 4,
            },
            headers={"X-Session-Id": self.session_id},
        )

        if response.status_code == 200:
            lobby_id = response.json().get("id")
            if lobby_id:
                # Get lobby state
                self.client.get(f"/api/lobbies/{lobby_id}")

                # Toggle ready
                self.client.post(
                    f"/api/lobbies/{lobby_id}/ready",
                    headers={"X-Session-Id": self.session_id},
                )

                # Leave lobby (host leaving closes it)
                self.client.post(
                    f"/api/lobbies/{lobby_id}/leave",
                    headers={"X-Session-Id": self.session_id},
                )

    @task(1)
    def get_player_info(self) -> None:
        """Get current player info."""
        if self.session_id:
            self.client.get(
                "/api/players/me",
                headers={"X-Session-Id": self.session_id},
            )

    @task(1)
    def update_player_name(self) -> None:
        """Update player display name."""
        if self.session_id:
            new_name = f"Updated-{uuid.uuid4().hex[:6]}"
            self.client.put(
                "/api/players/me",
                json={"display_name": new_name},
                headers={"X-Session-Id": self.session_id},
            )


class WebSocketBehavior(TaskSet):
    """Simulates WebSocket connections (via HTTP polling fallback)."""

    session_id: str | None = None
    game_id: str | None = None

    def on_start(self) -> None:
        """Create session and game for WebSocket testing."""
        # Create session
        response = self.client.post(
            "/api/players/session",
            json={"display_name": f"WS-Test-{uuid.uuid4().hex[:8]}"},
        )
        if response.status_code == 200:
            self.session_id = response.json().get("session_id")

        # Create game
        response = self.client.post(
            "/api/games",
            json={"num_players": 2, "player_names": ["Player1", "AI"]},
        )
        if response.status_code == 200:
            self.game_id = response.json().get("id")

    def on_stop(self) -> None:
        """Clean up game on stop."""
        if self.game_id:
            self.client.delete(f"/api/games/{self.game_id}")

    @task(10)
    def get_game_state(self) -> None:
        """Simulate polling for game state (WebSocket fallback)."""
        if self.game_id:
            self.client.get(f"/api/games/{self.game_id}")


class MonopolyPlayer(HttpUser):
    """Standard player that interacts with the API."""

    tasks = [PlayerBehavior]
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    weight = 3  # 3x more common than WebSocket users


class MonopolySpectator(HttpUser):
    """Spectator that mostly reads game state."""

    wait_time = between(0.5, 2)
    weight = 2

    @task(10)
    def list_games(self) -> None:
        """List games frequently."""
        self.client.get("/api/games")

    @task(5)
    def list_lobbies(self) -> None:
        """List lobbies."""
        self.client.get("/api/lobbies")

    @task(1)
    def check_health(self) -> None:
        """Health check."""
        self.client.get("/health")


class MonopolyWebSocketUser(HttpUser):
    """User that simulates WebSocket behavior via HTTP."""

    tasks = [WebSocketBehavior]
    wait_time = between(0.1, 0.5)  # Faster polling
    weight = 1


# Quick test without Locust (run directly)
if __name__ == "__main__":
    import sys

    import requests

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Quick API test against {base_url}")
    print("-" * 40)

    # Test health
    r = requests.get(f"{base_url}/health")
    print(f"Health check: {r.status_code} - {r.json()}")

    # Test API info
    r = requests.get(f"{base_url}/api")
    print(f"API info: {r.status_code} - {r.json()}")

    # Create session
    r = requests.post(
        f"{base_url}/api/players/session",
        json={"display_name": "QuickTest"},
    )
    print(f"Create session: {r.status_code}")
    session_id = r.json().get("session_id") if r.status_code == 200 else None

    # List games
    r = requests.get(f"{base_url}/api/games")
    print(f"List games: {r.status_code} - {len(r.json())} games")

    # Create game
    r = requests.post(
        f"{base_url}/api/games",
        json={"num_players": 2},
    )
    print(f"Create game: {r.status_code}")
    game_id = r.json().get("id") if r.status_code == 200 else None

    if game_id:
        # Get game state
        r = requests.get(f"{base_url}/api/games/{game_id}")
        print(f"Get game state: {r.status_code}")

        # Delete game
        r = requests.delete(f"{base_url}/api/games/{game_id}")
        print(f"Delete game: {r.status_code}")

    print("-" * 40)
    print("Quick test complete!")
