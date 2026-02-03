"""
Pytest fixtures for API tests.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from api.config import Settings
from api.main import create_app
from api.services.game_manager import GameManager


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        debug=True,
        log_level="DEBUG",
        max_concurrent_games=10,
        cors_origins=["http://localhost:3000"],
    )


@pytest.fixture
def app(settings: Settings):
    """Create test FastAPI application."""
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    """Create synchronous test client with lifespan context."""
    # Use context manager to ensure lifespan events fire
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(app) -> AsyncClient:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def game_manager() -> GameManager:
    """Create a standalone game manager for unit tests."""
    return GameManager()
