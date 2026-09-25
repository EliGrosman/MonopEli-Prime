"""
FastAPI application factory for Monopoly web backend.

Usage:
    # Development
    uvicorn api.main:app --reload --port 8000

    # Production
    uvicorn api.main:create_app --factory --workers 4 --port 8000
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from .config import Settings, get_settings
from .middleware.errors import ErrorHandlingMiddleware
from .middleware.logging import RequestLoggingMiddleware, setup_logging
from .middleware.metrics import MetricsMiddleware, metrics_endpoint
from .middleware.rate_limit import RateLimitMiddleware
from .routers import games, lobbies, players, websocket
from .services.ai_manager import AIManager
from .services.broadcast import ConnectionManager
from .services.game_manager import GameManager
from .services.lobby_manager import LobbyManager
from .services.session_manager import SessionManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown."""
    # Startup
    app.state.start_time = time.time()
    app.state.game_manager = GameManager()
    app.state.connection_manager = ConnectionManager()
    app.state.lobby_manager = LobbyManager()
    configured_settings: Settings = app.state.configured_settings
    app.state.ai_manager = AIManager(settings=configured_settings)
    app.state.session_manager = SessionManager()

    # Link connection manager to game manager for broadcasts
    app.state.game_manager.set_connection_manager(app.state.connection_manager)

    # Link connection manager to lobby manager for broadcasts
    app.state.lobby_manager.set_connection_manager(app.state.connection_manager)

    # Link game manager to lobby manager for game creation
    app.state.lobby_manager.set_game_manager(app.state.game_manager)

    # Link AI manager to game manager for AI turns
    app.state.game_manager.set_ai_manager(app.state.ai_manager)

    # Link AI manager to lobby manager for game start
    app.state.lobby_manager.set_ai_manager(app.state.ai_manager)

    await app.state.game_manager.start_cleanup_task()
    await app.state.session_manager.start_cleanup_task()

    yield

    # Shutdown
    await app.state.game_manager.shutdown()
    await app.state.session_manager.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Optional settings override (for testing)

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    # Set up logging
    setup_logging(
        log_level=settings.log_level,
        json_format=not settings.debug,  # JSON in production, readable in dev
    )

    app = FastAPI(
        title="MonopEli API",
        description="""
# MonopEli API

Multiplayer Monopoly game backend with real-time WebSocket support.

## Features

- **REST API** for game management (create, list, delete games)
- **WebSocket** for real-time gameplay communication
- **Lobby System** for matchmaking and game setup
- **AI Opponents** with multiple difficulty levels
- **Session Management** for player tracking

## Getting Started

1. Create a player session: `POST /api/players/session`
2. Create or join a lobby: `POST /api/lobbies` or `POST /api/lobbies/{id}/join`
3. When ready, start the game from the lobby
4. Connect via WebSocket for real-time gameplay

## WebSocket Protocol

Connect to `/ws/games/{game_id}?session_id={session_id}&player_id={player_id}`

Message types:
- `action`: Send game actions (roll_dice, buy_property, etc.)
- `heartbeat`: Keep connection alive
- `chat`: Send chat messages
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.configured_settings = settings

    # Install error handlers (must be before middleware)
    ErrorHandlingMiddleware.install(app)

    # Metrics middleware (outermost - records all requests)
    app.add_middleware(
        MetricsMiddleware,
        exclude_paths=["/metrics", "/health"],
    )

    # Rate limiting middleware (applied first, before other processing)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
        burst_size=settings.rate_limit_burst,
        exclude_paths=["/health", "/metrics", "/api/docs", "/api/redoc", "/api/openapi.json"],
    )

    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths=["/health", "/metrics"],
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str | float]:
        """Health check endpoint."""
        uptime = time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0.0
        return {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 1),
        }

    # Prometheus metrics endpoint
    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return metrics_endpoint(app)

    # API info endpoint
    @app.get("/api", tags=["info"])
    async def api_info() -> dict[str, str | int]:
        """API information endpoint."""
        game_manager: GameManager = app.state.game_manager
        conn_manager: ConnectionManager = app.state.connection_manager
        ai_manager: AIManager = app.state.ai_manager
        session_manager: SessionManager = app.state.session_manager
        return {
            "name": "MonopEli API",
            "version": "1.0.0",
            "active_games": game_manager.game_count(),
            "websocket_connections": conn_manager.total_connections(),
            "active_ai_agents": ai_manager.agent_count(),
            "active_sessions": session_manager.session_count(),
        }

    # Include routers
    app.include_router(games.router, prefix="/api/games", tags=["games"])
    app.include_router(lobbies.router, prefix="/api/lobbies", tags=["lobbies"])
    app.include_router(players.router, prefix="/api/players", tags=["players"])
    app.include_router(websocket.router, tags=["websocket"])

    # Serve static frontend files if available (for combined single-container deploy)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


# For uvicorn direct run: uvicorn api.main:app --reload
app = create_app()
