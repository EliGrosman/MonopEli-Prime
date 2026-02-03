"""
FastAPI application factory for Monopoly web backend.

Usage:
    # Development
    uvicorn api.main:app --reload --port 8000

    # Production
    uvicorn api.main:create_app --factory --workers 4 --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
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
    app.state.game_manager = GameManager()
    app.state.connection_manager = ConnectionManager()
    app.state.lobby_manager = LobbyManager()
    app.state.ai_manager = AIManager()
    app.state.session_manager = SessionManager()

    # Link connection manager to game manager for broadcasts
    app.state.game_manager.set_connection_manager(app.state.connection_manager)

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

    app = FastAPI(
        title="MonopEli API",
        description="Multiplayer Monopoly game backend",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
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
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

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

    return app


# For uvicorn direct run: uvicorn api.main:app --reload
app = create_app()
