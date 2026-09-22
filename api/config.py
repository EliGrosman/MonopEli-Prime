"""
Configuration management via environment variables.

All settings can be overridden via environment variables prefixed with MONOPELI_.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://192.168.0.30:3000",
        "http://localhost:5173",
        "http://eli-lab:5173",
        "http://192.168.0.30:5173",
    ]

    # Game settings
    max_concurrent_games: int = 1000
    max_players_per_game: int = 8
    game_timeout_minutes: int = 180  # 3 hours max game

    # WebSocket (Phase 3 Week 2)
    ws_heartbeat_interval: int = 30  # seconds
    ws_reconnect_window: int = 60  # seconds to reconnect

    # AI (Phase 3 Week 5)
    ai_think_delay_ms: int = 500  # Artificial delay for AI moves

    # Rate limiting (Phase 3 Week 6)
    rate_limit_per_minute: int = 60  # Requests per minute per IP
    rate_limit_burst: int = 10  # Maximum burst size

    # Future: Redis for scaling
    redis_url: str | None = None

    # Future: Database for persistence
    database_url: str | None = None

    model_config = {"env_prefix": "MONOPELI_", "env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
