"""
Configuration management via environment variables.

All settings can be overridden via environment variables prefixed with MONOPELI_.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
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

    # Guided Jev provider (server-side only)
    typesafe_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="TYPESAFE_API_KEY",
        exclude=True,
        repr=False,
    )
    jev_enabled: bool = False
    jev_model: str = "jev-1.13.0"
    jev_attempt_timeout_seconds: float = 5.0
    jev_connect_timeout_seconds: float = 2.0
    jev_max_retries: int = 1
    jev_max_concurrent_requests: int = 2
    jev_game_max_requests: int = 2000
    jev_game_max_input_tokens: int = 5_000_000
    jev_game_max_cost_usd: float = 0.25
    jev_process_max_requests: int = 20_000
    jev_process_max_input_tokens: int = 50_000_000
    jev_process_max_cost_usd: float = 3.0

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
