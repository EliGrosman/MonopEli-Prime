"""
Prometheus metrics middleware for monitoring.

Exposes:
- http_requests_total: Counter of requests by method, path, status_code
- http_request_duration_seconds: Histogram of request durations
- monopeli_active_games: Gauge of active games
- monopeli_active_ws_connections: Gauge of WebSocket connections
- monopeli_active_sessions: Gauge of active player sessions
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_GAMES = Gauge(
    "monopeli_active_games",
    "Number of active games",
)

ACTIVE_WS_CONNECTIONS = Gauge(
    "monopeli_active_ws_connections",
    "Number of active WebSocket connections",
)

ACTIVE_SESSIONS = Gauge(
    "monopeli_active_sessions",
    "Number of active player sessions",
)


def _normalize_path(path: str) -> str:
    """Normalize path to prevent high-cardinality labels.

    Replaces UUIDs and numeric IDs with placeholders.
    """
    parts = path.strip("/").split("/")
    normalized = []
    for part in parts:
        # Replace UUID-like segments
        if len(part) == 36 and part.count("-") == 4:
            normalized.append("{id}")
        # Replace numeric IDs
        elif part.isdigit():
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized) if normalized else "/"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that records Prometheus metrics for each request."""

    def __init__(
        self,
        app: Any,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/metrics", "/health"]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        method = request.method
        path = _normalize_path(request.url.path)

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=method,
            path=path,
            status_code=response.status_code,
        ).inc()

        REQUEST_DURATION.labels(
            method=method,
            path=path,
        ).observe(duration)

        return response


def update_gauges(app: FastAPI) -> None:
    """Update gauge metrics from application state."""
    if hasattr(app.state, "game_manager"):
        ACTIVE_GAMES.set(app.state.game_manager.game_count())
    if hasattr(app.state, "connection_manager"):
        ACTIVE_WS_CONNECTIONS.set(app.state.connection_manager.total_connections())
    if hasattr(app.state, "session_manager"):
        ACTIVE_SESSIONS.set(app.state.session_manager.session_count())


def metrics_endpoint(app: FastAPI) -> StarletteResponse:
    """Generate Prometheus metrics response, updating gauges first."""
    update_gauges(app)
    return StarletteResponse(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
