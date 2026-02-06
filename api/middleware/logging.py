"""
Structured request/response logging middleware.

Features:
- Request ID generation for tracing
- Timing information for performance monitoring
- JSON-structured logs for production
- Configurable log levels
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Create logger for this module
logger = logging.getLogger("api.requests")


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure logging for the API.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON format (for production)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    if json_format:
        # JSON format for production (structured logging)
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

    # Configure API logger
    api_logger = logging.getLogger("api")
    api_logger.setLevel(level)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.

    Generates unique request IDs for tracing and logs timing information.
    """

    def __init__(
        self,
        app: Any,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The FastAPI application
            exclude_paths: Paths to exclude from logging (e.g., /health)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health"]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request and log details.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            The response from the handler
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Use upstream request ID if provided (e.g., from Nginx), otherwise generate
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id

        # Record start time
        start_time = time.perf_counter()

        # Log request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            '{"request_id": "%s", "method": "%s", "path": "%s", '
            '"client_ip": "%s", "event": "request_start"}',
            request_id,
            request.method,
            request.url.path,
            client_ip,
        )

        # Process request
        response: Response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log response
        logger.info(
            '{"request_id": "%s", "method": "%s", "path": "%s", '
            '"status_code": %d, "duration_ms": %.2f, "event": "request_end"}',
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response


def get_request_id(request: Request) -> str:
    """Get the request ID from the request state.

    Args:
        request: The FastAPI request

    Returns:
        The request ID or "unknown" if not set
    """
    return getattr(request.state, "request_id", "unknown")
