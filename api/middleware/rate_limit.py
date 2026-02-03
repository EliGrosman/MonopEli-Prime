"""
Rate limiting middleware.

Implements token bucket algorithm for rate limiting with:
- Per-IP limiting
- Configurable limits
- Sliding window approach
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Allows burst traffic while limiting sustained rate.
    """

    capacity: int
    tokens: float = field(default=0.0)
    last_update: float = field(default_factory=time.time)
    refill_rate: float = 1.0  # tokens per second

    def __post_init__(self) -> None:
        """Initialize tokens to capacity."""
        self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limited
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Refill tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def retry_after(self) -> int:
        """Calculate seconds until next token is available."""
        if self.tokens >= 1:
            return 0
        tokens_needed = 1 - self.tokens
        return int(tokens_needed / self.refill_rate) + 1


class RateLimiter:
    """In-memory rate limiter using token bucket algorithm.

    For production, this should be backed by Redis for
    distributed rate limiting across multiple instances.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Sustained request rate limit
            burst_size: Maximum burst size (bucket capacity)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self._buckets: dict[str, TokenBucket] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Clean up every 5 minutes

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check if a request is allowed.

        Args:
            key: The rate limit key (e.g., IP address)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        self._maybe_cleanup()

        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.refill_rate,
            )

        bucket = self._buckets[key]
        if bucket.consume():
            return True, 0
        return False, bucket.retry_after

    def _maybe_cleanup(self) -> None:
        """Clean up old buckets periodically."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        # Remove buckets that have been full for a while (inactive clients)
        cutoff = now - self._cleanup_interval
        to_remove = [
            key
            for key, bucket in self._buckets.items()
            if bucket.last_update < cutoff and bucket.tokens >= bucket.capacity
        ]
        for key in to_remove:
            del self._buckets[key]

    @property
    def bucket_count(self) -> int:
        """Get the number of active buckets."""
        return len(self._buckets)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests.

    Uses token bucket algorithm with configurable limits.
    """

    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """Initialize the rate limit middleware.

        Args:
            app: The FastAPI application
            requests_per_minute: Requests per minute limit
            burst_size: Maximum burst size
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )
        self.exclude_paths = exclude_paths or ["/health", "/api/docs", "/api/redoc"]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request and apply rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            The response from the handler or 429 if rate limited
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        allowed, retry_after = self.limiter.is_allowed(client_ip)

        if not allowed:
            return Response(
                content='{"error": {"code": "RATE_LIMIT_EXCEEDED", '
                f'"message": "Rate limit exceeded", '
                f'"details": {{"retry_after": {retry_after}}}}}}}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Process request
        response: Response = await call_next(request)

        # Add rate limit headers
        bucket = self.limiter._buckets.get(client_ip)
        if bucket:
            response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
            response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)

        return response
