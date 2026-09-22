"""
Tests for API middleware.

Tests:
- RequestLoggingMiddleware
- ErrorHandlingMiddleware
- RateLimitMiddleware
"""

import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.middleware.errors import (
    APIError,
    ConflictError,
    ErrorHandlingMiddleware,
    NotFoundError,
    RateLimitError,
    ValidationError,
    create_error_response,
)
from api.middleware.logging import (
    RequestLoggingMiddleware,
    get_request_id,
    setup_logging,
)
from api.middleware.rate_limit import RateLimiter, RateLimitMiddleware, TokenBucket

# =============================================================================
# Logging Middleware Tests
# =============================================================================


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        setup_logging()  # Should not raise

    def test_setup_logging_debug(self) -> None:
        """Test debug level logging."""
        setup_logging(log_level="DEBUG")

    def test_setup_logging_json(self) -> None:
        """Test JSON format logging."""
        setup_logging(log_level="INFO", json_format=True)


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    @pytest.fixture
    def app_with_logging(self) -> FastAPI:
        """Create app with logging middleware."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        def test_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "healthy"}

        return app

    @pytest.fixture
    def client(self, app_with_logging: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app_with_logging, backend_options={"use_uvloop": True})

    def test_adds_request_id_header(self, client: TestClient) -> None:
        """Test that request ID is added to response headers."""
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_excludes_health_endpoint(self, client: TestClient) -> None:
        """Test that health endpoint is excluded from logging."""
        response = client.get("/health")
        assert response.status_code == 200
        # Health endpoint should still work but without request ID
        # (The middleware excludes it from logging but still processes it)

    def test_request_id_is_unique(self, client: TestClient) -> None:
        """Test that each request gets a unique ID."""
        response1 = client.get("/test")
        response2 = client.get("/test")

        id1 = response1.headers.get("X-Request-ID")
        id2 = response2.headers.get("X-Request-ID")

        assert id1 != id2


class TestGetRequestId:
    """Tests for get_request_id helper."""

    def test_get_request_id_missing(self) -> None:
        """Test get_request_id when not set."""
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state = MagicMock(spec=[])  # No request_id attribute

        result = get_request_id(request)
        assert result == "unknown"


# =============================================================================
# Error Handling Middleware Tests
# =============================================================================


class TestAPIError:
    """Tests for APIError exception."""

    def test_api_error_defaults(self) -> None:
        """Test APIError with default values."""
        error = APIError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.status_code == 500
        assert error.error_code == "INTERNAL_ERROR"
        assert error.details == {}

    def test_api_error_custom(self) -> None:
        """Test APIError with custom values."""
        error = APIError(
            message="Custom error",
            status_code=400,
            error_code="CUSTOM_ERROR",
            details={"field": "value"},
        )

        assert error.message == "Custom error"
        assert error.status_code == 400
        assert error.error_code == "CUSTOM_ERROR"
        assert error.details == {"field": "value"}


class TestNotFoundError:
    """Tests for NotFoundError exception."""

    def test_not_found_error(self) -> None:
        """Test NotFoundError creation."""
        error = NotFoundError("Game", "123")

        assert error.status_code == 404
        assert error.error_code == "NOT_FOUND"
        assert "Game" in error.message
        assert "123" in error.message
        assert error.details["resource"] == "Game"
        assert error.details["id"] == "123"


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error(self) -> None:
        """Test ValidationError creation."""
        error = ValidationError("Invalid input", details={"field": "name"})

        assert error.status_code == 400
        assert error.error_code == "VALIDATION_ERROR"
        assert error.message == "Invalid input"
        assert error.details["field"] == "name"


class TestConflictError:
    """Tests for ConflictError exception."""

    def test_conflict_error(self) -> None:
        """Test ConflictError creation."""
        error = ConflictError("Resource already exists")

        assert error.status_code == 409
        assert error.error_code == "CONFLICT"


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError creation."""
        error = RateLimitError(retry_after=30)

        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.details["retry_after"] == 30


class TestCreateErrorResponse:
    """Tests for create_error_response helper."""

    def test_create_error_response(self) -> None:
        """Test error response creation."""
        response = create_error_response(
            request_id="abc123",
            status_code=404,
            error_code="NOT_FOUND",
            message="Resource not found",
            details={"id": "123"},
        )

        assert response.status_code == 404
        import json

        body = json.loads(response.body)
        assert body["error"]["code"] == "NOT_FOUND"
        assert body["error"]["message"] == "Resource not found"
        assert body["error"]["request_id"] == "abc123"
        assert body["error"]["details"]["id"] == "123"


class TestErrorHandlingMiddleware:
    """Tests for ErrorHandlingMiddleware."""

    @pytest.fixture
    def app_with_errors(self) -> FastAPI:
        """Create app with error handling."""
        app = FastAPI()
        ErrorHandlingMiddleware.install(app)

        @app.get("/api-error")
        def api_error() -> None:
            raise APIError("Custom API error", status_code=400)

        @app.get("/not-found")
        def not_found() -> None:
            raise NotFoundError("Game", "123")

        @app.get("/http-error")
        def http_error() -> None:
            raise HTTPException(status_code=403, detail="Forbidden")

        @app.get("/unexpected")
        def unexpected() -> None:
            raise RuntimeError("Unexpected error")

        @app.get("/ok")
        def ok() -> dict[str, str]:
            return {"status": "ok"}

        return app

    @pytest.fixture
    def client(self, app_with_errors: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(
            app_with_errors,
            raise_server_exceptions=False,
            backend_options={"use_uvloop": True},
        )

    def test_handles_api_error(self, client: TestClient) -> None:
        """Test handling of APIError."""
        response = client.get("/api-error")

        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["error"]["message"] == "Custom API error"

    def test_handles_not_found_error(self, client: TestClient) -> None:
        """Test handling of NotFoundError."""
        response = client.get("/not-found")

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "NOT_FOUND"

    def test_handles_http_exception(self, client: TestClient) -> None:
        """Test handling of HTTP exceptions."""
        response = client.get("/http-error")

        assert response.status_code == 403
        data = response.json()
        assert data["error"]["code"] == "FORBIDDEN"

    def test_handles_unexpected_error(self, client: TestClient) -> None:
        """Test handling of unexpected exceptions."""
        response = client.get("/unexpected")

        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "INTERNAL_ERROR"
        # Should not expose internal error message
        assert "Unexpected error" not in data["error"]["message"]

    def test_normal_response(self, client: TestClient) -> None:
        """Test that normal responses are not affected."""
        response = client.get("/ok")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_initial_tokens(self) -> None:
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(capacity=10)
        assert bucket.tokens == 10.0

    def test_consume_success(self) -> None:
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10)
        assert bucket.consume(1) is True
        assert bucket.tokens == 9.0

    def test_consume_failure(self) -> None:
        """Test consumption failure when empty."""
        bucket = TokenBucket(capacity=1)
        assert bucket.consume(1) is True
        assert bucket.consume(1) is False

    def test_refill(self) -> None:
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        bucket.tokens = 0.0

        # Simulate time passing
        bucket.last_update = time.time() - 1.0  # 1 second ago

        # Should refill 10 tokens
        assert bucket.consume(1) is True
        assert bucket.tokens >= 8.0  # At least 9 tokens added, 1 consumed

    def test_retry_after(self) -> None:
        """Test retry_after calculation."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 0.0

        assert bucket.retry_after >= 1


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_first_request(self) -> None:
        """Test that first request is allowed."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        allowed, _ = limiter.is_allowed("192.168.1.1")
        assert allowed is True

    def test_blocks_after_burst(self) -> None:
        """Test blocking after burst is exceeded."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=3)

        # Use up burst
        for _ in range(3):
            allowed, _ = limiter.is_allowed("192.168.1.1")
            assert allowed is True

        # Next request should be blocked
        allowed, retry_after = limiter.is_allowed("192.168.1.1")
        assert allowed is False
        assert retry_after > 0

    def test_different_ips_independent(self) -> None:
        """Test that different IPs have independent limits."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        allowed1, _ = limiter.is_allowed("192.168.1.1")
        allowed2, _ = limiter.is_allowed("192.168.1.2")

        assert allowed1 is True
        assert allowed2 is True

    def test_bucket_count(self) -> None:
        """Test bucket count tracking."""
        limiter = RateLimiter()

        limiter.is_allowed("ip1")
        limiter.is_allowed("ip2")

        assert limiter.bucket_count == 2


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.fixture
    def app_with_rate_limit(self) -> FastAPI:
        """Create app with rate limiting."""
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            burst_size=3,
            exclude_paths=["/health"],
        )

        @app.get("/test")
        def test_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "healthy"}

        return app

    @pytest.fixture
    def client(self, app_with_rate_limit: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app_with_rate_limit, backend_options={"use_uvloop": True})

    def test_allows_normal_requests(self, client: TestClient) -> None:
        """Test that normal requests are allowed."""
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-RateLimit-Remaining" in response.headers

    def test_blocks_after_burst(self, client: TestClient) -> None:
        """Test blocking after burst limit."""
        # Use up burst
        for _ in range(3):
            response = client.get("/test")
            assert response.status_code == 200

        # Next request should be blocked
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_excludes_health(self, client: TestClient) -> None:
        """Test that health endpoint is excluded."""
        # Many requests to health should all succeed
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limit_headers(self, client: TestClient) -> None:
        """Test rate limit headers are present."""
        response = client.get("/test")

        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Limit" in response.headers
        assert int(response.headers["X-RateLimit-Limit"]) == 60


# =============================================================================
# Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for all middleware together."""

    @pytest.fixture
    def app_with_all_middleware(self) -> FastAPI:
        """Create app with all middleware."""
        app = FastAPI()

        # Install error handling
        ErrorHandlingMiddleware.install(app)

        # Add rate limiting
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            burst_size=5,
        )

        # Add logging
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        def test_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/error")
        def error_endpoint() -> None:
            raise NotFoundError("Resource", "123")

        return app

    @pytest.fixture
    def client(self, app_with_all_middleware: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(
            app_with_all_middleware,
            raise_server_exceptions=False,
            backend_options={"use_uvloop": True},
        )

    def test_successful_request(self, client: TestClient) -> None:
        """Test successful request with all middleware."""
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_error_with_request_id(self, client: TestClient) -> None:
        """Test error response includes request ID."""
        response = client.get("/error")

        assert response.status_code == 404
        data = response.json()
        # Request ID should be in response (though may be "unknown" in tests)
        assert "request_id" in data["error"]
