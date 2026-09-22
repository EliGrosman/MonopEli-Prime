"""
Global error handling middleware.

Provides consistent error responses across the API with:
- Structured error format
- Proper HTTP status codes
- Error logging
- Request ID tracking
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Create logger for this module
logger = logging.getLogger("api.errors")


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(
        self, resource: str, resource_id: str, details: dict[str, Any] | None = None
    ) -> None:
        """Initialize the not found error."""
        super().__init__(
            message=f"{resource} with id '{resource_id}' not found",
            status_code=404,
            error_code="NOT_FOUND",
            details=details or {"resource": resource, "id": resource_id},
        )


class ValidationError(APIError):
    """Request validation error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the validation error."""
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details or {},
        )


class ConflictError(APIError):
    """Resource conflict error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the conflict error."""
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT",
            details=details or {},
        )


class RateLimitError(APIError):
    """Rate limit exceeded error."""

    def __init__(self, retry_after: int = 60) -> None:
        """Initialize the rate limit error."""
        super().__init__(
            message="Rate limit exceeded. Please try again later.",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after},
        )


def create_error_response(
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Create a standardized error response.

    Args:
        request_id: The request ID for tracing
        status_code: HTTP status code
        error_code: Machine-readable error code
        message: Human-readable error message
        details: Additional error details

    Returns:
        JSONResponse with error information
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "request_id": request_id,
            }
        },
    )


class ErrorHandlingMiddleware:
    """Middleware-like exception handlers for FastAPI.

    This is implemented as exception handlers rather than middleware
    because FastAPI's exception handling works better this way.
    """

    @staticmethod
    def install(app: FastAPI) -> None:
        """Install exception handlers on the FastAPI app.

        Args:
            app: The FastAPI application
        """

        @app.exception_handler(APIError)
        async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
            """Handle APIError exceptions."""
            request_id = getattr(request.state, "request_id", "unknown")

            logger.warning(
                '{"request_id": "%s", "error_code": "%s", "message": "%s", "status_code": %d}',
                request_id,
                exc.error_code,
                exc.message,
                exc.status_code,
            )

            return create_error_response(
                request_id=request_id,
                status_code=exc.status_code,
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
            )

        @app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(
            request: Request, exc: StarletteHTTPException
        ) -> JSONResponse:
            """Handle HTTP exceptions."""
            request_id = getattr(request.state, "request_id", "unknown")

            logger.warning(
                '{"request_id": "%s", "status_code": %d, "detail": "%s"}',
                request_id,
                exc.status_code,
                exc.detail,
            )

            error_codes = {
                400: "BAD_REQUEST",
                401: "UNAUTHORIZED",
                403: "FORBIDDEN",
                404: "NOT_FOUND",
                405: "METHOD_NOT_ALLOWED",
                429: "RATE_LIMIT_EXCEEDED",
                500: "INTERNAL_ERROR",
            }

            return create_error_response(
                request_id=request_id,
                status_code=exc.status_code,
                error_code=error_codes.get(exc.status_code, "HTTP_ERROR"),
                message=str(exc.detail),
            )

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            """Handle request validation errors."""
            request_id = getattr(request.state, "request_id", "unknown")

            # Extract validation errors
            errors = []
            for error in exc.errors():
                loc = ".".join(str(part) for part in error.get("loc", []))
                errors.append({"field": loc, "message": error.get("msg", "")})

            logger.warning(
                '{"request_id": "%s", "validation_errors": %s}',
                request_id,
                errors,
            )

            return create_error_response(
                request_id=request_id,
                status_code=422,
                error_code="VALIDATION_ERROR",
                message="Request validation failed",
                details={"errors": errors},
            )

        @app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
            """Handle unexpected exceptions."""
            request_id = getattr(request.state, "request_id", "unknown")

            # Log the full exception for debugging
            logger.exception(
                '{"request_id": "%s", "error": "%s", "type": "%s"}',
                request_id,
                str(exc),
                type(exc).__name__,
            )

            return create_error_response(
                request_id=request_id,
                status_code=500,
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
            )
