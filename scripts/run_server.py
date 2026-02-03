#!/usr/bin/env python3
"""
Development server script for MonopEli API.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --port 8080
    python scripts/run_server.py --reload
"""

import argparse
import uvicorn

from api.config import get_settings


def main() -> None:
    """Run the development server."""
    parser = argparse.ArgumentParser(description="Run MonopEli API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    settings = get_settings()

    print(f"Starting MonopEli API server on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/api/docs")
    print(f"Debug mode: {settings.debug}")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
