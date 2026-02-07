"""API module: FastAPI server and CLI."""

from src.api.server import create_app, app

__all__ = ["create_app", "app"]
