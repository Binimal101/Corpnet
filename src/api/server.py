"""FastAPI application for DAC-HRAG.

Provides HTTP endpoints for queries, ingestion, and admin operations.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DAC-HRAG",
        description="Distributed Attributed Community-Hierarchical RAG API",
        version="0.1.0",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


# Default app instance
app = create_app()
