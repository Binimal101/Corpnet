"""Consumer FastAPI server — read-only HTTP proxy.

Forwards every read request to the producer's FastAPI server over
HTTP.  The producer URL (including any ``base_url`` prefix) is read
from ``consumer/config.yaml`` → ``producer_url``.

Endpoints (relative to *base_url*)::

    GET  /health   — Local health check + producer reachability
    POST /query    — Forward to producer POST /query
    POST /search   — Forward to producer POST /search
    GET  /info     — Forward to producer GET  /stats

Start with::

    python -m consumer.mcp_server      # reads host/port from config
    uvicorn consumer.mcp_server:app    # manual
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve paths relative to *this* file → consumer/
_CONSUMER_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONSUMER_ROOT.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import load_config  # noqa: E402

log = logging.getLogger(__name__)

# ── Load consumer config ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_config_path = os.environ.get(
    "CONSUMER_CONFIG",
    str(_CONSUMER_ROOT / "config.yaml"),
)
log.info("[consumer] Loading config from %s …", _config_path)
_raw_cfg = load_config(_config_path)

_PRODUCER_URL: str = _raw_cfg.get("producer_url", "http://localhost:8000").rstrip("/")

_server_cfg = _raw_cfg.get("server", {})
_HOST: str = _server_cfg.get("host", "0.0.0.0")
_PORT: int = int(_server_cfg.get("port", 8001))
_BASE_URL: str = _server_cfg.get("base_url", "/").rstrip("/") or "/"

log.info(
    "[consumer] producer_url=%s  host=%s  port=%d  base_url=%s",
    _PRODUCER_URL, _HOST, _PORT, _BASE_URL,
)

# Shared HTTP client — connection pooling, 30 s timeout
_http = httpx.Client(base_url=_PRODUCER_URL, timeout=30.0)


# ── Helper ───────────────────────────────────────────────────────────


def _forward(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
) -> Any:
    """Forward a request to the producer and return the parsed JSON."""
    try:
        resp = _http.request(method, path, json=json_body)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach producer at {_PRODUCER_URL}",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        )


# ── Request models ───────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., description="The natural-language question to answer.")


class SearchRequest(BaseModel):
    query_str: str = Field(..., description="The search term (case-insensitive substring match).")
    search_type: str = Field(
        "entities",
        description='What to search — "entities", "chunks", or "all".',
    )


# ── Router (mounted at base_url) ────────────────────────────────────

router = APIRouter()


@router.get("/health")
def health_endpoint() -> dict[str, Any]:
    """Local health check + producer reachability probe."""
    try:
        r = _http.get("/stats")
        r.raise_for_status()
        return {"status": "healthy", "producer": "reachable", "producer_url": _PRODUCER_URL}
    except Exception:
        return {"status": "healthy", "producer": "unreachable", "producer_url": _PRODUCER_URL}


@router.post("/query")
def query_endpoint(body: QueryRequest) -> dict:
    """Forward to producer POST /query."""
    return _forward("POST", "/query", json_body={"question": body.question})


@router.post("/search")
def search_endpoint(body: SearchRequest) -> dict:
    """Forward to producer POST /search."""
    return _forward(
        "POST",
        "/search",
        json_body={"query_str": body.query_str, "search_type": body.search_type},
    )


@router.get("/info")
def info_endpoint() -> dict:
    """Forward to producer GET /stats."""
    return _forward("GET", "/stats")


# ── FastAPI app — mounts the router at base_url ─────────────────────

app = FastAPI(
    title="ArchRAG Consumer",
    description=(
        "Read-only gateway for the ArchRAG pipeline. "
        "Forwards search, query, and info requests to the producer. "
        "This server does NOT support add, remove, or reindex — "
        "those operations belong to the producer directly."
    ),
    version="0.1.0",
)
app.include_router(router, prefix=_BASE_URL if _BASE_URL != "/" else "")


# ── Convenience: python -m consumer.mcp_server ──────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "consumer.mcp_server:app",
        host=_HOST,
        port=_PORT,
        reload=False,
    )
