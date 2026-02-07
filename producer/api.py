"""Producer FastAPI server.

Exposes the ArchRAG pipeline as HTTP endpoints, mounted under a
configurable ``base_url`` prefix read from ``producer/config.yaml``.

Endpoints (relative to *base_url*)::

    POST   /index    — Wipe & rebuild from a corpus file
    POST   /add      — Enqueue documents for batched indexing
    DELETE /remove   — Remove an entity from the knowledge graph
    POST   /reindex  — Flush the pending queue in the background
    GET    /stats    — Database statistics + queue status
    POST   /query    — Answer a natural-language question
    POST   /search   — Search entities / chunks by substring

Start with::

    python -m producer.api          # reads host/port from config.yaml
    uvicorn producer.api:app        # manual — ignores config host/port
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

# Resolve paths relative to *this* file → producer/
_PRODUCER_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PRODUCER_ROOT.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import build_orchestrator, load_config  # noqa: E402
from archrag.services.ingestion_queue import IngestionQueue  # noqa: E402

log = logging.getLogger(__name__)

# ── Eager initialisation ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_config_path = os.environ.get(
    "PRODUCER_CONFIG",
    str(_PRODUCER_ROOT / "config.yaml"),
)
log.info("[producer] Loading config from %s …", _config_path)

_raw_cfg = load_config(_config_path)
_server_cfg = _raw_cfg.get("server", {})
_HOST: str = _server_cfg.get("host", "0.0.0.0")
_PORT: int = int(_server_cfg.get("port", 8000))
_BASE_URL: str = _server_cfg.get("base_url", "/").rstrip("/") or "/"
log.info("[producer] Server config — host=%s  port=%d  base_url=%s", _HOST, _PORT, _BASE_URL)

_orch = build_orchestrator(_config_path)
log.info("[producer] Orchestrator ready.")

_queue = IngestionQueue(
    reindex_fn=_orch.add_documents,
    flush_interval=float(os.environ.get("PRODUCER_FLUSH_INTERVAL", "180")),
)
log.info("[producer] Ingestion queue ready.")

# ── Background reindex state ─────────────────────────────────────────

_reindex_lock = threading.Lock()
_reindex_status: str = "idle"


def _bg_reindex_worker() -> None:
    """Runs on a detached daemon thread — flushes the queue."""
    global _reindex_status
    log.info("[producer:reindex] Worker started")
    try:
        flushed = _queue.flush()
        st = _orch.stats()
        log.info(
            "[producer:reindex] Done — %d doc(s). "
            "%d entities, %d relations, %d chunks, %d levels.",
            flushed, st["entities"], st["relations"],
            st["chunks"], st["hierarchy_levels"],
        )
        _reindex_status = (
            f"done: {flushed} doc(s) reindexed. "
            f"{st['entities']} entities, {st['relations']} relations, "
            f"{st['chunks']} chunks, {st['hierarchy_levels']} levels."
        )
    except Exception as exc:
        log.exception("[producer:reindex] FAILED")
        _reindex_status = f"error: {exc}"


# ── Request / response models ───────────────────────────────────────


class IndexRequest(BaseModel):
    corpus_path: str = Field(..., description="Path to a JSONL or JSON-array file on disk.")


class AddRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(
        ...,
        description='List of document dicts, each with at least a "content" key.',
    )


class RemoveRequest(BaseModel):
    entity_name: str = Field(..., description="Exact entity name to delete.")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question to answer.")


class SearchRequest(BaseModel):
    query_str: str = Field(..., description="Search term (case-insensitive substring match).")
    search_type: str = Field(
        "entities",
        description='What to search — "entities", "chunks", or "all".',
    )


# ── Router (mounted at base_url) ────────────────────────────────────

router = APIRouter()


@router.post("/index")
def index_endpoint(body: IndexRequest) -> dict[str, Any]:
    """Wipe existing data and rebuild the full index from a corpus file."""
    _orch.index(body.corpus_path)
    st = _orch.stats()
    return {
        "status": "ok",
        "message": (
            f"Indexing complete. {st['entities']} entities, "
            f"{st['relations']} relations, {st['chunks']} chunks, "
            f"{st['hierarchy_levels']} hierarchy levels."
        ),
        **st,
    }


@router.post("/add")
def add_endpoint(body: AddRequest) -> dict[str, Any]:
    """Enqueue documents for batched indexing.

    Documents sit in a pending queue and are flushed every 3 minutes,
    or when ``POST /reindex`` is called.
    """
    pending = _queue.enqueue(body.documents)
    return {
        "status": "ok",
        "enqueued": len(body.documents),
        "pending": pending,
        "message": (
            f"Enqueued {len(body.documents)} doc(s). "
            f"{pending} total pending. "
            "Call POST /reindex or wait for auto-flush."
        ),
    }


@router.delete("/remove")
def remove_endpoint(body: RemoveRequest) -> dict[str, Any]:
    """Remove an entity (and its relations) from the knowledge graph."""
    found = _orch.remove_entity(body.entity_name)
    if found:
        return {"status": "ok", "message": f"Removed entity '{body.entity_name}'."}
    return {"status": "not_found", "message": f"Entity '{body.entity_name}' not found."}


@router.post("/reindex")
def reindex_endpoint() -> dict[str, Any]:
    """Immediately flush the pending queue in the background.

    Returns instantly; poll ``GET /stats`` to check progress.
    """
    global _reindex_status

    count = _queue.pending_count()
    if count == 0:
        return {
            "status": "empty",
            "message": f"Queue empty — nothing to reindex. Last: {_reindex_status}",
        }

    if _reindex_status == "running":
        return {"status": "already_running", "message": "A reindex is already in progress."}

    _reindex_status = "running"
    t = threading.Thread(target=_bg_reindex_worker, name="producer-reindex", daemon=True)
    t.start()
    return {
        "status": "started",
        "pending": count,
        "message": f"Reindex launched for {count} doc(s). Poll GET /stats for progress.",
    }


@router.get("/stats")
def stats_endpoint() -> dict[str, Any]:
    """Return current database statistics and queue status."""
    st = _orch.stats()
    return {
        **st,
        "pending": _queue.pending_count(),
        "reindex_status": _reindex_status,
    }


@router.post("/query")
def query_endpoint(body: QueryRequest) -> dict[str, str]:
    """Answer a natural-language question using hierarchical search + adaptive filtering."""
    answer = _orch.query(body.question)
    return {"answer": answer}


@router.post("/search")
def search_endpoint(body: SearchRequest) -> dict:
    """Search the knowledge graph by substring."""
    result: dict = {}

    if body.search_type in ("entities", "all"):
        result["entities"] = _orch.search_entities(body.query_str)

    if body.search_type in ("chunks", "all"):
        result["chunks"] = _orch.search_chunks(body.query_str)

    return result


# ── FastAPI app — mounts the router at base_url ─────────────────────

app = FastAPI(
    title="ArchRAG Producer",
    description=(
        "Full ArchRAG pipeline API. "
        "Index, add, remove, reindex, query, search, and inspect stats."
    ),
    version="0.1.0",
)
app.include_router(router, prefix=_BASE_URL if _BASE_URL != "/" else "")


# ── Convenience: python -m producer.api ──────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "producer.api:app",
        host=_HOST,
        port=_PORT,
        reload=False,
    )
