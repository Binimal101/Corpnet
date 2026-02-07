"""ArchRAG FastAPI Server.

Exposes the full ArchRAG pipeline as HTTP endpoints:
  - GET  /health        — Quick health check
  - POST /index         — Build full index from corpus file
  - POST /query         — Answer a question
  - POST /search        — Search entities / chunks
  - POST /add           — Enqueue documents for batched indexing
  - DELETE /remove      — Remove an entity
  - POST /reindex       — Flush the pending queue
  - GET  /info          — Database statistics + queue status

The orchestrator and ingestion queue are initialised eagerly at
startup so every endpoint is ready to respond immediately.

The ``POST /add`` endpoint enqueues documents into a thread-safe
IngestionQueue which auto-flushes every 3 minutes.
The ``POST /reindex`` endpoint forces an immediate flush.

Start with::

    uvicorn archrag.mcp_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Walk up from this file (archrag/mcp_server.py) to the project root and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import build_orchestrator
from archrag.services.ingestion_queue import IngestionQueue

log = logging.getLogger(__name__)

# ── FastAPI app ──

app = FastAPI(
    title="ArchRAG",
    description=(
        "ArchRAG: Attributed Community-based Hierarchical RAG server. "
        "Index corpora, query the knowledge graph, search for entities/chunks, "
        "enqueue new documents (batched), and reindex on demand."
    ),
    version="0.1.0",
)

# ── Eager initialisation (runs once at startup) ──

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_config_path = os.environ.get("ARCHRAG_CONFIG", "config.yaml")
log.info("Loading config from %s …", _config_path)

log.info("Building orchestrator (embedding, LLM, stores) …")
_orch = build_orchestrator(_config_path)
log.info("Orchestrator ready.")

log.info("Starting ingestion queue …")
_queue = IngestionQueue(
    reindex_fn=_orch.add_documents,
    flush_interval=float(os.environ.get("ARCHRAG_FLUSH_INTERVAL", "180")),
)
log.info("Ingestion queue ready (flush every %s s).", os.environ.get("ARCHRAG_FLUSH_INTERVAL", "180"))

log.info("All components initialised — server is ready.")


# ── Request models ──


class IndexRequest(BaseModel):
    corpus_path: str = Field(..., description="Path to a JSONL or JSON-array corpus file on disk.")


class QueryRequest(BaseModel):
    question: str = Field(..., description="The natural-language question to answer.")


class SearchRequest(BaseModel):
    query_str: str = Field(..., description="The search term (case-insensitive substring match).")
    search_type: str = Field(
        "entities",
        description='What to search — "entities", "chunks", or "all".',
    )


class AddRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(
        ...,
        description='List of document dicts, each with at least a "content" key.',
    )


class RemoveRequest(BaseModel):
    entity_name: str = Field(..., description="The exact name of the entity to delete.")


# ── Background reindex ──

_reindex_lock = threading.Lock()
_reindex_status: str = "idle"  # "idle" | "running" | "done: ..."


def _bg_reindex_worker():
    """Runs entirely on a detached thread — no asyncio involvement."""
    global _reindex_status
    log.info("[reindex-bg] Worker thread started")
    try:
        log.info("[reindex-bg] Calling _queue.flush() ...")
        flushed = _queue.flush()
        log.info("[reindex-bg] flush() returned: %d doc(s) flushed", flushed)

        log.info("[reindex-bg] Calling _orch.stats() ...")
        st = _orch.stats()
        log.info(
            "[reindex-bg] Complete: %d doc(s). "
            "DB: %d entities, %d relations, %d chunks, %d levels.",
            flushed, st["entities"], st["relations"],
            st["chunks"], st["hierarchy_levels"],
        )
        _reindex_status = (
            f"done: {flushed} doc(s) reindexed. "
            f"{st['entities']} entities, {st['relations']} relations, "
            f"{st['chunks']} chunks, {st['hierarchy_levels']} levels."
        )
    except Exception as exc:
        log.exception("[reindex-bg] FAILED")
        _reindex_status = f"error: {exc}"


# ── Endpoints ──


@app.get("/health")
def health_endpoint() -> dict[str, str]:
    """Quick health check. Returns immediately."""
    return {"status": "healthy"}


@app.post("/index")
def index_endpoint(body: IndexRequest) -> dict[str, Any]:
    """Build the full index from a corpus file (JSONL or JSON array).

    This wipes existing data and rebuilds from scratch.
    """
    _orch.index(body.corpus_path)
    st = _orch.stats()
    return {
        "status": "ok",
        "message": (
            f"Indexing complete. "
            f"{st['entities']} entities, {st['relations']} relations, "
            f"{st['chunks']} chunks, {st['hierarchy_levels']} hierarchy levels."
        ),
        **st,
    }


@app.post("/query")
def query_endpoint(body: QueryRequest) -> dict[str, str]:
    """Answer a question using hierarchical search + adaptive filtering."""
    answer = _orch.query(body.question)
    return {"answer": answer}


@app.post("/search")
def search_endpoint(body: SearchRequest) -> dict:
    """Search the knowledge graph by substring."""
    result: dict = {}

    if body.search_type in ("entities", "all"):
        entities = _orch.search_entities(body.query_str)
        result["entities"] = entities

    if body.search_type in ("chunks", "all"):
        chunks = _orch.search_chunks(body.query_str)
        result["chunks"] = chunks

    return result


@app.post("/add")
def add_endpoint(body: AddRequest) -> dict[str, Any]:
    """Enqueue new documents for batched indexing.

    Documents are NOT indexed immediately — they sit in a pending
    queue and are flushed every 3 minutes, or when ``POST /reindex`` is called.
    """
    pending = _queue.enqueue(body.documents)
    return {
        "status": "ok",
        "enqueued": len(body.documents),
        "pending": pending,
        "message": (
            f"Enqueued {len(body.documents)} document(s). "
            f"{pending} total pending. "
            f"They will be indexed on the next flush (every 3 min) or call POST /reindex."
        ),
    }


@app.delete("/remove")
def remove_endpoint(body: RemoveRequest) -> dict[str, str]:
    """Remove an entity (and its relations) from the knowledge graph."""
    if _orch.remove_entity(body.entity_name):
        return {"status": "ok", "message": f"Removed entity '{body.entity_name}' and its relations."}
    return {"status": "not_found", "message": f"Entity '{body.entity_name}' not found."}


@app.post("/reindex")
def reindex_endpoint() -> dict[str, Any]:
    """Immediately flush the pending document queue and reindex.

    Runs in the background so other endpoints remain responsive.
    Use ``GET /info`` to check when it finishes.
    """
    global _reindex_status

    log.info("[reindex] Endpoint called")

    count = _queue.pending_count()
    log.info("[reindex] Pending count: %d", count)
    if count == 0:
        last = _reindex_status
        log.info("[reindex] Queue empty. Last status: %s", last)
        return {
            "status": "empty",
            "message": f"Queue is empty — nothing to reindex. Last reindex status: {last}",
        }

    if _reindex_status == "running":
        log.info("[reindex] Already running, skipping")
        return {"status": "already_running", "message": "A reindex is already in progress. Use GET /info to check status."}

    _reindex_status = "running"
    log.info("[reindex] Spawning background thread ...")
    t = threading.Thread(target=_bg_reindex_worker, name="reindex-bg", daemon=True)
    t.start()
    log.info("[reindex] Thread spawned (id=%s), returning immediately", t.ident)

    return {
        "status": "started",
        "pending": count,
        "message": (
            f"Reindex started in background for {count} document(s). "
            f"Use GET /info to check progress — pending count will drop to 0 when done."
        ),
    }


@app.get("/info")
def info_endpoint() -> dict[str, Any]:
    """Show database statistics and queue status."""
    st = _orch.stats()
    return {
        **st,
        "pending": _queue.pending_count(),
        "reindex_status": _reindex_status,
    }


# ── Convenience: python -m archrag.mcp_server ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "archrag.mcp_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("ARCHRAG_PORT", "8000")),
        reload=False,
    )
