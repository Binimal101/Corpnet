"""Producer functional API.

Exposes the **write-side** of the ArchRAG pipeline as plain Python
functions.  No MCP — this module is imported directly or will later
be wrapped in a network-accessible server.

Every public function returns a plain ``dict`` so callers
(future MCP wrapper, REST endpoint, CLI) can serialise the result
in whatever format they need.

Usage::

    from producer.api import index, add, remove, reindex, stats

    index("corpus.jsonl")
    add([{"title": "Doc", "context": "..."}])
    reindex()
    stats()
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Resolve paths relative to *this* file → producer/
_PRODUCER_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PRODUCER_ROOT.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import build_orchestrator          # noqa: E402
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


# ── Public functional API ────────────────────────────────────────────


def index(corpus_path: str) -> dict[str, Any]:
    """Wipe existing data and rebuild the full index from a corpus file.

    Args:
        corpus_path: Path to a JSONL or JSON-array file on disk.

    Returns:
        Stats dict after indexing completes.
    """
    _orch.index(corpus_path)
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


def add(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Enqueue documents for batched indexing.

    Documents sit in a pending queue and are flushed every 3 minutes,
    or when :func:`reindex` is called.

    Args:
        documents: Each dict should have at least a ``"text"`` key
                   (or ``"title"`` + ``"context"``).

    Returns:
        Status with pending count.
    """
    pending = _queue.enqueue(documents)
    return {
        "status": "ok",
        "enqueued": len(documents),
        "pending": pending,
        "message": (
            f"Enqueued {len(documents)} doc(s). "
            f"{pending} total pending. "
            "Call reindex() or wait for auto-flush."
        ),
    }


def remove(entity_name: str) -> dict[str, Any]:
    """Remove an entity (and its relations) from the knowledge graph.

    Args:
        entity_name: Exact entity name to delete.

    Returns:
        Status indicating whether the entity was found.
    """
    found = _orch.remove_entity(entity_name)
    if found:
        return {"status": "ok", "message": f"Removed entity '{entity_name}'."}
    return {"status": "not_found", "message": f"Entity '{entity_name}' not found."}


def reindex() -> dict[str, Any]:
    """Immediately flush the pending queue in the background.

    Returns instantly; use :func:`info` to poll progress.

    Returns:
        Status with pending count at launch time.
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
        "message": f"Reindex launched for {count} doc(s). Poll info() for progress.",
    }


def stats() -> dict[str, Any]:
    """Return current database statistics and queue status.

    Returns:
        Dict with entity/relation/chunk counts, hierarchy depth,
        pending queue size, and reindex status.
    """
    st = _orch.stats()
    return {
        **st,
        "pending": _queue.pending_count(),
        "reindex_status": _reindex_status,
    }
