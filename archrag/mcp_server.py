"""ArchRAG FastMCP Server.

Exposes the full ArchRAG pipeline as MCP tools:
  - index, query, search, add, remove, reindex, info

The `add` tool enqueues documents into a thread-safe
IngestionQueue which auto-flushes every 3 minutes.
The `reindex` tool forces an immediate flush.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

# Walk up from this file (archrag/mcp_server.py) to the project root and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

log = logging.getLogger(__name__)

# ── Server singleton ──

mcp = FastMCP(
    "ArchRAG",
    instructions=(
        "ArchRAG: Attributed Community-based Hierarchical RAG server. "
        "Use 'index' to build from a corpus, 'query' to ask questions, "
        "'search' to find entities/chunks, 'add' to enqueue new documents "
        "(batched), and 'reindex' to flush the queue immediately."
    ),
)

# ── Lazy globals (initialised on first tool call) ──

_orch = None
_queue = None
_init_lock = __import__("threading").Lock()


def _get_orchestrator():
    """Lazy-init the orchestrator + ingestion queue (thread-safe)."""
    global _orch, _queue
    if _orch is not None:
        return _orch, _queue

    with _init_lock:
        # Double-check after acquiring lock
        if _orch is not None:
            return _orch, _queue

        from archrag.config import build_orchestrator
        from archrag.services.ingestion_queue import IngestionQueue

        config_path = os.environ.get("ARCHRAG_CONFIG", "config.yaml")
        log.info("Initialising ArchRAG orchestrator from %s", config_path)
        _orch = build_orchestrator(config_path)

        _queue = IngestionQueue(
            reindex_fn=_orch.add_documents,
            flush_interval=float(os.environ.get("ARCHRAG_FLUSH_INTERVAL", "180")),
        )
        return _orch, _queue


# ── MCP Tools ──


@mcp.tool()
def index(corpus_path: str) -> str:
    """Build the full index from a corpus file (JSONL or JSON array).

    This wipes existing data and rebuilds from scratch.

    Args:
        corpus_path: Path to the corpus file on disk.
    """
    orch, _ = _get_orchestrator()
    orch.index(corpus_path)
    st = orch.stats()
    return (
        f"Indexing complete. "
        f"{st['entities']} entities, {st['relations']} relations, "
        f"{st['chunks']} chunks, {st['hierarchy_levels']} hierarchy levels."
    )


@mcp.tool()
def query(question: str) -> str:
    """Answer a question using hierarchical search + adaptive filtering.

    Args:
        question: The natural-language question to answer.
    """
    orch, _ = _get_orchestrator()
    return orch.query(question)


@mcp.tool()
def search(
    query_str: str,
    search_type: str = "entities",
) -> str:
    """Search the knowledge graph by substring.

    Args:
        query_str: The search term (case-insensitive substring match).
        search_type: What to search — "entities", "chunks", or "all".
    """
    orch, _ = _get_orchestrator()
    parts: list[str] = []

    if search_type in ("entities", "all"):
        entities = orch.search_entities(query_str)
        if entities:
            lines = [f"  [{e['type']}] {e['name']}: {e['description'][:120]}" for e in entities]
            parts.append(f"Entities ({len(entities)}):\n" + "\n".join(lines))
        else:
            parts.append(f"No entities matching '{query_str}'.")

    if search_type in ("chunks", "all"):
        chunks = orch.search_chunks(query_str)
        if chunks:
            lines = [f"  {c['id']}: {c['text'][:120]}..." for c in chunks]
            parts.append(f"Chunks ({len(chunks)}):\n" + "\n".join(lines))
        else:
            parts.append(f"No chunks matching '{query_str}'.")

    return "\n\n".join(parts)


@mcp.tool()
def add(documents: list[dict[str, Any]]) -> str:
    """Enqueue new documents for batched indexing.

    Documents are NOT indexed immediately — they sit in a pending
    queue and are flushed every 3 minutes, or when 'reindex' is called.

    Each document should have a "text" field (or "title"+"context").

    Args:
        documents: List of document dicts, each with at least a "text" key.
    """
    _, queue = _get_orchestrator()
    pending = queue.enqueue(documents)
    return (
        f"Enqueued {len(documents)} document(s). "
        f"{pending} total pending. "
        f"They will be indexed on the next flush (every 3 min) or call 'reindex'."
    )


@mcp.tool()
def remove(entity_name: str) -> str:
    """Remove an entity (and its relations) from the knowledge graph.

    Args:
        entity_name: The exact name of the entity to delete.
    """
    orch, _ = _get_orchestrator()
    if orch.remove_entity(entity_name):
        return f"Removed entity '{entity_name}' and its relations."
    return f"Entity '{entity_name}' not found."


@mcp.tool()
def reindex() -> str:
    """Immediately flush the pending document queue and reindex.

    Use this after 'add' calls when you want results available right away
    instead of waiting for the 3-minute auto-flush.
    """
    _, queue = _get_orchestrator()
    count = queue.pending_count()
    if count == 0:
        return "Queue is empty — nothing to reindex."

    import threading

    def _background_flush():
        try:
            flushed = queue.flush()
            log.info("Background reindex complete: %d document(s) flushed.", flushed)
        except Exception:
            log.exception("Background reindex failed")

    threading.Thread(target=_background_flush, daemon=True).start()

    return (
        f"Reindex started in background for {count} document(s). "
        f"Use 'info' to check progress (pending count will drop to 0 when done)."
    )

@mcp.tool()
def info() -> str:
    """Show database statistics and queue status."""
    orch, queue = _get_orchestrator()
    st = orch.stats()
    pending = queue.pending_count()
    return (
        f"Entities:         {st['entities']}\n"
        f"Relations:        {st['relations']}\n"
        f"Chunks:           {st['chunks']}\n"
        f"Hierarchy levels: {st['hierarchy_levels']}\n"
        f"Pending in queue: {pending}"
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
