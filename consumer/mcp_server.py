"""Consumer MCP Server — read-only.

Exposes **only** search, query, info, and health tools.
No add / remove / index / reindex — those live in the producer.

The consumer's storage adapters are stubs that return empty results.
Once the network adapter is built, they will proxy reads to the
producer over MCP / HTTP and this server will return real data
without ever touching archrag.db.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

# Resolve paths relative to *this* file → consumer/
_CONSUMER_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONSUMER_ROOT.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import build_orchestrator  # noqa: E402

log = logging.getLogger(__name__)

# ── Server singleton ──

mcp = FastMCP(
    "ArchRAG-Consumer",
    instructions=(
        "ArchRAG Consumer: read-only search & query server. "
        "Use 'query' to ask questions, 'search' to find entities/chunks. "
        "This server does NOT support add, remove, or reindex — "
        "those operations belong to the producer."
    ),
)

# ── Eager initialisation ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_config_path = os.environ.get(
    "CONSUMER_CONFIG",
    str(_CONSUMER_ROOT / "config.yaml"),
)
log.info("[consumer] Loading config from %s …", _config_path)
_orch = build_orchestrator(_config_path)
log.info("[consumer] Orchestrator ready (stub stores — no DB).")


# ── MCP Tools (read-only) ────────────────────────────────────────────


@mcp.tool()
def health() -> str:
    """Quick health check. Returns immediately."""
    return "healthy"


@mcp.tool()
def query(question: str) -> str:
    """Answer a question using hierarchical search + adaptive filtering.

    Args:
        question: The natural-language question to answer.
    """
    return _orch.query(question)


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
    parts: list[str] = []

    if search_type in ("entities", "all"):
        entities = _orch.search_entities(query_str)
        if entities:
            lines = [
                f"  [{e['type']}] {e['name']}: {e['description'][:120]}"
                for e in entities
            ]
            parts.append(f"Entities ({len(entities)}):\n" + "\n".join(lines))
        else:
            parts.append(f"No entities matching '{query_str}'.")

    if search_type in ("chunks", "all"):
        chunks = _orch.search_chunks(query_str)
        if chunks:
            lines = [f"  {c['id']}: {c['content'][:120]}..." for c in chunks]
            parts.append(f"Chunks ({len(chunks)}):\n" + "\n".join(lines))
        else:
            parts.append(f"No chunks matching '{query_str}'.")

    return "\n\n".join(parts)


@mcp.tool()
def info() -> str:
    """Show database statistics (from stub stores — will be zeros until connected to producer)."""
    st = _orch.stats()
    return (
        f"Entities:         {st['entities']}\n"
        f"Relations:        {st['relations']}\n"
        f"Chunks:           {st['chunks']}\n"
        f"Hierarchy levels: {st['hierarchy_levels']}\n"
        f"(stub stores — not yet connected to producer)"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
