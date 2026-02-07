"""ArchRAG FastMCP Server.

Exposes the full ArchRAG pipeline as MCP tools:
  - index, query, search, add, remove, reindex, info
  - add_note, get_note, get_related_notes, search_notes (MemoryNote system)

The `add` tool enqueues documents into a thread-safe
IngestionQueue which auto-flushes every 3 minutes.
The `reindex` tool forces an immediate flush.

The `add_note` tool creates enriched MemoryNotes with LLM-generated
metadata following the A-Mem paper design.
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
        "(batched), and 'reindex' to flush the queue immediately. "
        "Use 'add_note' for enriched memory notes with auto-generated "
        "keywords, context, tags, and links to related memories."
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
    count = queue.flush()
    if count == 0:
        return "Queue is empty — nothing to reindex."

    orch, _ = _get_orchestrator()
    st = orch.stats()
    return (
        f"Reindexed {count} document(s). "
        f"DB now has {st['entities']} entities, {st['relations']} relations, "
        f"{st['chunks']} chunks, {st['hierarchy_levels']} hierarchy levels."
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
        f"Memory notes:     {st.get('memory_notes', 0)}\n"
        f"Pending in queue: {pending}"
    )


# ── MemoryNote Tools (A-Mem inspired) ──


@mcp.tool()
def add_note(
    content: str,
    category: str | None = None,
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    skip_kg: bool = False,
    rebuild_hierarchy: bool = False,
) -> str:
    """Add a structured memory note with LLM-generated metadata.

    Uses the unified pipeline to ensure consistent processing:
        Input → MemoryNote → Chunks → KG entities/relations

    Unlike 'add' which batches documents for later indexing,
    this immediately creates an enriched MemoryNote with:
    - Auto-generated keywords and tags
    - TextChunks for entity/relation extraction

    Args:
        content: The main text content of the memory.
        category: Optional classification category.
        tags: Optional list of user-provided tags (merged with LLM-generated).
        keywords: Optional list of user-provided keywords (merged with LLM-generated).
        skip_kg: If True, only create MemoryNote without KG extraction (default: False).
        rebuild_hierarchy: Whether to rebuild clustering/C-HNSW after adding (default: False).
    """
    orch, _ = _get_orchestrator()

    input_data: dict[str, Any] = {"content": content}
    if category:
        input_data["category"] = category
    if tags:
        input_data["tags"] = tags
    if keywords:
        input_data["keywords"] = keywords

    try:
        result = orch.add_memory_note(
            input_data,
            skip_kg=skip_kg,
            rebuild_hierarchy=rebuild_hierarchy,
        )
        return (
            f"Created memory note {result['id']} (via unified pipeline).\n"
            f"Keywords: {result['keywords']}\n"
            f"Context: {result['context']}\n"
            f"Tags: {result['tags']}\n"
            f"Links: {len(result['links'])} related notes\n"
            f"KG extraction: {'skipped' if skip_kg else 'completed'}"
        )
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def get_note(note_id: str) -> str:
    """Retrieve a memory note by ID.

    Args:
        note_id: The unique identifier of the note.
    """
    orch, _ = _get_orchestrator()
    result = orch.get_memory_note(note_id)

    if result is None:
        return f"Note '{note_id}' not found."

    return (
        f"ID: {result['id']}\n"
        f"Content: {result['content']}\n"
        f"Keywords: {result['keywords']}\n"
        f"Tags: {result['tags']}\n"
        f"Category: {result['category']}\n"
        f"Last updated: {result.get('last_updated', 'N/A')}\n"
        f"Retrieval count: {result['retrieval_count']}\n"
        f"Embedding model: {result.get('embedding_model', 'N/A')}"
    )


@mcp.tool()
def get_related_notes(note_id: str) -> str:
    """Get notes linked to a given note.

    Args:
        note_id: The unique identifier of the source note.
    """
    orch, _ = _get_orchestrator()
    related = orch.get_related_notes(note_id)

    if not related:
        return f"No related notes found for '{note_id}'."

    lines = []
    for r in related:
        lines.append(
            f"  [{r['relation_type']}] {r['id']}: {r['content'][:100]}..."
        )

    return f"Related notes ({len(related)}):\n" + "\n".join(lines)


@mcp.tool()
def search_notes(query_str: str, k: int = 10) -> str:
    """Semantic search for memory notes by content similarity.

    Args:
        query_str: The search query (semantic matching).
        k: Maximum number of results to return.
    """
    orch, _ = _get_orchestrator()
    results = orch.search_notes_by_content(query_str, k=k)

    if not results:
        return f"No notes found matching '{query_str}'."

    lines = []
    for r in results:
        tags_str = ", ".join(r['tags'][:3]) if r['tags'] else "no tags"
        lines.append(f"  {r['id']}: {r['content'][:100]}... [{tags_str}]")

    return f"Notes ({len(results)}):\n" + "\n".join(lines)


@mcp.tool()
def delete_note(note_id: str) -> str:
    """Delete a memory note by ID.

    Args:
        note_id: The unique identifier of the note to delete.
    """
    orch, _ = _get_orchestrator()
    if orch.delete_memory_note(note_id):
        return f"Deleted note '{note_id}'."
    return f"Note '{note_id}' not found."


# ── External Database Tools ──


@mcp.tool()
def connect_database(
    connector_type: str,
    connection_string: str | None = None,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Connect to an external database for syncing data into ArchRAG.

    Supports both SQL databases (via SQLAlchemy) and NoSQL (MongoDB).

    For SQL databases, provide a connection_string:
      connector_type="sql", connection_string="postgresql://user:pass@host/db"

    For MongoDB, provide individual parameters:
      connector_type="nosql", host="localhost", port=27017, database="mydb"

    Args:
        connector_type: "sql" for SQL databases, "nosql" for MongoDB.
        connection_string: Full connection URI (SQL) or MongoDB URI.
        host: Database host (for NoSQL without connection_string).
        port: Database port (for NoSQL without connection_string).
        database: Database name (for NoSQL without connection_string).
        username: Optional username (for NoSQL without connection_string).
        password: Optional password (for NoSQL without connection_string).
    """
    orch, _ = _get_orchestrator()

    config: dict[str, Any] = {}

    if connection_string:
        config["connection_string"] = connection_string
    else:
        if host:
            config["host"] = host
        if port:
            config["port"] = port
        if database:
            config["database"] = database
        if username:
            config["username"] = username
        if password:
            config["password"] = password

    try:
        result = orch.connect_database(connector_type, config)
        tables = result.get("tables", [])
        info = result.get("connection_info", {})

        return (
            f"Connected to {connector_type} database.\n"
            f"Host: {info.get('host', 'N/A')}\n"
            f"Database: {info.get('database', 'N/A')}\n"
            f"Tables found: {len(tables)}\n"
            f"Tables: {', '.join(tables[:10])}{'...' if len(tables) > 10 else ''}"
        )
    except Exception as e:
        return f"Connection failed: {e}"


@mcp.tool()
def disconnect_database() -> str:
    """Disconnect from the external database."""
    orch, _ = _get_orchestrator()
    if orch.disconnect_database():
        return "Disconnected from database."
    return "No database connection to disconnect."


@mcp.tool()
def list_tables() -> str:
    """List all tables/collections in the connected database."""
    orch, _ = _get_orchestrator()

    try:
        schema = orch.get_database_schema()
        tables = schema.get("tables", [])
        db_name = schema.get("database", "Unknown")

        if not tables:
            return f"No tables found in database '{db_name}'."

        lines = [f"Database: {db_name}", f"Tables ({len(tables)}):"]
        for table in tables:
            lines.append(f"  - {table}")

        return "\n".join(lines)
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def get_table_schema(table_name: str) -> str:
    """Get schema information for a specific table.

    Args:
        table_name: The name of the table to inspect.
    """
    orch, _ = _get_orchestrator()

    try:
        schema = orch.get_database_schema(table=table_name)

        lines = [
            f"Table: {schema['table']}",
            f"Database: {schema['database']}",
            f"Primary key: {schema.get('primary_key', 'N/A')}",
            f"\nColumns ({len(schema['columns'])}):",
        ]

        for col in schema["columns"]:
            text_marker = " [TEXT]" if col["is_text"] else ""
            null_marker = " (nullable)" if col["nullable"] else ""
            lines.append(f"  - {col['name']}: {col['type']}{text_marker}{null_marker}")

        if schema.get("relationships"):
            lines.append(f"\nRelationships ({len(schema['relationships'])}):")
            for rel in schema["relationships"]:
                lines.append(f"  - {rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")

        return "\n".join(lines)
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def sync_database(
    tables: list[str] | None = None,
    text_columns: dict[str, list[str]] | None = None,
    incremental: bool = True,
) -> str:
    """Sync records from the connected database into ArchRAG.

    Uses the unified pipeline to ensure all records flow through:
        Database Record → MemoryNote → TextChunks → KG entities/relations

    This creates enriched MemoryNotes with LLM-generated metadata,
    extracts entities and relations for the knowledge graph,
    and enables full hierarchical traversal during queries.

    Args:
        tables: Specific tables to sync. If None, syncs all tables.
        text_columns: Map of table name -> columns to use for text extraction.
                      If None, auto-detects text columns from schema.
        incremental: If True, only sync new/changed records since last sync.
                     If False, performs a full sync (re-imports everything).
    """
    orch, _ = _get_orchestrator()

    try:
        result = orch.sync_from_database(
            tables=tables,
            text_columns_map=text_columns,
            incremental=incremental,
        )

        tables_synced = result.get("tables_synced", [])
        records_added = result.get("records_added", 0)
        records_failed = result.get("records_failed", 0)
        duration = result.get("duration_seconds", 0)
        errors = result.get("errors", [])

        lines = [
            f"Sync {'(incremental)' if incremental else '(full)'} completed.",
            f"Tables synced: {', '.join(tables_synced)}",
            f"Records added: {records_added}",
            f"Records failed: {records_failed}",
            f"Duration: {duration:.2f}s",
        ]

        if errors:
            lines.append(f"\nErrors ({len(errors)}):")
            for err in errors[:5]:
                lines.append(f"  - {err}")
            if len(errors) > 5:
                lines.append(f"  ... and {len(errors) - 5} more")

        return "\n".join(lines)
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def get_sync_status() -> str:
    """Get the sync status for all tables in the connected database."""
    orch, _ = _get_orchestrator()

    status = orch.get_sync_status()

    if not status.get("connected"):
        return "No database connected."

    tables = status.get("tables", {})
    if not tables:
        return "Connected but no tables have been synced yet."

    lines = ["Sync status:"]
    for table, info in tables.items():
        lines.append(
            f"  {table}: {info['record_count']} records, "
            f"last sync: {info['last_sync_at']}"
        )

    return "\n".join(lines)


# ── Auto-Sync Tools ──


@mcp.tool()
def enable_auto_sync(
    poll_interval: int = 300,
    tables: list[str] | None = None,
    text_columns: dict[str, list[str]] | None = None,
) -> str:
    """Enable automatic background syncing of database.

    The system will poll the connected database at regular intervals
    and automatically sync new records into ArchRAG. This is connector-agnostic:
    works with SQL, NoSQL, or any future database adapter.

    Args:
        poll_interval: Seconds between polls (default: 300 = 5 minutes).
        tables: Specific tables to monitor. If None, monitors all tables.
        text_columns: Map of table name -> columns for text extraction.
    """
    orch, _ = _get_orchestrator()

    try:
        config = orch.enable_auto_sync(
            poll_interval=float(poll_interval),
            tables=tables,
            text_columns_map=text_columns,
        )
        return (
            f"Auto-sync ENABLED\n"
            f"Poll interval: {config['poll_interval']}s\n"
            f"Tables: {config['tables'] or 'all'}"
        )
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def disable_auto_sync() -> str:
    """Disable automatic background syncing.

    The worker thread will stop polling for new records.
    """
    orch, _ = _get_orchestrator()

    if orch.disable_auto_sync():
        return "Auto-sync DISABLED"
    return "Auto-sync was not enabled"


@mcp.tool()
def configure_auto_sync(
    poll_interval: int | None = None,
    tables: list[str] | None = None,
    text_columns: dict[str, list[str]] | None = None,
) -> str:
    """Update auto-sync configuration without enabling/disabling.

    Args:
        poll_interval: Seconds between polls.
        tables: Specific tables to monitor.
        text_columns: Map of table name -> columns for text extraction.
    """
    orch, _ = _get_orchestrator()

    try:
        config = orch.configure_auto_sync(
            poll_interval=float(poll_interval) if poll_interval else None,
            tables=tables,
            text_columns_map=text_columns,
        )
        return (
            f"Auto-sync configuration updated:\n"
            f"Poll interval: {config['poll_interval']}s\n"
            f"Tables: {config['tables'] or 'all'}"
        )
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def get_auto_sync_status() -> str:
    """Get auto-sync status including configuration and statistics."""
    orch, _ = _get_orchestrator()

    status = orch.get_auto_sync_status()

    if not status.get("enabled") and status.get("config") is None:
        return "Auto-sync not initialized. Call enable_auto_sync() first."

    lines = [
        f"Auto-sync: {'ENABLED' if status['enabled'] else 'DISABLED'}",
        f"Currently syncing: {status.get('syncing', False)}",
    ]

    if status.get("config"):
        cfg = status["config"]
        lines.extend([
            f"\nConfiguration:",
            f"  Poll interval: {cfg['poll_interval']}s",
            f"  Tables: {cfg['tables'] or 'all'}",
        ])

    if status.get("stats"):
        stats = status["stats"]
        lines.extend([
            f"\nStatistics:",
            f"  Total polls: {stats['total_polls']}",
            f"  Records synced: {stats['total_records_synced']}",
            f"  Records failed: {stats['total_records_failed']}",
            f"  Last poll: {stats['last_poll_time'] or 'never'}",
            f"  Last poll records: {stats['last_poll_records']}",
        ])
        if stats.get("last_error"):
            lines.append(f"  Last error: {stats['last_error']}")

    return "\n".join(lines)


@mcp.tool()
def trigger_sync_now() -> str:
    """Manually trigger an immediate sync.

    Bypasses the poll interval and syncs now.
    Useful when you know new data has been added.
    """
    orch, _ = _get_orchestrator()

    try:
        result = orch.trigger_sync_now()

        if result.get("skipped"):
            return f"Sync skipped: {result.get('error', 'already in progress')}"

        return (
            f"Sync completed:\n"
            f"Tables: {result.get('tables_synced', [])}\n"
            f"Records added: {result.get('records_added', 0)}\n"
            f"Records failed: {result.get('records_failed', 0)}\n"
            f"Duration: {result.get('duration_seconds', 0):.2f}s"
        )
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def request_debounced_sync(tables: list[str] | None = None) -> str:
    """Request a debounced sync for tables.

    Multiple requests within 30 seconds are batched together.
    This prevents redundant syncs during high-volume periods.

    Args:
        tables: Tables to sync. If None, syncs all tables.
    """
    orch, _ = _get_orchestrator()

    try:
        result = orch.request_sync(tables)
        return (
            f"Sync request queued.\n"
            f"Pending tables: {result['pending_tables']}\n"
            f"{result['message']}"
        )
    except RuntimeError as e:
        return f"Error: {e}"


@mcp.tool()
def flush_sync_queue() -> str:
    """Immediately flush the sync queue.

    Forces all pending sync requests to execute now,
    ignoring the debounce window.
    """
    orch, _ = _get_orchestrator()

    result = orch.flush_sync_queue()

    if result.get("skipped"):
        return f"Queue empty or not initialized: {result.get('reason', 'unknown')}"

    if result.get("error"):
        return f"Flush failed: {result['error']}"

    return (
        f"Queue flushed:\n"
        f"Tables: {result.get('tables_synced', [])}\n"
        f"Records added: {result.get('records_added', 0)}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
