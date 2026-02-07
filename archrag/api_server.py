"""ArchRAG FastAPI Server.

Exposes the full ArchRAG pipeline as a REST API:
  GET  /health              — liveness check
  GET  /info                — database statistics & queue status
  GET  /visualize           — serve the hierarchy visualisation HTML
  POST /index               — full rebuild from corpus file
  POST /query               — answer a question via hierarchical search
  POST /search              — substring search across entities/chunks
  POST /add                 — enqueue documents for batched indexing
  POST /reindex             — flush the pending queue immediately
  DELETE /entities/{name}   — remove an entity by name
  DELETE /clear             — wipe the entire database

The orchestrator and ingestion queue are initialised eagerly at
startup so every endpoint is ready to respond immediately.

Run with:
    uvicorn archrag.api_server:app --host 0.0.0.0 --port 8000

Or directly:
    python -m archrag.api_server
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Walk up from this file (archrag/api_server.py) to the project root and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.config import build_orchestrator
from archrag.services.ingestion_queue import IngestionQueue
from archrag.visualization import generate_visualization

log = logging.getLogger(__name__)

# ── Visualization auto-regen ────────────────────────────────────────────────

_VIZ_DB_PATH = str(_PROJECT_ROOT / "data" / "archrag.db")
_VIZ_OUT_PATH = str(_PROJECT_ROOT / "data" / "hierarchy_viz.html")


def _regen_viz() -> None:
    """Regenerate the hierarchy visualisation in the background."""
    try:
        generate_visualization(db_path=_VIZ_DB_PATH, out_path=_VIZ_OUT_PATH)
        log.info("[viz] Hierarchy visualisation regenerated → %s", _VIZ_OUT_PATH)
    except Exception:
        log.exception("[viz] Failed to regenerate visualisation")


def _regen_viz_async() -> None:
    """Fire-and-forget visualisation regeneration."""
    threading.Thread(target=_regen_viz, name="viz-regen", daemon=True).start()

# ── Global state (initialised in lifespan) ──────────────────────────────────

_orch = None
_queue = None
_reindex_status: str = "idle"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialise the orchestrator and ingestion queue at startup."""
    global _orch, _queue

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = os.environ.get("ARCHRAG_CONFIG", "config.yaml")
    log.info("Loading config from %s …", config_path)

    log.info("Building orchestrator (embedding, LLM, stores) …")
    _orch = build_orchestrator(config_path)
    log.info("Orchestrator ready.")

    log.info("Starting ingestion queue …")
    flush_interval = float(os.environ.get("ARCHRAG_FLUSH_INTERVAL", "180"))
    _queue = IngestionQueue(
        reindex_fn=_orch.add_documents,
        flush_interval=flush_interval,
    )
    log.info("Ingestion queue ready (flush every %s s).", flush_interval)
    log.info("All components initialised — API server is ready.")

    yield

    # Shutdown
    log.info("Shutting down ingestion queue …")
    _queue.shutdown()
    log.info("Shutdown complete.")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ArchRAG API",
    description=(
        "**ArchRAG**: Attributed Community-based Hierarchical "
        "Retrieval-Augmented Generation.\n\n"
        "A REST API for building, querying, and managing a knowledge graph "
        "with hierarchical community clustering and C-HNSW vector indexing."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Pydantic models ────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field(..., description="Server health status", examples=["healthy"])


class InfoResponse(BaseModel):
    entities: int = Field(..., description="Total entities in the knowledge graph")
    relations: int = Field(..., description="Total relations in the knowledge graph")
    chunks: int = Field(..., description="Total text chunks stored")
    hierarchy_levels: int = Field(..., description="Number of community hierarchy levels")
    pending_in_queue: int = Field(..., description="Documents waiting to be indexed")
    reindex_status: str = Field(
        ..., description="Current reindex status: idle, running, done:…, or error:…"
    )


class IndexRequest(BaseModel):
    corpus_path: str = Field(
        ...,
        description="Absolute or relative path to a corpus file (JSONL or JSON array) on the server's filesystem.",
        examples=["corpus.jsonl"],
    )


class IndexResponse(BaseModel):
    message: str = Field(..., description="Indexing result summary")
    entities: int = Field(..., description="Total entities after indexing")
    relations: int = Field(..., description="Total relations after indexing")
    chunks: int = Field(..., description="Total chunks after indexing")
    hierarchy_levels: int = Field(..., description="Hierarchy levels after indexing")


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="A natural-language question to answer using the knowledge graph.",
        examples=["What did Marie Curie discover?"],
    )


class QueryResponse(BaseModel):
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer from hierarchical search + adaptive filtering")


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Substring to search for (case-insensitive).",
        examples=["Einstein"],
    )
    search_type: str = Field(
        default="entities",
        description="What to search: 'entities', 'chunks', or 'all'.",
        examples=["entities", "chunks", "all"],
    )


class EntityResult(BaseModel):
    id: str
    name: str
    type: str
    description: str


class ChunkResult(BaseModel):
    id: str
    text: str
    source: str


class SearchResponse(BaseModel):
    entities: list[EntityResult] | None = Field(
        None, description="Matched entities (when search_type is 'entities' or 'all')"
    )
    chunks: list[ChunkResult] | None = Field(
        None, description="Matched chunks (when search_type is 'chunks' or 'all')"
    )


class Document(BaseModel):
    text: str = Field(
        ...,
        description="The document text to index.",
        examples=["Max Planck originated quantum theory in 1900."],
    )
    source: str | None = Field(None, description="Optional source identifier")
    id: int | str | None = Field(None, description="Optional document ID")

    model_config = {"extra": "allow"}


class AddRequest(BaseModel):
    documents: list[Document] = Field(
        ...,
        min_length=1,
        description="List of documents to enqueue. Each must have a 'text' field.",
    )


class AddResponse(BaseModel):
    enqueued: int = Field(..., description="Number of documents enqueued in this call")
    pending: int = Field(..., description="Total documents pending in the queue")
    message: str = Field(..., description="Status message")


class ReindexResponse(BaseModel):
    message: str = Field(..., description="Reindex status message")
    pending_before: int = Field(..., description="Documents in queue before flush")
    status: str = Field(..., description="Current reindex status")


class RemoveResponse(BaseModel):
    removed: bool = Field(..., description="Whether the entity was found and removed")
    entity_name: str = Field(..., description="The entity name that was requested")
    message: str = Field(..., description="Result message")


class ClearResponse(BaseModel):
    message: str = Field(..., description="Result message")
    entities: int = Field(..., description="Entity count after clear (should be 0)")
    relations: int = Field(..., description="Relation count after clear (should be 0)")
    chunks: int = Field(..., description="Chunk count after clear (should be 0)")
    hierarchy_levels: int = Field(..., description="Hierarchy levels after clear (should be 0)")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Quick liveness check. Returns immediately without heavy computation.",
)
async def health():
    return HealthResponse(status="healthy")


@app.get(
    "/info",
    response_model=InfoResponse,
    tags=["System"],
    summary="Database statistics",
    description="Returns counts of entities, relations, chunks, hierarchy levels, queue status, and reindex state.",
)
async def info():
    st = _orch.stats()
    pending = _queue.pending_count()
    return InfoResponse(
        entities=st["entities"],
        relations=st["relations"],
        chunks=st["chunks"],
        hierarchy_levels=st["hierarchy_levels"],
        pending_in_queue=pending,
        reindex_status=_reindex_status,
    )


@app.post(
    "/index",
    response_model=IndexResponse,
    tags=["Indexing"],
    summary="Full index rebuild",
    description=(
        "Wipe all existing data and rebuild the entire pipeline from a corpus file. "
        "The file must be a JSONL (one JSON object per line) or a JSON array. "
        "Each document needs at least a `text` field.\n\n"
        "**Warning**: This is a destructive operation that replaces all data."
    ),
    responses={404: {"model": ErrorResponse}},
)
async def index(req: IndexRequest):
    path = Path(req.corpus_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus file not found: {req.corpus_path}")

    _orch.index(req.corpus_path)
    _regen_viz_async()
    st = _orch.stats()
    return IndexResponse(
        message="Indexing complete.",
        entities=st["entities"],
        relations=st["relations"],
        chunks=st["chunks"],
        hierarchy_levels=st["hierarchy_levels"],
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Retrieval"],
    summary="Ask a question",
    description=(
        "Answer a natural-language question using hierarchical search across the "
        "C-HNSW index followed by LLM-based adaptive filtering. Returns a "
        "synthesised answer grounded in the knowledge graph."
    ),
)
async def query(req: QueryRequest):
    answer = _orch.query(req.question)
    return QueryResponse(question=req.question, answer=answer)


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
    summary="Substring search",
    description=(
        "Case-insensitive substring search across entities, chunks, or both. "
        "Useful for exploring what's in the knowledge graph."
    ),
)
async def search(req: SearchRequest):
    entities = None
    chunks = None

    if req.search_type in ("entities", "all"):
        raw = _orch.search_entities(req.query)
        entities = [EntityResult(**e) for e in raw]

    if req.search_type in ("chunks", "all"):
        raw = _orch.search_chunks(req.query)
        chunks = [ChunkResult(**c) for c in raw]

    return SearchResponse(entities=entities, chunks=chunks)


@app.post(
    "/add",
    response_model=AddResponse,
    tags=["Indexing"],
    summary="Add documents",
    description=(
        "Enqueue one or more documents for batched indexing. Documents are held in a "
        "pending queue and flushed automatically every 3 minutes, or immediately "
        "when `/reindex` is called.\n\n"
        "Each document must have a `text` field. Additional fields are preserved as metadata."
    ),
)
async def add(req: AddRequest):
    docs = [doc.model_dump(exclude_none=True) for doc in req.documents]
    pending = _queue.enqueue(docs)
    return AddResponse(
        enqueued=len(docs),
        pending=pending,
        message=f"Enqueued {len(docs)} document(s). Call /reindex to flush immediately.",
    )


def _bg_reindex_worker():
    """Background reindex thread — no asyncio involvement."""
    global _reindex_status
    log.info("[reindex-bg] Worker thread started")
    try:
        flushed = _queue.flush()
        _regen_viz()  # synchronous inside bg thread — it's fine
        st = _orch.stats()
        _reindex_status = (
            f"done: {flushed} doc(s) reindexed. "
            f"{st['entities']} entities, {st['relations']} relations, "
            f"{st['chunks']} chunks, {st['hierarchy_levels']} levels."
        )
        log.info("[reindex-bg] %s", _reindex_status)
    except Exception as exc:
        log.exception("[reindex-bg] FAILED")
        _reindex_status = f"error: {exc}"


@app.post(
    "/reindex",
    response_model=ReindexResponse,
    tags=["Indexing"],
    summary="Flush queue & reindex",
    description=(
        "Immediately flush the pending document queue and trigger a full reindex "
        "(KG extraction → hierarchical clustering → C-HNSW rebuild). Runs in a "
        "background thread using blue/green snapshot swap so reads are never blocked.\n\n"
        "Use `GET /info` to monitor progress — `reindex_status` will change from "
        "`running` to `done: …` when complete."
    ),
)
async def reindex():
    global _reindex_status

    count = _queue.pending_count()
    if count == 0:
        return ReindexResponse(
            message="Queue is empty — nothing to reindex.",
            pending_before=0,
            status=_reindex_status,
        )

    if _reindex_status == "running":
        return ReindexResponse(
            message="A reindex is already in progress. Use /info to check status.",
            pending_before=count,
            status="running",
        )

    _reindex_status = "running"
    t = threading.Thread(target=_bg_reindex_worker, name="reindex-bg", daemon=True)
    t.start()

    return ReindexResponse(
        message=f"Reindex started in background for {count} document(s).",
        pending_before=count,
        status="running",
    )


@app.delete(
    "/entities/{entity_name}",
    response_model=RemoveResponse,
    tags=["Management"],
    summary="Remove an entity",
    description=(
        "Delete an entity by exact name from the knowledge graph. "
        "All relations involving this entity are also removed."
    ),
    responses={404: {"model": ErrorResponse}},
)
async def remove_entity(entity_name: str):
    found = _orch.remove_entity(entity_name)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_name}' not found in the knowledge graph.",
        )
    _regen_viz_async()
    return RemoveResponse(
        removed=True,
        entity_name=entity_name,
        message=f"Removed entity '{entity_name}' and its relations.",
    )


@app.delete(
    "/clear",
    response_model=ClearResponse,
    tags=["Management"],
    summary="Clear entire database",
    description=(
        "**Destructive.** Wipe all entities, relations, chunks, communities, "
        "and vectors from the database. The knowledge graph will be empty "
        "afterwards. The hierarchy visualisation is regenerated automatically."
    ),
)
async def clear_db():
    st = _orch.clear_all()
    _regen_viz_async()
    return ClearResponse(
        message="Database cleared.",
        entities=st["entities"],
        relations=st["relations"],
        chunks=st["chunks"],
        hierarchy_levels=st["hierarchy_levels"],
    )


@app.get(
    "/visualize",
    tags=["System"],
    summary="Hierarchy visualisation",
    description=(
        "Returns an interactive Plotly HTML page with DAG, Treemap, and Sunburst "
        "views of the community hierarchy. The visualisation is auto-regenerated "
        "after every mutation (index, reindex, remove, clear)."
    ),
    response_class=HTMLResponse,
)
async def visualize():
    viz_path = Path(_VIZ_OUT_PATH)
    if not viz_path.exists():
        # Generate on first request
        _regen_viz()
    if viz_path.exists():
        html = viz_path.read_text(encoding="utf-8")
    else:
        html = "<html><body><h2>No hierarchy data. Index a corpus first.</h2></body></html>"
    return HTMLResponse(content=html)


# ── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ARCHRAG_HOST", "0.0.0.0")
    port = int(os.environ.get("ARCHRAG_PORT", "8000"))
    uvicorn.run(
        "archrag.api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
