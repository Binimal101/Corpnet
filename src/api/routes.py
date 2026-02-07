"""API routes for DAC-HRAG.

Single routes file containing all endpoints:
- Health checks
- Query endpoints
- Ingestion endpoints
- Admin endpoints
"""

from __future__ import annotations

from typing import Any
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.types import DocumentChunk, IngestRecord

router = APIRouter()


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    ready: bool
    database: bool = False
    peers: int = 0


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness check endpoint."""
    # TODO: Check actual database and peer connectivity
    return ReadyResponse(ready=True, database=False, peers=0)


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    similarity_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Cosine similarity floor for community routing"
    )
    top_k: int = Field(default=10, ge=1, le=100)


class SourceInfo(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    peer_id: str


class QueryResponseModel(BaseModel):
    query_id: str
    answer: str
    sources: list[SourceInfo]
    latency_ms: dict[str, float] = {}


@router.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequest):
    """Execute a RAG query."""
    # TODO: Initialize query engine with real dependencies
    return QueryResponseModel(
        query_id=str(uuid.uuid4()),
        answer="Query engine not yet initialized. Please configure the system first.",
        sources=[],
        latency_ms={"total": 0},
    )


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestRecordModel(BaseModel):
    metadata: dict[str, Any] = Field(
        ...,
        description="Arbitrary key-value pairs describing the entity"
    )
    embedding: list[float] = Field(
        ...,
        description="Pre-computed embedding vector"
    )


class IngestRequest(BaseModel):
    records: list[IngestRecordModel] = Field(
        ...,
        description="List of records to ingest"
    )
    source: str = Field(
        ...,
        description="Source identifier"
    )


class IngestResponse(BaseModel):
    ingested: int
    chunk_ids: list[str]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest pre-embedded records."""
    # TODO: Initialize pipeline with real dependencies
    chunk_ids = [str(uuid.uuid4()) for _ in request.records]
    return IngestResponse(ingested=len(chunk_ids), chunk_ids=chunk_ids)


class TextIngestRequest(BaseModel):
    text: str
    doc_id: str
    source: str
    metadata: dict[str, Any] = {}


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest):
    """Ingest raw text (will be chunked and embedded)."""
    # TODO: Initialize pipeline with real dependencies
    return IngestResponse(ingested=0, chunk_ids=[])


# ── Admin ────────────────────────────────────────────────────────────────────

class PeerInfoModel(BaseModel):
    peer_id: str
    address: str
    port: int
    communities: list[str]
    is_super_peer: bool


class HierarchyInfo(BaseModel):
    total_levels: int
    total_communities: int
    levels: list[int]


@router.get("/admin/peers", response_model=list[PeerInfoModel])
async def get_peers():
    """Get all registered peers."""
    # TODO: Return actual peers
    return []


@router.get("/admin/hierarchy", response_model=HierarchyInfo)
async def get_hierarchy():
    """Get hierarchy information."""
    # TODO: Return actual hierarchy
    return HierarchyInfo(total_levels=0, total_communities=0, levels=[])


@router.post("/admin/recluster")
async def trigger_recluster():
    """Trigger hierarchy reclustering."""
    # TODO: Implement reclustering trigger
    return {"status": "not_implemented"}


@router.get("/admin/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "chunks": 0,
        "communities": 0,
        "peers": 0,
        "queries_total": 0,
    }
