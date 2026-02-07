"""Shared dataclasses and type definitions for DAC-HRAG.

All types implement to_dict() and from_dict() for msgpack serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class DocumentChunk:
    """A document chunk with embedding, entities, and relations.
    
    Entities and relations are extracted during ingestion and stored
    directly on the chunk for later KG construction.
    """
    chunk_id: str
    doc_id: str  # Parent document ID
    text: str
    embedding: list[float] | None = None  # None before embedding step
    labels: set[str] = field(default_factory=set)  # Permission labels (empty for MVP)
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)  # Extracted entity names
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # (source, rel_type, target)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "embedding": self.embedding,
            "labels": list(self.labels),
            "metadata": self.metadata,
            "entities": self.entities,
            "relations": [list(r) for r in self.relations],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DocumentChunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            text=d["text"],
            embedding=d.get("embedding"),
            labels=set(d.get("labels", [])),
            metadata=d.get("metadata", {}),
            entities=d.get("entities", []),
            relations=[tuple(r) for r in d.get("relations", [])],
        )


@dataclass
class IngestRecord:
    """A single record uploaded by a peer node with pre-computed embedding."""
    metadata: dict[str, Any]
    embedding: list[float]
    
    def to_dict(self) -> dict:
        return {"metadata": self.metadata, "embedding": self.embedding}
    
    @classmethod
    def from_dict(cls, d: dict) -> "IngestRecord":
        return cls(metadata=d["metadata"], embedding=d["embedding"])


@dataclass
class Entity:
    """A named entity in the knowledge graph."""
    entity_id: str
    name: str
    description: str
    embedding: list[float] | None = None
    chunk_ids: list[str] = field(default_factory=list)  # Chunks this entity appears in
    labels: set[str] = field(default_factory=set)  # Union of chunk labels
    
    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "description": self.description,
            "embedding": self.embedding,
            "chunk_ids": self.chunk_ids,
            "labels": list(self.labels),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            entity_id=d["entity_id"],
            name=d["name"],
            description=d["description"],
            embedding=d.get("embedding"),
            chunk_ids=d.get("chunk_ids", []),
            labels=set(d.get("labels", [])),
        )


@dataclass
class Community:
    """A community in the hierarchical clustering structure."""
    community_id: str
    level: int
    summary: str
    summary_embedding: list[float]
    labels: set[str] = field(default_factory=set)  # Union of member labels (empty for MVP)
    member_ids: list[str] = field(default_factory=list)  # Child entity or community IDs
    parent_id: str | None = None  # Upward link
    children_ids: list[str] = field(default_factory=list)  # Downward links
    peer_id: str = ""  # Which peer owns this community
    
    def to_dict(self) -> dict:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "summary": self.summary,
            "summary_embedding": self.summary_embedding,
            "labels": list(self.labels),
            "member_ids": self.member_ids,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "peer_id": self.peer_id,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Community":
        return cls(
            community_id=d["community_id"],
            level=d["level"],
            summary=d["summary"],
            summary_embedding=d["summary_embedding"],
            labels=set(d.get("labels", [])),
            member_ids=d.get("member_ids", []),
            parent_id=d.get("parent_id"),
            children_ids=d.get("children_ids", []),
            peer_id=d.get("peer_id", ""),
        )


@dataclass
class PeerInfo:
    """Information about a peer node in the network."""
    peer_id: str
    address: str
    port: int
    communities: list[str] = field(default_factory=list)  # Community IDs this peer owns
    is_super_peer: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "address": self.address,
            "port": self.port,
            "communities": self.communities,
            "is_super_peer": self.is_super_peer,
            "last_heartbeat": self.last_heartbeat,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PeerInfo":
        return cls(
            peer_id=d["peer_id"],
            address=d["address"],
            port=d["port"],
            communities=d.get("communities", []),
            is_super_peer=d.get("is_super_peer", False),
            last_heartbeat=d.get("last_heartbeat", time.time()),
        )


@dataclass
class SearchResult:
    """A single result from a vector search."""
    chunk_id: str
    text: str
    score: float
    source: str
    peer_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "peer_id": self.peer_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SearchResult":
        return cls(
            chunk_id=d["chunk_id"],
            text=d["text"],
            score=d["score"],
            source=d["source"],
            peer_id=d["peer_id"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class QueryRequest:
    """A query request with embedded vector."""
    query_id: str
    text: str
    embedding: list[float]
    topic_hints: list[str] = field(default_factory=list)
    similarity_threshold: float = 0.35
    top_k: int = 10
    hop_count: int = 0
    ttl: int = 10
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "embedding": self.embedding,
            "topic_hints": self.topic_hints,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "hop_count": self.hop_count,
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "QueryRequest":
        return cls(
            query_id=d["query_id"],
            text=d["text"],
            embedding=d["embedding"],
            topic_hints=d.get("topic_hints", []),
            similarity_threshold=d.get("similarity_threshold", 0.35),
            top_k=d.get("top_k", 10),
            hop_count=d.get("hop_count", 0),
            ttl=d.get("ttl", 10),
        )


@dataclass
class QueryResponse:
    """Full query response with observability data."""
    query_id: str
    answer: str
    results: list[SearchResult] = field(default_factory=list)
    routing_path: list[str] = field(default_factory=list)  # Community IDs traversed
    latency_ms: dict[str, float] = field(default_factory=dict)  # Per-stage timing
    total_messages: int = 0  # Inter-peer messages used
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "results": [r.to_dict() for r in self.results],
            "routing_path": self.routing_path,
            "latency_ms": self.latency_ms,
            "total_messages": self.total_messages,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "QueryResponse":
        return cls(
            query_id=d["query_id"],
            answer=d["answer"],
            results=[SearchResult.from_dict(r) for r in d.get("results", [])],
            routing_path=d.get("routing_path", []),
            latency_ms=d.get("latency_ms", {}),
            total_messages=d.get("total_messages", 0),
        )
