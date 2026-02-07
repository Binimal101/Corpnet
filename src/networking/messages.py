"""Wire protocol message types with msgpack serialization.

All messages use msgpack for compact binary serialization.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import msgpack

from src.core.types import SearchResult


@dataclass
class BaseMessage:
    """Base class for all wire protocol messages."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_peer_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseMessage":
        raise NotImplementedError
    
    def to_bytes(self) -> bytes:
        """Serialize to msgpack bytes."""
        return msgpack.packb(self.to_dict())
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "BaseMessage":
        """Deserialize from msgpack bytes."""
        return cls.from_dict(msgpack.unpackb(data))


@dataclass
class QueryMessage(BaseMessage):
    """Query message sent to a peer for local search execution."""
    query_id: str = ""
    query_embedding: list[float] = field(default_factory=list)
    topic_hints: list[str] = field(default_factory=list)
    similarity_threshold: float = 0.35
    top_k: int = 10
    hop_count: int = 0
    ttl: int = 10
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "query",
            "message_id": self.message_id,
            "sender_peer_id": self.sender_peer_id,
            "timestamp": self.timestamp,
            "query_id": self.query_id,
            "query_embedding": self.query_embedding,
            "topic_hints": self.topic_hints,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "hop_count": self.hop_count,
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueryMessage":
        return cls(
            message_id=d.get("message_id", str(uuid.uuid4())),
            sender_peer_id=d.get("sender_peer_id", ""),
            timestamp=d.get("timestamp", time.time()),
            query_id=d.get("query_id", ""),
            query_embedding=d.get("query_embedding", []),
            topic_hints=d.get("topic_hints", []),
            similarity_threshold=d.get("similarity_threshold", 0.35),
            top_k=d.get("top_k", 10),
            hop_count=d.get("hop_count", 0),
            ttl=d.get("ttl", 10),
        )
    
    @classmethod
    def from_request(cls, request) -> "QueryMessage":
        """Create from a QueryRequest."""
        return cls(
            query_id=request.query_id,
            query_embedding=request.embedding,
            topic_hints=request.topic_hints,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k,
            hop_count=request.hop_count,
            ttl=request.ttl,
        )


@dataclass
class QueryResponseMessage(BaseMessage):
    """Response containing query results from a peer."""
    query_id: str = ""
    results: list[dict] = field(default_factory=list)  # Serialized SearchResults
    routing_path: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "query_response",
            "message_id": self.message_id,
            "sender_peer_id": self.sender_peer_id,
            "timestamp": self.timestamp,
            "query_id": self.query_id,
            "results": self.results,
            "routing_path": self.routing_path,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueryResponseMessage":
        return cls(
            message_id=d.get("message_id", str(uuid.uuid4())),
            sender_peer_id=d.get("sender_peer_id", ""),
            timestamp=d.get("timestamp", time.time()),
            query_id=d.get("query_id", ""),
            results=d.get("results", []),
            routing_path=d.get("routing_path", []),
            latency_ms=d.get("latency_ms", {}),
            success=d.get("success", True),
            error=d.get("error"),
        )
    
    def get_search_results(self) -> list[SearchResult]:
        """Convert results to SearchResult objects."""
        return [SearchResult.from_dict(r) for r in self.results]


@dataclass
class RegistrationMessage(BaseMessage):
    """Peer registration message."""
    peer_id: str = ""
    address: str = ""
    port: int = 0
    communities: list[str] = field(default_factory=list)
    is_super_peer: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "registration",
            "message_id": self.message_id,
            "sender_peer_id": self.sender_peer_id,
            "timestamp": self.timestamp,
            "peer_id": self.peer_id,
            "address": self.address,
            "port": self.port,
            "communities": self.communities,
            "is_super_peer": self.is_super_peer,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegistrationMessage":
        return cls(
            message_id=d.get("message_id", str(uuid.uuid4())),
            sender_peer_id=d.get("sender_peer_id", ""),
            timestamp=d.get("timestamp", time.time()),
            peer_id=d.get("peer_id", ""),
            address=d.get("address", ""),
            port=d.get("port", 0),
            communities=d.get("communities", []),
            is_super_peer=d.get("is_super_peer", False),
        )


@dataclass
class HeartbeatMessage(BaseMessage):
    """Heartbeat message for liveness checking."""
    peer_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "heartbeat",
            "message_id": self.message_id,
            "sender_peer_id": self.sender_peer_id,
            "timestamp": self.timestamp,
            "peer_id": self.peer_id,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeartbeatMessage":
        return cls(
            message_id=d.get("message_id", str(uuid.uuid4())),
            sender_peer_id=d.get("sender_peer_id", ""),
            timestamp=d.get("timestamp", time.time()),
            peer_id=d.get("peer_id", ""),
        )


def deserialize_message(data: bytes) -> BaseMessage:
    """Deserialize a message from bytes, detecting type automatically."""
    d = msgpack.unpackb(data)
    msg_type = d.get("type", "")
    
    type_map = {
        "query": QueryMessage,
        "query_response": QueryResponseMessage,
        "registration": RegistrationMessage,
        "heartbeat": HeartbeatMessage,
    }
    
    cls = type_map.get(msg_type)
    if cls:
        return cls.from_dict(d)
    
    raise ValueError(f"Unknown message type: {msg_type}")
