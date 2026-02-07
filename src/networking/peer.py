"""Peer node implementation.

A peer is a single node in the P2P network. It can be:
- A leaf peer: owns data, runs local searches
- A super-peer: owns communities, routes queries to children
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from src.core.types import PeerInfo, SearchResult

if TYPE_CHECKING:
    from src.indexing.vector_store import VectorStore
    from src.clustering.hierarchy import CommunityHierarchy

logger = logging.getLogger(__name__)


class PeerState(Enum):
    """State of a peer in the network."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LEADER = "leader"  # Super-peer


@dataclass
class Peer:
    """A peer node in the DAC-HRAG network.
    
    Can function as:
    - Leaf peer: searches local vector store
    - Super-peer: routes queries through hierarchy
    """
    
    peer_id: str
    address: str = "localhost"
    port: int = 50051
    is_super_peer: bool = False
    is_leaf: bool = True
    state: PeerState = PeerState.DISCONNECTED
    
    # Owned data
    vector_store: "VectorStore | None" = None
    hierarchy: "CommunityHierarchy | None" = None
    communities: list[str] = field(default_factory=list)
    
    # Network state
    known_peers: dict[str, PeerInfo] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    
    def to_peer_info(self) -> PeerInfo:
        """Convert to PeerInfo for network communication."""
        return PeerInfo(
            peer_id=self.peer_id,
            address=self.address,
            port=self.port,
            communities=list(self.communities),
            is_super_peer=self.is_super_peer,
            last_heartbeat=self.last_heartbeat,
        )
    
    def register_peer(self, peer_info: PeerInfo) -> None:
        """Register a known peer."""
        self.known_peers[peer_info.peer_id] = peer_info
        logger.debug(f"Registered peer {peer_info.peer_id}")
    
    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer from known peers."""
        self.known_peers.pop(peer_id, None)
        logger.debug(f"Unregistered peer {peer_id}")
    
    def get_peer(self, peer_id: str) -> PeerInfo | None:
        """Get a known peer by ID."""
        return self.known_peers.get(peer_id)
    
    def get_super_peers(self) -> list[PeerInfo]:
        """Get all known super-peers."""
        return [p for p in self.known_peers.values() if p.is_super_peer]
    
    def get_peers_for_community(self, community_id: str) -> list[PeerInfo]:
        """Get peers that own a specific community."""
        return [p for p in self.known_peers.values() if community_id in p.communities]
    
    async def search_local(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search local vector store."""
        if not self.vector_store:
            logger.warning(f"Peer {self.peer_id} has no vector store")
            return []
        
        results = self.vector_store.search(embedding, top_k=top_k)
        
        # Tag results with this peer's ID
        for r in results:
            r.peer_id = self.peer_id
        
        return results
    
    def search_local_sync(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Synchronous local search."""
        import asyncio
        return asyncio.run(self.search_local(embedding, top_k))
    
    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def is_alive(self, timeout: float = 60.0) -> bool:
        """Check if peer is alive based on last heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout
