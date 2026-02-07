"""Coordinator for peer registration and hierarchy management.

The coordinator:
- Tracks all peers in the network
- Manages super-peer election
- Publishes hierarchy updates
- Routes initial queries to appropriate super-peers
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.types import PeerInfo

if TYPE_CHECKING:
    from src.clustering.hierarchy import CommunityHierarchy

logger = logging.getLogger(__name__)


@dataclass
class Coordinator:
    """Central coordinator for the DAC-HRAG network.
    
    Manages peer registration, super-peer election, and hierarchy updates.
    """
    
    address: str = "localhost"
    port: int = 50050
    
    # Peer registry
    peers: dict[str, PeerInfo] = field(default_factory=dict)
    super_peers: list[str] = field(default_factory=list)
    
    # Hierarchy state
    hierarchy: "CommunityHierarchy | None" = None
    hierarchy_version: int = 0
    
    def register_peer(self, peer_info: PeerInfo) -> bool:
        """Register a new peer."""
        self.peers[peer_info.peer_id] = peer_info
        logger.info(f"Registered peer {peer_info.peer_id} at {peer_info.address}:{peer_info.port}")
        
        # Check if this should be a super-peer
        if peer_info.is_super_peer and peer_info.peer_id not in self.super_peers:
            self.super_peers.append(peer_info.peer_id)
            logger.info(f"Added super-peer {peer_info.peer_id}")
        
        return True
    
    def unregister_peer(self, peer_id: str) -> bool:
        """Unregister a peer."""
        if peer_id in self.peers:
            del self.peers[peer_id]
            if peer_id in self.super_peers:
                self.super_peers.remove(peer_id)
            logger.info(f"Unregistered peer {peer_id}")
            return True
        return False
    
    def heartbeat(self, peer_id: str) -> bool:
        """Update peer heartbeat."""
        if peer_id in self.peers:
            self.peers[peer_id].last_heartbeat = time.time()
            return True
        return False
    
    def get_super_peer(self) -> PeerInfo | None:
        """Get an available super-peer for query routing."""
        for sp_id in self.super_peers:
            peer = self.peers.get(sp_id)
            if peer and self._is_alive(peer):
                return peer
        return None
    
    def get_all_peers(self) -> list[PeerInfo]:
        """Get all registered peers."""
        return list(self.peers.values())
    
    def get_alive_peers(self) -> list[PeerInfo]:
        """Get all alive peers."""
        return [p for p in self.peers.values() if self._is_alive(p)]
    
    def update_hierarchy(self, hierarchy: "CommunityHierarchy") -> None:
        """Update the global hierarchy."""
        self.hierarchy = hierarchy
        self.hierarchy_version += 1
        logger.info(f"Updated hierarchy to version {self.hierarchy_version}")
    
    def assign_communities(self) -> dict[str, list[str]]:
        """Assign communities to peers.
        
        Returns mapping of peer_id -> list of community_ids.
        """
        if not self.hierarchy:
            return {}
        
        # Simple round-robin assignment
        alive_peers = self.get_alive_peers()
        if not alive_peers:
            return {}
        
        assignments: dict[str, list[str]] = {p.peer_id: [] for p in alive_peers}
        
        for level in range(self.hierarchy.num_levels):
            for i, community in enumerate(self.hierarchy.get_level(level)):
                peer = alive_peers[i % len(alive_peers)]
                assignments[peer.peer_id].append(community.community_id)
        
        return assignments
    
    def _is_alive(self, peer: PeerInfo, timeout: float = 60.0) -> bool:
        """Check if a peer is alive based on heartbeat."""
        return (time.time() - peer.last_heartbeat) < timeout
    
    def cleanup_dead_peers(self, timeout: float = 60.0) -> list[str]:
        """Remove dead peers from registry."""
        dead = [pid for pid, peer in self.peers.items() if not self._is_alive(peer, timeout)]
        for pid in dead:
            self.unregister_peer(pid)
        return dead
