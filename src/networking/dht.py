"""Distributed hash table for peer discovery.

Provides decentralized peer discovery as an alternative to
coordinator-based discovery.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from src.core.types import PeerInfo

logger = logging.getLogger(__name__)


def hash_key(key: str) -> int:
    """Hash a key to a 160-bit integer (SHA-1)."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16)


@dataclass
class DHTNode:
    """A node in the distributed hash table."""
    
    peer_id: str
    node_id: int = 0  # Position in key space
    
    # Routing table: buckets of peers by distance
    buckets: list[list[PeerInfo]] = field(default_factory=list)
    bucket_size: int = 20  # k in Kademlia
    
    # Local storage
    storage: dict[str, PeerInfo] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = hash_key(self.peer_id)
        # Initialize 160 buckets (one per bit of node ID)
        if not self.buckets:
            self.buckets = [[] for _ in range(160)]
    
    def distance(self, other_id: int) -> int:
        """XOR distance between node IDs."""
        return self.node_id ^ other_id
    
    def bucket_index(self, node_id: int) -> int:
        """Get bucket index for a node ID based on distance."""
        dist = self.distance(node_id)
        if dist == 0:
            return 0
        return dist.bit_length() - 1
    
    def add_peer(self, peer: PeerInfo) -> bool:
        """Add a peer to the routing table."""
        peer_node_id = hash_key(peer.peer_id)
        bucket_idx = self.bucket_index(peer_node_id)
        
        bucket = self.buckets[bucket_idx]
        
        # Check if already present
        for i, existing in enumerate(bucket):
            if existing.peer_id == peer.peer_id:
                # Move to end (most recently seen)
                bucket.pop(i)
                bucket.append(peer)
                return True
        
        # Add if bucket not full
        if len(bucket) < self.bucket_size:
            bucket.append(peer)
            return True
        
        # Bucket full — could ping oldest and replace if dead
        return False
    
    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from the routing table."""
        peer_node_id = hash_key(peer_id)
        bucket_idx = self.bucket_index(peer_node_id)
        
        bucket = self.buckets[bucket_idx]
        for i, peer in enumerate(bucket):
            if peer.peer_id == peer_id:
                bucket.pop(i)
                return True
        return False
    
    def find_closest(self, key: str, count: int = 10) -> list[PeerInfo]:
        """Find the closest peers to a key."""
        key_id = hash_key(key)
        
        # Collect all known peers with distances
        all_peers: list[tuple[int, PeerInfo]] = []
        for bucket in self.buckets:
            for peer in bucket:
                peer_id = hash_key(peer.peer_id)
                dist = key_id ^ peer_id
                all_peers.append((dist, peer))
        
        # Sort by distance and return closest
        all_peers.sort(key=lambda x: x[0])
        return [peer for _, peer in all_peers[:count]]
    
    def store(self, key: str, peer: PeerInfo) -> None:
        """Store a peer at a key."""
        self.storage[key] = peer
    
    def lookup(self, key: str) -> PeerInfo | None:
        """Lookup a peer by key."""
        return self.storage.get(key)
    
    def get_all_peers(self) -> list[PeerInfo]:
        """Get all peers in the routing table."""
        peers = []
        for bucket in self.buckets:
            peers.extend(bucket)
        return peers
