"""Tests for networking module."""

import pytest

from src.networking.messages import QueryMessage, QueryResponseMessage, deserialize_message
from src.networking.peer import Peer, PeerState
from src.networking.coordinator import Coordinator
from src.core.types import PeerInfo


class TestMessages:
    """Tests for wire protocol messages."""
    
    def test_query_message_serialization(self):
        """Test QueryMessage to_bytes and from_bytes."""
        msg = QueryMessage(
            query_id="test-query",
            query_embedding=[0.1] * 10,
            topic_hints=["python", "web"],
            similarity_threshold=0.4,
            top_k=5,
        )
        
        data = msg.to_bytes()
        restored = QueryMessage.from_bytes(data)
        
        assert restored.query_id == msg.query_id
        assert restored.query_embedding == msg.query_embedding
        assert restored.topic_hints == msg.topic_hints
        assert restored.similarity_threshold == msg.similarity_threshold
    
    def test_query_response_serialization(self):
        """Test QueryResponseMessage serialization."""
        msg = QueryResponseMessage(
            query_id="test-query",
            results=[{"chunk_id": "c1", "text": "test", "score": 0.9, "source": "doc", "peer_id": "p1", "metadata": {}}],
            routing_path=["comm1", "comm2"],
            success=True,
        )
        
        data = msg.to_bytes()
        restored = QueryResponseMessage.from_bytes(data)
        
        assert restored.query_id == msg.query_id
        assert len(restored.results) == 1
        assert restored.routing_path == msg.routing_path
    
    def test_deserialize_message(self):
        """Test automatic message type detection."""
        msg = QueryMessage(query_id="test")
        data = msg.to_bytes()
        
        restored = deserialize_message(data)
        
        assert isinstance(restored, QueryMessage)
        assert restored.query_id == "test"


class TestPeer:
    """Tests for peer nodes."""
    
    def test_peer_creation(self):
        """Test creating a peer."""
        peer = Peer(
            peer_id="test-peer",
            address="localhost",
            port=50051,
        )
        
        assert peer.peer_id == "test-peer"
        assert peer.state == PeerState.DISCONNECTED
    
    def test_peer_to_peer_info(self):
        """Test converting peer to PeerInfo."""
        peer = Peer(
            peer_id="test-peer",
            address="localhost",
            port=50051,
            is_super_peer=True,
            communities=["c1", "c2"],
        )
        
        info = peer.to_peer_info()
        
        assert info.peer_id == "test-peer"
        assert info.is_super_peer is True
        assert info.communities == ["c1", "c2"]
    
    def test_peer_registration(self):
        """Test registering known peers."""
        peer = Peer(peer_id="p1", address="localhost", port=50051)
        
        other = PeerInfo(
            peer_id="p2",
            address="localhost",
            port=50052,
            communities=["c1"],
        )
        
        peer.register_peer(other)
        
        assert peer.get_peer("p2") is not None
        assert len(peer.get_peers_for_community("c1")) == 1


class TestCoordinator:
    """Tests for coordinator."""
    
    def test_register_peer(self):
        """Test peer registration."""
        coord = Coordinator()
        
        peer = PeerInfo(
            peer_id="p1",
            address="localhost",
            port=50051,
        )
        
        result = coord.register_peer(peer)
        
        assert result is True
        assert "p1" in coord.peers
    
    def test_super_peer_tracking(self):
        """Test super-peer tracking."""
        coord = Coordinator()
        
        peer = PeerInfo(
            peer_id="sp1",
            address="localhost",
            port=50051,
            is_super_peer=True,
        )
        
        coord.register_peer(peer)
        
        assert "sp1" in coord.super_peers
        assert coord.get_super_peer() is not None
    
    def test_unregister_peer(self):
        """Test peer unregistration."""
        coord = Coordinator()
        
        peer = PeerInfo(peer_id="p1", address="localhost", port=50051, is_super_peer=True)
        coord.register_peer(peer)
        
        assert coord.unregister_peer("p1") is True
        assert "p1" not in coord.peers
        assert "p1" not in coord.super_peers
    
    def test_heartbeat(self):
        """Test heartbeat update."""
        coord = Coordinator()
        
        peer = PeerInfo(peer_id="p1", address="localhost", port=50051)
        coord.register_peer(peer)
        
        import time
        old_heartbeat = coord.peers["p1"].last_heartbeat
        time.sleep(0.01)
        
        coord.heartbeat("p1")
        
        assert coord.peers["p1"].last_heartbeat > old_heartbeat
