"""Local in-process transport for testing.

Allows multiple peers to communicate without network I/O.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.networking.messages import BaseMessage, QueryMessage, QueryResponseMessage

if TYPE_CHECKING:
    from src.networking.peer import Peer

logger = logging.getLogger(__name__)


class LocalTransport:
    """In-process transport for testing peer communication.
    
    All peers share a registry and communicate via direct method calls.
    """
    
    def __init__(self):
        self.peers: dict[str, "Peer"] = {}
        self.routers: dict[str, any] = {}  # peer_id -> HierarchicalRouter
    
    def register(self, peer: "Peer", router: any = None) -> None:
        """Register a peer with the transport."""
        self.peers[peer.peer_id] = peer
        if router:
            self.routers[peer.peer_id] = router
        logger.debug(f"Registered peer {peer.peer_id} with local transport")
    
    def unregister(self, peer_id: str) -> None:
        """Unregister a peer."""
        self.peers.pop(peer_id, None)
        self.routers.pop(peer_id, None)
    
    async def send(self, to_peer_id: str, message: BaseMessage) -> QueryResponseMessage | None:
        """Send a message to a peer and wait for response."""
        peer = self.peers.get(to_peer_id)
        if not peer:
            logger.warning(f"Peer {to_peer_id} not found")
            return None
        
        router = self.routers.get(to_peer_id)
        
        if isinstance(message, QueryMessage):
            # Convert to QueryRequest and route
            from src.core.types import QueryRequest
            request = QueryRequest(
                query_id=message.query_id,
                text="",  # Not needed for routing
                embedding=message.query_embedding,
                topic_hints=message.topic_hints,
                similarity_threshold=message.similarity_threshold,
                top_k=message.top_k,
                hop_count=message.hop_count,
                ttl=message.ttl,
            )
            
            if router:
                results = await router.handle_query(request)
            else:
                results = await peer.search_local(message.query_embedding, message.top_k)
            
            return QueryResponseMessage(
                query_id=message.query_id,
                sender_peer_id=to_peer_id,
                results=[r.to_dict() for r in results],
                success=True,
            )
        
        return None
    
    async def broadcast(
        self,
        peer_ids: list[str],
        message: BaseMessage,
    ) -> list[QueryResponseMessage]:
        """Send a message to multiple peers in parallel."""
        tasks = [self.send(pid, message) for pid in peer_ids]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for resp in responses:
            if isinstance(resp, QueryResponseMessage):
                results.append(resp)
            elif isinstance(resp, Exception):
                logger.warning(f"Broadcast failed: {resp}")
        
        return results
