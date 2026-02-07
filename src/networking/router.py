"""Hierarchical router implementing recursive descent routing.

The super-peer orchestrates the entire query — queries cascade DOWN
through the hierarchy, results bubble BACK UP.

Query flow:
1. Client sends query to super-peer
2. Super-peer searches community embeddings at its level
3. Forwards query to child peers for matching communities
4. Children recursively search their levels
5. Leaf peers search local vector stores
6. Results bubble back up through the hierarchy
7. Super-peer aggregates and returns to client
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from src.core.types import Community, PeerInfo, QueryRequest, SearchResult

if TYPE_CHECKING:
    from src.networking.peer import Peer
    from src.networking.messages import QueryMessage, QueryResponseMessage
    from src.indexing.chnsw import CHNSWIndex
    from src.query.reranker import Reranker

logger = logging.getLogger(__name__)


class HierarchicalRouter:
    """Recursive descent router for hierarchical query routing.
    
    Each peer at each layer:
    1. Receives the query
    2. Searches its own community embeddings (threshold-based)
    3. Forwards to child peers for matching communities
    4. Collects results from children
    5. Returns merged results to its parent
    """
    
    def __init__(
        self,
        peer: "Peer",
        chnsw: "CHNSWIndex | None" = None,
        reranker: "Reranker | None" = None,
        min_communities: int = 1,
        max_communities: int = 20,
    ):
        self.peer = peer
        self.chnsw = chnsw
        self.reranker = reranker
        self.min_communities = min_communities
        self.max_communities = max_communities
    
    async def handle_query(self, query: QueryRequest) -> list[SearchResult]:
        """Handle a query at this peer.
        
        This is the main entry point called when a query arrives.
        """
        start = time.time()
        
        # Check TTL
        if query.ttl <= 0:
            logger.warning(f"Query {query.query_id} exceeded TTL")
            return []
        
        # Decrement hop count and TTL
        query.hop_count += 1
        query.ttl -= 1
        
        # Am I a leaf peer?
        if self.peer.is_leaf:
            results = await self._search_local(query)
            logger.info(f"Leaf search: {len(results)} results in {(time.time()-start)*1000:.1f}ms")
            return results
        
        # I'm a routing peer — find matching child communities
        matching_communities = await self._search_community_embeddings(query)
        
        if not matching_communities:
            logger.info(f"No matching communities for query {query.query_id}")
            return []
        
        # Forward query to peers owning those communities
        all_results = await self._forward_to_children(query, matching_communities)
        
        # Merge and re-rank results
        merged = self._merge_results(all_results, query.top_k)
        
        logger.info(
            f"Routing complete: {len(matching_communities)} communities, "
            f"{len(all_results)} raw results, {len(merged)} merged in "
            f"{(time.time()-start)*1000:.1f}ms"
        )
        
        return merged
    
    async def _search_local(self, query: QueryRequest) -> list[SearchResult]:
        """Search local vector store (leaf peer)."""
        return await self.peer.search_local(query.embedding, top_k=query.top_k)
    
    async def _search_community_embeddings(self, query: QueryRequest) -> list[Community]:
        """Search community embeddings at this peer's level."""
        if not self.chnsw or not self.peer.hierarchy:
            return []
        
        # Determine my level in the hierarchy
        my_level = self._get_my_level()
        if my_level < 0:
            return []
        
        # Get communities at my level that I own
        my_communities = [
            c for c in self.peer.hierarchy.get_level(my_level)
            if c.community_id in self.peer.communities
        ]
        
        # Search for matches using threshold
        matching = await self.chnsw.search_layer(
            level=my_level,
            query_embedding=query.embedding,
            similarity_threshold=query.similarity_threshold,
            candidates=my_communities,
        )
        
        # Apply safety bounds
        if len(matching) < self.min_communities and matching:
            matching = matching[:1]
        elif len(matching) > self.max_communities:
            matching = matching[:self.max_communities]
        
        return matching
    
    async def _forward_to_children(
        self,
        query: QueryRequest,
        communities: list[Community],
    ) -> list[SearchResult]:
        """Forward query to peers owning the matched communities."""
        # Group by peer
        peer_communities: dict[str, list[Community]] = {}
        for community in communities:
            peer_id = community.peer_id
            if peer_id:
                if peer_id not in peer_communities:
                    peer_communities[peer_id] = []
                peer_communities[peer_id].append(community)
        
        # Parallel dispatch to child peers
        all_results: list[SearchResult] = []
        
        # TODO: Implement actual network transport
        # For now, just return empty results
        # In production, this would use gRPC or similar to forward queries
        
        logger.debug(f"Would forward to {len(peer_communities)} peers")
        
        return all_results
    
    def _merge_results(self, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Merge and re-rank results from multiple sources."""
        if self.reranker:
            return self.reranker.merge_and_rerank(results, top_k)
        
        # Simple sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def _get_my_level(self) -> int:
        """Determine what level this peer operates at in the hierarchy."""
        if not self.peer.hierarchy:
            return -1
        
        # Find the level containing my communities
        for level in range(self.peer.hierarchy.num_levels - 1, -1, -1):
            level_communities = self.peer.hierarchy.get_level(level)
            for c in level_communities:
                if c.community_id in self.peer.communities:
                    return level
        
        return -1
    
    def handle_query_sync(self, query: QueryRequest) -> list[SearchResult]:
        """Synchronous version of handle_query."""
        return asyncio.run(self.handle_query(query))
