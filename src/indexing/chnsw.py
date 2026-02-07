"""C-HNSW (Community Hierarchical Navigable Small World) index.

One VectorStore per hierarchy level storing community summary embeddings.
Search uses pure embedding similarity — no text processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.types import Community, SearchResult

if TYPE_CHECKING:
    from src.indexing.vector_store import VectorStore
    from src.clustering.hierarchy import CommunityHierarchy

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalSearchResult:
    """Results from a hierarchical C-HNSW search."""
    results_by_layer: dict[int, list[SearchResult]] = field(default_factory=dict)
    traversal_path: list[str] = field(default_factory=list)
    
    @property
    def all_results(self) -> list[SearchResult]:
        """Flatten results across layers, sorted by score."""
        all_r = []
        for results in self.results_by_layer.values():
            all_r.extend(results)
        all_r.sort(key=lambda r: r.score, reverse=True)
        return all_r


class CHNSWIndex:
    """C-HNSW index: one VectorStore per hierarchy level.
    
    Each layer stores community summary EMBEDDINGS (not text).
    Search is pure vector similarity — cosine distance between
    query embedding and stored community embeddings.
    """
    
    def __init__(self):
        self.layers: dict[int, "VectorStore"] = {}  # level -> vector store
        self.hierarchy: "CommunityHierarchy | None" = None
        self.community_store: dict[str, Community] = {}  # community_id -> Community
    
    def build(
        self,
        hierarchy: "CommunityHierarchy",
        layer_stores: dict[int, "VectorStore"],
    ) -> None:
        """Build the C-HNSW index from a community hierarchy.
        
        Args:
            hierarchy: The community hierarchy.
            layer_stores: One VectorStore per level (level -> store).
        """
        self.hierarchy = hierarchy
        self.layers = layer_stores
        
        # Index all communities
        for level in range(hierarchy.num_levels):
            for community in hierarchy.get_level(level):
                self.community_store[community.community_id] = community
        
        logger.info(f"Built C-HNSW with {len(self.layers)} layers, {len(self.community_store)} communities")
    
    async def search_layer(
        self,
        level: int,
        query_embedding: list[float],
        similarity_threshold: float,
        candidates: list[Community] | None = None,
    ) -> list[Community]:
        """Search a single hierarchy layer by embedding similarity.
        
        Returns communities with similarity >= threshold, sorted descending.
        Always returns at least 1 result (best match) even if below threshold.
        """
        if level not in self.layers:
            return []
        
        layer_store = self.layers[level]
        
        # Search with threshold
        results = layer_store.search_by_threshold(
            embedding=query_embedding,
            threshold=similarity_threshold,
        )
        
        # Filter to candidates if provided
        if candidates:
            candidate_ids = {c.community_id for c in candidates}
            results = [r for r in results if r.chunk_id in candidate_ids]
        
        # Convert results back to Community objects
        communities = []
        for result in results:
            community = self.community_store.get(result.chunk_id)
            if community:
                communities.append(community)
        
        return communities
    
    def search_layer_sync(
        self,
        level: int,
        query_embedding: list[float],
        similarity_threshold: float,
        candidates: list[Community] | None = None,
    ) -> list[Community]:
        """Synchronous version of search_layer."""
        import asyncio
        return asyncio.run(self.search_layer(level, query_embedding, similarity_threshold, candidates))
    
    def search(
        self,
        query_embedding: list[float],
        top_k_per_layer: int = 5,
    ) -> HierarchicalSearchResult:
        """Search all layers top-down, returning results from each."""
        result = HierarchicalSearchResult()
        
        if not self.layers or not self.hierarchy:
            return result
        
        for level in range(self.hierarchy.num_levels - 1, -1, -1):
            if level not in self.layers:
                continue
            
            store = self.layers[level]
            layer_results = store.search(query_embedding, top_k=top_k_per_layer)
            result.results_by_layer[level] = layer_results
            
            for r in layer_results:
                result.traversal_path.append(r.chunk_id)
        
        logger.info(f"C-HNSW search: {len(result.results_by_layer)} layers, {len(result.all_results)} results")
        return result
