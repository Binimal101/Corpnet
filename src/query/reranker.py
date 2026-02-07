"""Result reranking and aggregation.

Combines:
- Result merging from multiple peers
- Deduplication
- Adaptive filtering based on similarity
- Re-ranking by relevance
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.core.types import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Merges, deduplicates, and re-ranks search results.
    
    Used by super-peers to aggregate results from child peers.
    """
    
    def __init__(
        self,
        relevance_threshold: float = 0.3,
        dedup_threshold: float = 0.95,
    ):
        self.relevance_threshold = relevance_threshold
        self.dedup_threshold = dedup_threshold
    
    def merge_and_rerank(
        self,
        results: list[SearchResult],
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Merge results from multiple sources, deduplicate, and re-rank.
        
        Args:
            results: All results from various sources.
            top_k: Number of results to return.
            query_embedding: Optional query embedding for re-scoring.
        
        Returns:
            Top-k deduplicated, re-ranked results.
        """
        if not results:
            return []
        
        # Deduplicate by chunk_id
        seen_ids: set[str] = set()
        unique: list[SearchResult] = []
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                unique.append(r)
        
        # Deduplicate by text similarity (catch near-duplicates)
        unique = self._dedup_by_text(unique)
        
        # Re-score if query embedding provided
        if query_embedding:
            unique = self._rescore(unique, query_embedding)
        
        # Filter by relevance threshold
        unique = [r for r in unique if r.score >= self.relevance_threshold]
        
        # Sort by score
        unique.sort(key=lambda r: r.score, reverse=True)
        
        logger.debug(f"Reranked: {len(results)} -> {len(unique)} results")
        return unique[:top_k]
    
    def _dedup_by_text(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove near-duplicate results based on text similarity."""
        if len(results) <= 1:
            return results
        
        # Simple character-based similarity
        unique: list[SearchResult] = [results[0]]
        
        for r in results[1:]:
            is_dup = False
            for existing in unique:
                sim = self._text_similarity(r.text, existing.text)
                if sim >= self.dedup_threshold:
                    is_dup = True
                    # Keep the one with higher score
                    if r.score > existing.score:
                        unique.remove(existing)
                        unique.append(r)
                    break
            
            if not is_dup:
                unique.append(r)
        
        return unique
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity (Jaccard on words)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _rescore(
        self,
        results: list[SearchResult],
        query_embedding: list[float],
    ) -> list[SearchResult]:
        """Re-score results using query embedding."""
        query = np.array(query_embedding)
        query = query / (np.linalg.norm(query) + 1e-10)
        
        for r in results:
            # If we had chunk embeddings stored, we'd use them here
            # For now, just use the existing score
            pass
        
        return results


def adaptive_filter(
    results: list[SearchResult],
    query_embedding: list[float] | None = None,
    threshold: float = 0.3,
    top_k: int = 10,
) -> list[SearchResult]:
    """Convenience function for adaptive filtering.
    
    Applies threshold-based filtering and returns top-k results.
    """
    reranker = Reranker(relevance_threshold=threshold)
    return reranker.merge_and_rerank(results, top_k, query_embedding)
