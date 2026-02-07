"""Recall canary for monitoring index quality.

Periodically runs known queries to verify recall stays above threshold.
Triggers re-clustering if recall drops.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class CanaryQuery:
    """A canary query with known ground truth results."""
    query_id: str
    embedding: list[float]
    expected_chunk_ids: list[str]


@dataclass
class RecallResult:
    """Result of a recall check."""
    query_id: str
    recall_at_k: float
    retrieved_count: int
    expected_count: int
    timestamp: float = field(default_factory=time.time)


class RecallCanary:
    """Monitors index recall using canary queries.
    
    Runs periodic checks and alerts if recall drops below threshold.
    """
    
    def __init__(
        self,
        vector_store: "VectorStore",
        recall_threshold: float = 0.85,
        k: int = 10,
    ):
        self.vector_store = vector_store
        self.recall_threshold = recall_threshold
        self.k = k
        self.canary_queries: list[CanaryQuery] = []
        self.results: list[RecallResult] = []
    
    def add_canary(self, query: CanaryQuery) -> None:
        """Add a canary query."""
        self.canary_queries.append(query)
    
    def check_recall(self) -> list[RecallResult]:
        """Run all canary queries and check recall.
        
        Returns list of results. Logs warnings if recall drops below threshold.
        """
        results = []
        
        for canary in self.canary_queries:
            # Run search
            search_results = self.vector_store.search(canary.embedding, top_k=self.k)
            retrieved_ids = {r.chunk_id for r in search_results}
            
            # Calculate recall
            expected_set = set(canary.expected_chunk_ids[:self.k])
            if expected_set:
                recall = len(retrieved_ids & expected_set) / len(expected_set)
            else:
                recall = 1.0
            
            result = RecallResult(
                query_id=canary.query_id,
                recall_at_k=recall,
                retrieved_count=len(retrieved_ids),
                expected_count=len(expected_set),
            )
            results.append(result)
            
            if recall < self.recall_threshold:
                logger.warning(
                    f"Recall canary {canary.query_id} below threshold: "
                    f"{recall:.2%} < {self.recall_threshold:.2%}"
                )
        
        self.results.extend(results)
        
        # Log summary
        if results:
            avg_recall = sum(r.recall_at_k for r in results) / len(results)
            logger.info(f"Recall check: {len(results)} queries, avg recall {avg_recall:.2%}")
        
        return results
    
    def needs_recluster(self) -> bool:
        """Check if recent results indicate re-clustering is needed."""
        if not self.results:
            return False
        
        # Check last 5 results
        recent = self.results[-5:]
        avg_recall = sum(r.recall_at_k for r in recent) / len(recent)
        
        return avg_recall < self.recall_threshold
    
    def get_avg_recall(self, window: int = 10) -> float:
        """Get average recall over recent queries."""
        if not self.results:
            return 1.0
        
        recent = self.results[-window:]
        return sum(r.recall_at_k for r in recent) / len(recent)
