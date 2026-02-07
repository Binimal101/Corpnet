"""Query engine: client-side entrypoint for queries.

The query engine is the CLIENT-SIDE entrypoint. It:
1. Embeds the query
2. Extracts topics
3. Sends to a super-peer (which handles routing + aggregation)
4. Receives results and generates an answer

The super-peer (via HierarchicalRouter) handles the actual routing.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from src.core.types import QueryRequest, QueryResponse, SearchResult

if TYPE_CHECKING:
    from src.core.embeddings import EmbeddingProvider
    from src.core.llm import LLMProvider
    from src.core.config import Settings
    from src.networking.peer import Peer
    from src.query.generator import AnswerGenerator

logger = logging.getLogger(__name__)


class QueryEngine:
    """Client-side query entrypoint.
    
    Sends query to a super-peer and receives aggregated results.
    """
    
    def __init__(
        self,
        embedder: "EmbeddingProvider",
        llm: "LLMProvider",
        generator: "AnswerGenerator | None" = None,
        super_peer: "Peer | None" = None,
        settings: "Settings | None" = None,
    ):
        self.embedder = embedder
        self.llm = llm
        self.generator = generator
        self.super_peer = super_peer
        self.settings = settings
    
    async def query(
        self,
        text: str,
        top_k: int = 10,
        similarity_threshold: float | None = None,
    ) -> QueryResponse:
        """Execute a full query pipeline.
        
        Args:
            text: The user's query text.
            top_k: Number of results to return.
            similarity_threshold: Override for routing threshold.
        
        Returns:
            QueryResponse with answer, results, and observability data.
        """
        latency = {}
        start = time.time()
        query_id = str(uuid.uuid4())
        
        # 1. Embed query
        t0 = time.time()
        embedding = self.embedder.embed_text(text)
        latency["embed"] = (time.time() - t0) * 1000
        
        # 2. Extract topics
        t0 = time.time()
        topics = self.llm.extract_topics(text)
        latency["extract_topics"] = (time.time() - t0) * 1000
        logger.info(f"Extracted topics: {topics}")
        
        # 3. Build request
        threshold = similarity_threshold
        if threshold is None and self.settings:
            threshold = self.settings.routing.similarity_threshold
        if threshold is None:
            threshold = 0.35
        
        request = QueryRequest(
            query_id=query_id,
            text=text,
            embedding=embedding,
            topic_hints=topics,
            similarity_threshold=threshold,
            top_k=top_k,
        )
        
        # 4. Route through super-peer
        t0 = time.time()
        results: list[SearchResult] = []
        routing_path: list[str] = []
        total_messages = 0
        
        if self.super_peer:
            # TODO: Send via transport layer
            # For now, search directly if super_peer has a router
            results = await self.super_peer.search_local(embedding, top_k)
            latency["route_and_search"] = (time.time() - t0) * 1000
        else:
            latency["route_and_search"] = 0
            logger.warning("No super-peer configured, returning empty results")
        
        # 5. Generate answer
        t0 = time.time()
        if self.generator and results:
            answer = await self.generator.generate(text, results)
        else:
            answer = self._simple_answer(text, results)
        latency["generate"] = (time.time() - t0) * 1000
        
        latency["total"] = (time.time() - start) * 1000
        
        return QueryResponse(
            query_id=query_id,
            answer=answer,
            results=results,
            routing_path=routing_path,
            latency_ms=latency,
            total_messages=total_messages,
        )
    
    def query_sync(
        self,
        text: str,
        top_k: int = 10,
        similarity_threshold: float | None = None,
    ) -> QueryResponse:
        """Synchronous version of query."""
        import asyncio
        return asyncio.run(self.query(text, top_k, similarity_threshold))
    
    def _simple_answer(self, query: str, results: list[SearchResult]) -> str:
        """Generate a simple answer without LLM."""
        if not results:
            return "No relevant information found."
        
        context = "\n\n".join([f"[{r.source}]: {r.text[:200]}..." for r in results[:3]])
        return f"Based on the following sources:\n\n{context}"
