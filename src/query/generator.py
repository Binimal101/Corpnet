"""Answer generation using LLM.

Takes query and retrieved results, generates a coherent answer
with source attribution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.types import SearchResult

if TYPE_CHECKING:
    from src.core.llm import LLMProvider

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers from retrieved context using LLM."""
    
    def __init__(self, llm: "LLMProvider", max_context_chunks: int = 5):
        self.llm = llm
        self.max_context_chunks = max_context_chunks
    
    async def generate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        """Generate an answer from query and retrieved results.
        
        Args:
            query: The user's original query.
            results: Retrieved search results.
        
        Returns:
            Generated answer text.
        """
        if not results:
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from top results
        context_parts = []
        for i, result in enumerate(results[:self.max_context_chunks]):
            source = result.source or result.chunk_id[:8]
            context_parts.append(f"[Source {i+1}: {source}]\n{result.text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context, answer the question. Cite sources by number when using information from them.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            answer = self.llm.generate(prompt)
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._fallback_answer(results)
    
    def generate_sync(self, query: str, results: list[SearchResult]) -> str:
        """Synchronous version of generate."""
        import asyncio
        return asyncio.run(self.generate(query, results))
    
    def _fallback_answer(self, results: list[SearchResult]) -> str:
        """Generate a simple fallback answer without LLM."""
        if not results:
            return "No relevant information found."
        
        top = results[0]
        return f"Based on the retrieved information from {top.source or 'the knowledge base'}:\n\n{top.text[:500]}..."
