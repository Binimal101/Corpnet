"""Community summarization using LLM.

Generates text summaries that serve TWO purposes:
1. EMBEDDING (for routing): Summary text is embedded into a vector for similarity search
2. TEXT (for generation): Summary provides LLM context for answer generation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.embeddings import EmbeddingProvider
    from src.core.llm import LLMProvider

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """Generate summaries for communities using LLM.
    
    Returns both text summary and embedding for routing.
    """
    
    def __init__(self, llm: "LLMProvider", embedder: "EmbeddingProvider"):
        self.llm = llm
        self.embedder = embedder
    
    async def summarize(
        self,
        entities: list[str],
        relations: list[str],
    ) -> tuple[str, list[float]]:
        """Generate a summary for a community.
        
        Args:
            entities: List of entity names/descriptions.
            relations: List of relation descriptions.
        
        Returns:
            Tuple of (text_summary, summary_embedding).
        """
        text_summary = self.llm.summarize_community(entities, relations)
        summary_embedding = self.embedder.embed_text(text_summary)
        
        logger.debug(f"Summarized community with {len(entities)} entities")
        return text_summary, summary_embedding
    
    def summarize_sync(
        self,
        entities: list[str],
        relations: list[str],
    ) -> tuple[str, list[float]]:
        """Synchronous version of summarize."""
        text_summary = self.llm.summarize_community(entities, relations)
        summary_embedding = self.embedder.embed_text(text_summary)
        return text_summary, summary_embedding
