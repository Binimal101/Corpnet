"""Entity and relation extraction from document chunks.

Extracts entities and relations using LLM, storing them directly
on the DocumentChunk object.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from src.core.types import DocumentChunk

if TYPE_CHECKING:
    from src.core.llm import LLMProvider

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entities and relations from DocumentChunks using LLM.
    
    Results are stored DIRECTLY on the chunk object.
    """
    
    def __init__(self, llm: "LLMProvider"):
        self.llm = llm
    
    async def extract(self, chunk: DocumentChunk) -> DocumentChunk:
        """Extract entities and relations, storing them on the chunk.
        
        Mutates chunk in-place: populates entities and relations fields.
        """
        prompt = f"""Extract named entities and relationships from the following text.

Text:
{chunk.text}

Return in this exact format:
ENTITIES:
- entity1
- entity2
- entity3

RELATIONS:
- entity1 | relationship | entity2
- entity2 | relationship | entity3

Be concise. Only extract clear, explicit entities and relationships."""

        try:
            response = self.llm.generate(prompt)
            chunk.entities = self._parse_entities(response)
            chunk.relations = self._parse_relations(response)
            logger.debug(f"Extracted {len(chunk.entities)} entities, {len(chunk.relations)} relations from {chunk.chunk_id}")
        except Exception as e:
            logger.warning(f"Entity extraction failed for {chunk.chunk_id}: {e}")
            chunk.entities = []
            chunk.relations = []
        
        return chunk
    
    async def extract_batch(self, chunks: list[DocumentChunk], concurrency: int = 5) -> list[DocumentChunk]:
        """Batch extraction with concurrency limit."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def extract_with_limit(chunk: DocumentChunk) -> DocumentChunk:
            async with semaphore:
                return await self.extract(chunk)
        
        return await asyncio.gather(*[extract_with_limit(c) for c in chunks])
    
    def extract_sync(self, chunk: DocumentChunk) -> DocumentChunk:
        """Synchronous version of extract."""
        prompt = f"""Extract named entities and relationships from the following text.

Text:
{chunk.text}

Return in this exact format:
ENTITIES:
- entity1
- entity2
- entity3

RELATIONS:
- entity1 | relationship | entity2
- entity2 | relationship | entity3

Be concise. Only extract clear, explicit entities and relationships."""

        try:
            response = self.llm.generate(prompt)
            chunk.entities = self._parse_entities(response)
            chunk.relations = self._parse_relations(response)
        except Exception as e:
            logger.warning(f"Entity extraction failed for {chunk.chunk_id}: {e}")
            chunk.entities = []
            chunk.relations = []
        
        return chunk
    
    def _parse_entities(self, response: str) -> list[str]:
        """Parse entity names from LLM response."""
        entities = []
        
        # Look for ENTITIES section
        match = re.search(r'ENTITIES:\s*(.*?)(?:RELATIONS:|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            for line in section.strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    entity = line[2:].strip()
                    if entity:
                        entities.append(entity)
                elif line and not line.startswith('#'):
                    entities.append(line)
        
        return entities[:20]  # Limit to prevent noise
    
    def _parse_relations(self, response: str) -> list[tuple[str, str, str]]:
        """Parse relations from LLM response."""
        relations = []
        
        # Look for RELATIONS section
        match = re.search(r'RELATIONS:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            for line in section.strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    line = line[2:]
                
                # Parse "entity1 | relationship | entity2" format
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3 and all(parts):
                    relations.append((parts[0], parts[1], parts[2]))
        
        return relations[:20]  # Limit to prevent noise
