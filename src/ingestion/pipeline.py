"""Ingestion pipeline orchestrator.

Supports two paths:
1. Pre-embedded path: Records arrive with embeddings (primary for peer uploads)
2. Raw text path: Documents need chunking, extraction, and embedding
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from src.core.types import DocumentChunk, IngestRecord

if TYPE_CHECKING:
    from src.core.embeddings import EmbeddingProvider
    from src.core.llm import LLMProvider
    from src.ingestion.chunker import RecursiveChunker
    from src.ingestion.entity_extractor import EntityExtractor
    from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


def serialize_metadata(metadata: dict[str, Any]) -> str:
    """Convert metadata dict to flat text representation.
    
    {"name": "John", "role": "engineer"} → "name: John | role: engineer"
    """
    return " | ".join(f"{k}: {v}" for k, v in metadata.items())


@dataclass
class IngestionPipeline:
    """Orchestrates document ingestion.
    
    Two paths:
    - Pre-embedded: for peer uploads with pre-computed embeddings
    - Raw text: for file ingestion requiring full processing
    """
    
    embedder: "EmbeddingProvider"
    llm: "LLMProvider"
    vector_store: "VectorStore | None" = None
    chunker: "RecursiveChunker | None" = None
    entity_extractor: "EntityExtractor | None" = None
    
    async def ingest_records(self, records: list[IngestRecord], source: str) -> list[str]:
        """Pre-embedded path: records arrive with embeddings.
        
        No chunking or embedding computation needed.
        """
        logger.info(f"Ingesting {len(records)} pre-embedded records from {source}")
        
        chunk_ids = []
        for record in records:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=source,
                text=serialize_metadata(record.metadata),
                embedding=record.embedding,
                labels=set(),
                metadata=record.metadata,
                entities=[],
                relations=[],
            )
            
            if self.vector_store:
                self.vector_store.insert(chunk)
            
            chunk_ids.append(chunk.chunk_id)
        
        logger.info(f"Ingested {len(chunk_ids)} chunks from {source}")
        return chunk_ids
    
    async def ingest_text(
        self,
        text: str,
        doc_id: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        extract_entities: bool = True,
    ) -> list[str]:
        """Raw text path: full processing pipeline.
        
        Steps:
        1. Chunk text
        2. Extract entities/relations (optional)
        3. Embed chunks
        4. Store in vector store
        """
        from src.ingestion.chunker import RecursiveChunker
        from src.ingestion.entity_extractor import EntityExtractor
        
        logger.info(f"Ingesting document {doc_id} from {source}")
        
        # Step 1: Chunk
        chunker = self.chunker or RecursiveChunker()
        chunks = chunker.chunk(text, doc_id, metadata)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Extract entities (optional)
        if extract_entities:
            extractor = self.entity_extractor or EntityExtractor(self.llm)
            for chunk in chunks:
                extractor.extract_sync(chunk)
            logger.info("Extracted entities and relations")
        
        # Step 3: Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        logger.info("Computed embeddings")
        
        # Step 4: Store
        chunk_ids = []
        for chunk in chunks:
            if self.vector_store:
                self.vector_store.insert(chunk)
            chunk_ids.append(chunk.chunk_id)
        
        logger.info(f"Stored {len(chunk_ids)} chunks")
        return chunk_ids
    
    def ingest_text_sync(
        self,
        text: str,
        doc_id: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        extract_entities: bool = True,
    ) -> list[str]:
        """Synchronous version of ingest_text."""
        import asyncio
        return asyncio.run(self.ingest_text(text, doc_id, source, metadata, extract_entities))
