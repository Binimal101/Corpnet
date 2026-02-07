"""Text chunking for document ingestion.

Implements recursive character-based chunking with overlap.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from src.core.types import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] | None = None


class RecursiveChunker:
    """Recursive character-based text chunker.
    
    Splits text on decreasingly granular separators until chunks
    are within the target size.
    """
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[DocumentChunk]:
        """Split text into chunks.
        
        Args:
            text: The text to chunk.
            doc_id: Parent document ID.
            metadata: Optional metadata to attach to each chunk.
        
        Returns:
            List of DocumentChunk objects (without embeddings).
        """
        if not text.strip():
            return []
        
        chunks = self._recursive_split(text, self.separators)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            
            chunk_metadata = dict(metadata or {})
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            result.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text.strip(),
                embedding=None,
                labels=set(),
                metadata=chunk_metadata,
                entities=[],
                relations=[],
            ))
        
        logger.info(f"Chunked document {doc_id} into {len(result)} chunks")
        return result
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if not separators:
            # No more separators, just split by character count
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            return self._split_by_size(text)
        
        parts = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            candidate = current_chunk + (separator if current_chunk else "") + part
            
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) > self.chunk_size:
                    # Part is too big, recursively split with next separator
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Apply overlap
        return self._apply_overlap(chunks)
    
    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if self.chunk_overlap < (end - start) else end
        return chunks
    
    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between adjacent chunks."""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Add overlap from the end of the previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            overlapped = overlap_text + " " + curr_chunk
            
            if len(overlapped) <= self.chunk_size * 1.5:  # Allow some flexibility
                result.append(overlapped)
            else:
                result.append(curr_chunk)
        
        return result


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """Convenience function for chunking text."""
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk(text, doc_id, metadata)
