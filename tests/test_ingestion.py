"""Tests for ingestion module."""

import pytest

from src.ingestion.chunker import RecursiveChunker, chunk_text
from src.ingestion.entity_extractor import EntityExtractor
from src.core.types import DocumentChunk


class TestChunker:
    """Tests for text chunking."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        text = "This is a test. " * 100  # ~1600 chars
        chunks = chunk_text(text, doc_id="test", chunk_size=256)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 256 * 1.5  # Allow some flexibility
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", doc_id="test")
        assert len(chunks) == 0
    
    def test_small_text(self):
        """Test chunking text smaller than chunk size."""
        text = "Small text."
        chunks = chunk_text(text, doc_id="test", chunk_size=256)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_chunk_metadata(self):
        """Test that chunks have correct metadata."""
        text = "Test text. " * 50
        chunks = chunk_text(
            text,
            doc_id="test-doc",
            metadata={"source": "test"},
        )
        
        assert len(chunks) > 0
        assert chunks[0].doc_id == "test-doc"
        assert chunks[0].metadata.get("source") == "test"
        assert "chunk_index" in chunks[0].metadata
    
    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique."""
        text = "Test text. " * 100
        chunks = chunk_text(text, doc_id="test", chunk_size=100)
        
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestEntityExtractor:
    """Tests for entity extraction."""
    
    def test_extraction_populates_chunk(self, llm):
        """Test that extraction populates chunk fields."""
        extractor = EntityExtractor(llm)
        
        chunk = DocumentChunk(
            chunk_id="test",
            doc_id="test-doc",
            text="Python is a programming language. FastAPI is built on Python.",
            embedding=None,
            labels=set(),
            metadata={},
            entities=[],
            relations=[],
        )
        
        result = extractor.extract_sync(chunk)
        
        # Mock LLM won't extract real entities, but fields should be populated
        assert result is chunk
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)
