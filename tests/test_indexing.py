"""Tests for indexing module."""

import pytest
import uuid

from src.indexing.vector_store import InMemoryVectorStore
from src.core.types import DocumentChunk


class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""
    
    def test_insert_and_count(self, vector_store, sample_chunk):
        """Test inserting a chunk."""
        vector_store.insert(sample_chunk)
        assert vector_store.count() == 1
    
    def test_bulk_insert(self, vector_store, sample_chunks):
        """Test bulk insertion."""
        ids = vector_store.bulk_insert(sample_chunks)
        
        assert len(ids) == len(sample_chunks)
        assert vector_store.count() == len(sample_chunks)
    
    def test_search(self, populated_store, sample_embedding):
        """Test similarity search."""
        results = populated_store.search(sample_embedding, top_k=3)
        
        assert len(results) <= 3
        assert all(r.score <= 1.0 for r in results)
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_search_by_threshold(self, populated_store, sample_embedding):
        """Test threshold-based search."""
        results = populated_store.search_by_threshold(
            embedding=sample_embedding,
            threshold=0.5,
        )
        
        # Should return at least 1 result (guaranteed)
        assert len(results) >= 1
    
    def test_delete(self, vector_store, sample_chunk):
        """Test chunk deletion."""
        vector_store.insert(sample_chunk)
        assert vector_store.count() == 1
        
        vector_store.delete(sample_chunk.chunk_id)
        assert vector_store.count() == 0
    
    def test_search_empty_store(self, vector_store, sample_embedding):
        """Test searching empty store."""
        results = vector_store.search(sample_embedding, top_k=10)
        assert len(results) == 0
    
    def test_insert_without_embedding_fails(self, vector_store):
        """Test that inserting without embedding fails."""
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id="test",
            text="test",
            embedding=None,  # No embedding!
            labels=set(),
            metadata={},
            entities=[],
            relations=[],
        )
        
        with pytest.raises(ValueError):
            vector_store.insert(chunk)
