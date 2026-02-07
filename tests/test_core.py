"""Tests for core module."""

import pytest

from src.core.types import DocumentChunk, Community, QueryRequest, QueryResponse, SearchResult
from src.core.config import get_settings, reset_settings, Settings
from src.core.embeddings import MockEmbedding
from src.core.llm import MockLLM


class TestTypes:
    """Tests for core types."""
    
    def test_document_chunk_serialization(self, sample_chunk):
        """Test DocumentChunk to_dict and from_dict."""
        d = sample_chunk.to_dict()
        restored = DocumentChunk.from_dict(d)
        
        assert restored.chunk_id == sample_chunk.chunk_id
        assert restored.text == sample_chunk.text
        assert restored.entities == sample_chunk.entities
        assert restored.relations == sample_chunk.relations
    
    def test_community_serialization(self, sample_community):
        """Test Community to_dict and from_dict."""
        d = sample_community.to_dict()
        restored = Community.from_dict(d)
        
        assert restored.community_id == sample_community.community_id
        assert restored.level == sample_community.level
        assert restored.summary == sample_community.summary
    
    def test_query_request_defaults(self):
        """Test QueryRequest default values."""
        req = QueryRequest(
            query_id="test",
            text="test query",
            embedding=[0.1] * 768,
        )
        
        assert req.similarity_threshold == 0.35
        assert req.top_k == 10
        assert req.ttl == 10
    
    def test_search_result_serialization(self):
        """Test SearchResult to_dict and from_dict."""
        result = SearchResult(
            chunk_id="chunk-1",
            text="Test result",
            score=0.95,
            source="test-doc",
            peer_id="peer-001",
            metadata={"key": "value"},
        )
        
        d = result.to_dict()
        restored = SearchResult.from_dict(d)
        
        assert restored.chunk_id == result.chunk_id
        assert restored.score == result.score
        assert restored.metadata == result.metadata


class TestConfig:
    """Tests for configuration loading."""
    
    def test_default_settings(self):
        """Test loading default settings."""
        reset_settings()
        settings = get_settings("config/default.yaml")
        
        assert settings.embeddings.dimension == 768
        assert settings.routing.similarity_threshold == 0.35
    
    def test_settings_singleton(self):
        """Test that settings are cached."""
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        
        assert s1 is s2


class TestEmbeddings:
    """Tests for embedding providers."""
    
    def test_mock_embedding_dimension(self, embedder):
        """Test mock embedder dimension."""
        assert embedder.dimension == 768
    
    def test_mock_embedding_deterministic(self, embedder):
        """Test mock embedder produces deterministic output."""
        text = "test text"
        emb1 = embedder.embed_text(text)
        emb2 = embedder.embed_text(text)
        
        assert emb1 == emb2
    
    def test_mock_embedding_normalized(self, embedder):
        """Test mock embedder produces normalized vectors."""
        import numpy as np
        
        emb = embedder.embed_text("test")
        norm = np.linalg.norm(emb)
        
        assert abs(norm - 1.0) < 0.01
    
    def test_batch_embedding(self, embedder):
        """Test batch embedding."""
        texts = ["text 1", "text 2", "text 3"]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 768


class TestLLM:
    """Tests for LLM providers."""
    
    def test_mock_llm_generate(self, llm):
        """Test mock LLM generation."""
        response = llm.generate("test prompt")
        assert "Mock response" in response
    
    def test_mock_llm_extract_topics(self, llm):
        """Test mock LLM topic extraction."""
        topics = llm.extract_topics("How does Python programming work?")
        assert isinstance(topics, list)
    
    def test_mock_llm_summarize(self, llm):
        """Test mock LLM summarization."""
        summary = llm.summarize_community(
            entities=["Python", "FastAPI"],
            relations=["FastAPI uses Python"],
        )
        assert isinstance(summary, str)
        assert len(summary) > 0
