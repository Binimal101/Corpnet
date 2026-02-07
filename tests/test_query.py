"""Tests for query module."""

import pytest

from src.query.reranker import Reranker, adaptive_filter
from src.query.generator import AnswerGenerator
from src.core.types import SearchResult


class TestReranker:
    """Tests for result reranking."""
    
    def test_merge_and_dedup(self):
        """Test merging and deduplicating results."""
        reranker = Reranker()
        
        results = [
            SearchResult(chunk_id="c1", text="Test 1", score=0.9, source="doc1", peer_id="p1"),
            SearchResult(chunk_id="c2", text="Test 2", score=0.8, source="doc1", peer_id="p1"),
            SearchResult(chunk_id="c1", text="Test 1", score=0.85, source="doc1", peer_id="p2"),  # Duplicate
        ]
        
        merged = reranker.merge_and_rerank(results, top_k=10)
        
        # Should have deduplicated c1
        ids = [r.chunk_id for r in merged]
        assert ids.count("c1") == 1
    
    def test_sort_by_score(self):
        """Test that results are sorted by score."""
        reranker = Reranker(relevance_threshold=0.0)
        
        results = [
            SearchResult(chunk_id="c1", text="Test 1", score=0.5, source="doc", peer_id="p1"),
            SearchResult(chunk_id="c2", text="Test 2", score=0.9, source="doc", peer_id="p1"),
            SearchResult(chunk_id="c3", text="Test 3", score=0.7, source="doc", peer_id="p1"),
        ]
        
        merged = reranker.merge_and_rerank(results, top_k=10)
        
        assert merged[0].score == 0.9
        assert merged[1].score == 0.7
        assert merged[2].score == 0.5
    
    def test_relevance_threshold(self):
        """Test relevance threshold filtering."""
        reranker = Reranker(relevance_threshold=0.6)
        
        results = [
            SearchResult(chunk_id="c1", text="Test 1", score=0.9, source="doc", peer_id="p1"),
            SearchResult(chunk_id="c2", text="Test 2", score=0.5, source="doc", peer_id="p1"),  # Below threshold
            SearchResult(chunk_id="c3", text="Test 3", score=0.7, source="doc", peer_id="p1"),
        ]
        
        merged = reranker.merge_and_rerank(results, top_k=10)
        
        assert len(merged) == 2
        assert all(r.score >= 0.6 for r in merged)
    
    def test_top_k_limit(self):
        """Test top-k limiting."""
        reranker = Reranker(relevance_threshold=0.0)
        
        results = [
            SearchResult(chunk_id=f"c{i}", text=f"Test {i}", score=0.9-i*0.1, source="doc", peer_id="p1")
            for i in range(10)
        ]
        
        merged = reranker.merge_and_rerank(results, top_k=3)
        
        assert len(merged) == 3


class TestAdaptiveFilter:
    """Tests for adaptive_filter convenience function."""
    
    def test_adaptive_filter(self):
        """Test adaptive filtering."""
        results = [
            SearchResult(chunk_id="c1", text="Test 1", score=0.9, source="doc", peer_id="p1"),
            SearchResult(chunk_id="c2", text="Test 2", score=0.4, source="doc", peer_id="p1"),
        ]
        
        filtered = adaptive_filter(results, threshold=0.5, top_k=10)
        
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "c1"


class TestAnswerGenerator:
    """Tests for answer generation."""
    
    def test_generate_with_results(self, llm):
        """Test generating answer with results."""
        generator = AnswerGenerator(llm)
        
        results = [
            SearchResult(chunk_id="c1", text="Python is a programming language.", score=0.9, source="doc1", peer_id="p1"),
            SearchResult(chunk_id="c2", text="FastAPI is built on Python.", score=0.8, source="doc2", peer_id="p1"),
        ]
        
        answer = generator.generate_sync("What is Python?", results)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_generate_without_results(self, llm):
        """Test generating answer without results."""
        generator = AnswerGenerator(llm)
        
        answer = generator.generate_sync("What is Python?", [])
        
        assert "couldn't find" in answer.lower() or "no" in answer.lower()
