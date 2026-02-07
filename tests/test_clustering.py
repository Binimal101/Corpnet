"""Tests for clustering module."""

import pytest

from src.clustering.knowledge_graph import KnowledgeGraph
from src.clustering.hierarchy import CommunityHierarchy
from src.core.types import Community, DocumentChunk


class TestKnowledgeGraph:
    """Tests for knowledge graph."""
    
    def test_add_chunk(self, sample_chunk):
        """Test adding a chunk to the graph."""
        kg = KnowledgeGraph()
        kg.add_chunk(sample_chunk)
        
        # Should have nodes for entities
        for entity in sample_chunk.entities:
            assert entity in kg.nodes
    
    def test_add_edges(self):
        """Test adding edges via chunk relations."""
        kg = KnowledgeGraph()
        
        chunk = DocumentChunk(
            chunk_id="test",
            doc_id="doc",
            text="test",
            embedding=[0.1] * 768,
            labels=set(),
            metadata={},
            entities=["A", "B", "C"],
            relations=[("A", "relates_to", "B"), ("B", "contains", "C")],
        )
        
        kg.add_chunk(chunk)
        
        assert kg.node_count() == 3
        assert kg.edge_count() == 2
    
    def test_get_edges_for_node(self):
        """Test retrieving edges for a node."""
        kg = KnowledgeGraph()
        
        chunk = DocumentChunk(
            chunk_id="test",
            doc_id="doc",
            text="test",
            embedding=[0.1] * 768,
            labels=set(),
            metadata={},
            entities=["A", "B"],
            relations=[("A", "relates_to", "B")],
        )
        
        kg.add_chunk(chunk)
        
        edges = kg.get_edges_for("A")
        assert len(edges) == 1
        assert edges[0].source == "A"
        assert edges[0].target == "B"


class TestCommunityHierarchy:
    """Tests for community hierarchy."""
    
    def test_add_level(self, sample_community):
        """Test adding a level to hierarchy."""
        hierarchy = CommunityHierarchy()
        hierarchy.add_level([sample_community])
        
        assert hierarchy.num_levels == 1
        assert hierarchy.total_communities() == 1
    
    def test_get_community(self, sample_community):
        """Test retrieving a community by ID."""
        hierarchy = CommunityHierarchy()
        hierarchy.add_level([sample_community])
        
        retrieved = hierarchy.get_community(sample_community.community_id)
        assert retrieved is not None
        assert retrieved.summary == sample_community.summary
    
    def test_parent_child_relationships(self):
        """Test setting parent-child relationships."""
        hierarchy = CommunityHierarchy()
        
        child = Community(
            community_id="child",
            level=0,
            summary="Child",
            summary_embedding=[0.1] * 768,
        )
        parent = Community(
            community_id="parent",
            level=1,
            summary="Parent",
            summary_embedding=[0.1] * 768,
            children_ids=["child"],
        )
        
        hierarchy.add_level([child])
        hierarchy.add_level([parent])
        hierarchy.set_parent("child", "parent")
        
        assert hierarchy.get_parent("child").community_id == "parent"
        assert "child" in hierarchy.get_community("parent").children_ids
    
    def test_top_level(self):
        """Test top_level property."""
        hierarchy = CommunityHierarchy()
        
        c1 = Community(community_id="c1", level=0, summary="", summary_embedding=[])
        c2 = Community(community_id="c2", level=1, summary="", summary_embedding=[])
        
        hierarchy.add_level([c1])
        assert hierarchy.top_level == 0
        
        hierarchy.add_level([c2])
        assert hierarchy.top_level == 1
    
    def test_path_to_root(self):
        """Test path from node to root."""
        hierarchy = CommunityHierarchy()
        
        c1 = Community(community_id="c1", level=0, summary="", summary_embedding=[])
        c2 = Community(community_id="c2", level=1, summary="", summary_embedding=[])
        c3 = Community(community_id="c3", level=2, summary="", summary_embedding=[])
        
        hierarchy.add_level([c1])
        hierarchy.add_level([c2])
        hierarchy.add_level([c3])
        
        hierarchy.set_parent("c1", "c2")
        hierarchy.set_parent("c2", "c3")
        
        path = hierarchy.get_path_to_root("c1")
        assert path == ["c1", "c2", "c3"]
