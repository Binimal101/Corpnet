"""Tests for domain models."""

from archrag.domain.models import (
    AnalysisPoint,
    AnalysisReport,
    CHNSWIndex,
    CHNSWNode,
    Community,
    CommunityHierarchy,
    Entity,
    KnowledgeGraph,
    Relation,
    SearchResult,
    TextChunk,
)


class TestTextChunk:
    def test_creation(self):
        c = TextChunk(content="hello world", category="general")
        assert c.content == "hello world"
        assert c.category == "general"
        assert len(c.id) == 12
        assert c.keywords == []
        assert c.tags == []
        assert c.retrieval_count == 0
        assert c.embedding is None

    def test_full_fields(self):
        c = TextChunk(
            content="x",
            last_updated="202602071200",
            keywords=["ai", "ml"],
            tags=["research"],
            category="science",
            retrieval_count=5,
            embedding_model="text-embedding-3-small",
            embedding=[0.1, 0.2],
        )
        assert c.last_updated == "202602071200"
        assert c.keywords == ["ai", "ml"]
        assert c.tags == ["research"]
        assert c.retrieval_count == 5
        assert c.embedding_model == "text-embedding-3-small"
        assert c.embedding == [0.1, 0.2]


class TestEntity:
    def test_creation(self):
        e = Entity(name="Alice", description="A researcher")
        assert e.name == "Alice"
        assert e.embedding is None

    def test_with_embedding(self):
        e = Entity(name="Bob", description="Prof", embedding=[1.0, 2.0])
        assert e.embedding == [1.0, 2.0]


class TestKnowledgeGraph:
    def test_add_and_query(self, sample_kg):
        assert len(sample_kg.entities) == 5
        assert len(sample_kg.relations) == 4

    def test_neighbours(self, sample_kg):
        nbrs = sample_kg.neighbours("e0")
        assert set(nbrs) == {"e1", "e2", "e3"}

    def test_entity_by_name(self, sample_kg):
        e = sample_kg.entity_by_name("alice")
        assert e is not None
        assert e.id == "e0"

    def test_entity_by_name_not_found(self, sample_kg):
        assert sample_kg.entity_by_name("nobody") is None


class TestCommunityHierarchy:
    def test_empty(self):
        h = CommunityHierarchy()
        assert h.height == 0
        assert h.all_communities() == []

    def test_with_levels(self):
        c1 = Community(id="c1", level=0, member_ids=["e1", "e2"])
        c2 = Community(id="c2", level=0, member_ids=["e3"])
        c3 = Community(id="c3", level=1, member_ids=["c1", "c2"])
        h = CommunityHierarchy(levels=[[c1, c2], [c3]])
        assert h.height == 2
        assert len(h.all_communities()) == 3
        assert h.communities_at(0) == [c1, c2]
        assert h.communities_at(1) == [c3]
        assert h.communities_at(99) == []


class TestCHNSWIndex:
    def test_basics(self):
        idx = CHNSWIndex(M=16)
        n = CHNSWNode(id="n1", level=0, embedding=[1.0, 0.0])
        idx.nodes["n1"] = n
        idx.layers.append(["n1"])
        assert idx.height == 1
        assert len(idx.nodes_at(0)) == 1


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(node_id="x", level=0, distance=0.1, text="hi")
        assert r.distance == 0.1


class TestAnalysisReport:
    def test_creation(self):
        p = AnalysisPoint(description="point", score=80.0)
        r = AnalysisReport(level=0, points=[p])
        assert r.points[0].score == 80.0
