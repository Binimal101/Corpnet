"""Tests for the consumer's stub storage adapters.

Verifies that:
  - All read methods return empty results (not errors)
  - All write methods raise NotImplementedError
  - Stubs implement the full port interface
"""

from __future__ import annotations

import numpy as np
import pytest

from consumer.adapters.stub_store import (
    StubDocumentStore,
    StubGraphStore,
    StubVectorIndex,
)
from archrag.domain.models import (
    Community,
    CommunityHierarchy,
    Entity,
    Relation,
    TextChunk,
)


# ── StubGraphStore ──────────────────────────────────────────────────


class TestStubGraphStore:
    def setup_method(self):
        self.store = StubGraphStore()

    # reads → empty
    def test_get_entity_returns_none(self):
        assert self.store.get_entity("x") is None

    def test_get_all_entities_empty(self):
        assert self.store.get_all_entities() == []

    def test_get_entity_by_name_returns_none(self):
        assert self.store.get_entity_by_name("Alice") is None

    def test_get_relations_for_empty(self):
        assert self.store.get_relations_for("x") == []

    def test_get_all_relations_empty(self):
        assert self.store.get_all_relations() == []

    def test_get_neighbours_empty(self):
        assert self.store.get_neighbours("x") == []

    def test_search_entities_by_name_empty(self):
        assert self.store.search_entities_by_name("alice") == []

    def test_clone_returns_new_stub(self):
        clone = self.store.clone()
        assert isinstance(clone, StubGraphStore)
        assert clone is not self.store

    # writes → NotImplementedError
    def test_save_entity_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_entity(Entity(name="A", description="B"))

    def test_save_entities_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_entities([])

    def test_save_relation_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_relation(Relation(source_id="a", target_id="b", description="c"))

    def test_save_relations_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_relations([])

    def test_delete_entity_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.delete_entity("x")

    def test_clear_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.clear()


# ── StubDocumentStore ───────────────────────────────────────────────


class TestStubDocumentStore:
    def setup_method(self):
        self.store = StubDocumentStore()

    # reads → empty
    def test_get_chunk_returns_none(self):
        assert self.store.get_chunk("x") is None

    def test_get_all_chunks_empty(self):
        assert self.store.get_all_chunks() == []

    def test_get_community_returns_none(self):
        assert self.store.get_community("x") is None

    def test_get_communities_at_level_empty(self):
        assert self.store.get_communities_at_level(0) == []

    def test_load_hierarchy_returns_none(self):
        assert self.store.load_hierarchy() is None

    def test_get_meta_returns_none(self):
        assert self.store.get_meta("anything") is None

    def test_search_chunks_empty(self):
        assert self.store.search_chunks("test") == []

    def test_clone_returns_new_stub(self):
        clone = self.store.clone()
        assert isinstance(clone, StubDocumentStore)
        assert clone is not self.store

    # writes → NotImplementedError
    def test_save_chunk_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_chunk(TextChunk(content="hello"))

    def test_save_chunks_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_chunks([])

    def test_save_community_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_community(Community())

    def test_save_communities_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_communities([])

    def test_save_hierarchy_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save_hierarchy(CommunityHierarchy())

    def test_put_meta_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.put_meta("k", "v")

    def test_delete_chunk_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.delete_chunk("x")

    def test_clear_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.clear()


# ── StubVectorIndex ─────────────────────────────────────────────────


class TestStubVectorIndex:
    def setup_method(self):
        self.store = StubVectorIndex()

    # reads → empty
    def test_search_empty(self):
        q = np.zeros(8, dtype=np.float32)
        assert self.store.search(q, 5) == []

    def test_get_vector_returns_none(self):
        assert self.store.get_vector("x") is None

    def test_load_is_noop(self):
        self.store.load("anything.json")  # should not raise

    def test_clone_returns_new_stub(self):
        clone = self.store.clone()
        assert isinstance(clone, StubVectorIndex)
        assert clone is not self.store

    # writes → NotImplementedError
    def test_add_vectors_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.add_vectors(["a"], np.zeros((1, 8)))

    def test_save_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.save("out.json")

    def test_clear_raises(self):
        with pytest.raises(NotImplementedError):
            self.store.clear()
