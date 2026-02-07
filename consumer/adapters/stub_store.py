"""Stub storage adapters for the consumer.

These implement the read-side of GraphStorePort, DocumentStorePort,
and VectorIndexPort with **no backing database**.  Every read method
returns an empty result.  Every write method raises
``NotImplementedError`` — the consumer never writes.

These stubs exist so the consumer can boot and wire an orchestrator
today.  They will be replaced by **network adapters** that proxy
read calls to the producer over MCP / HTTP once we add distribution.

Swap plan:
    StubGraphStore     →  RemoteGraphStore(producer_url)
    StubDocumentStore  →  RemoteDocumentStore(producer_url)
    StubVectorIndex    →  RemoteVectorIndex(producer_url)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from archrag.domain.models import (
    Community,
    CommunityHierarchy,
    Entity,
    Relation,
    TextChunk,
)
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.vector_index import VectorIndexPort


# ── Graph Store Stub ─────────────────────────────────────────────────


class StubGraphStore(GraphStorePort):
    """No-op graph store — reads return empty, writes raise.

    Will be replaced by ``RemoteGraphStore`` that calls the producer.
    """

    _NOT_CONNECTED = "StubGraphStore: not connected to producer (network adapter pending)"

    # ── write (blocked) ──

    def save_entity(self, entity: Entity) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_entities(self, entities: list[Entity]) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_relation(self, relation: Relation) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_relations(self, relations: list[Relation]) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def delete_entity(self, entity_id: str) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def clear(self) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    # ── read (empty) ──

    def get_entity(self, entity_id: str) -> Entity | None:
        return None

    def get_all_entities(self) -> list[Entity]:
        return []

    def get_entity_by_name(self, name: str) -> Entity | None:
        return None

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        return []

    def get_all_relations(self) -> list[Relation]:
        return []

    def get_neighbours(self, entity_id: str) -> list[Entity]:
        return []

    def search_entities_by_name(self, query: str) -> list[Entity]:
        return []

    # ── lifecycle ──

    def clone(self) -> "StubGraphStore":
        return StubGraphStore()


# ── Document Store Stub ──────────────────────────────────────────────


class StubDocumentStore(DocumentStorePort):
    """No-op document store — reads return empty, writes raise.

    Will be replaced by ``RemoteDocumentStore`` that calls the producer.
    """

    _NOT_CONNECTED = "StubDocumentStore: not connected to producer (network adapter pending)"

    # ── write (blocked) ──

    def save_chunk(self, chunk: TextChunk) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_chunks(self, chunks: list[TextChunk]) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_community(self, community: Community) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_communities(self, communities: list[Community]) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save_hierarchy(self, hierarchy: CommunityHierarchy) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def put_meta(self, key: str, value: Any) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def delete_chunk(self, chunk_id: str) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def clear(self) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    # ── read (empty) ──

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        return None

    def get_all_chunks(self) -> list[TextChunk]:
        return []

    def get_community(self, community_id: str) -> Community | None:
        return None

    def get_communities_at_level(self, level: int) -> list[Community]:
        return []

    def load_hierarchy(self) -> CommunityHierarchy | None:
        return None

    def get_meta(self, key: str) -> Any:
        return None

    def search_chunks(self, query: str) -> list[TextChunk]:
        return []

    # ── lifecycle ──

    def clone(self) -> "StubDocumentStore":
        return StubDocumentStore()


# ── Vector Index Stub ────────────────────────────────────────────────


class StubVectorIndex(VectorIndexPort):
    """No-op vector index — searches return empty, writes raise.

    Will be replaced by ``RemoteVectorIndex`` that calls the producer.
    """

    _NOT_CONNECTED = "StubVectorIndex: not connected to producer (network adapter pending)"

    # ── write (blocked) ──

    def add_vectors(
        self,
        ids: list[str],
        vectors: np.ndarray,
        *,
        layer: int = 0,
    ) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def save(self, path: str) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    def clear(self) -> None:
        raise NotImplementedError(self._NOT_CONNECTED)

    # ── read (empty) ──

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        layer: int = 0,
        candidate_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        return []

    def get_vector(self, id: str) -> np.ndarray | None:
        return None

    def load(self, path: str) -> None:
        pass  # Nothing to load — stub

    # ── lifecycle ──

    def clone(self) -> "StubVectorIndex":
        return StubVectorIndex()
