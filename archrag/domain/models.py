"""Pure domain models — zero external dependencies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


# ── Corpus ──────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A segment of a source document."""

    text: str
    id: str = field(default_factory=_uid)
    source_doc: str = ""
    access_id: str = ""  # Hierarchical access scope (inherits from parent note)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Knowledge Graph ─────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A node in the knowledge graph."""

    name: str
    description: str
    id: str = field(default_factory=_uid)
    entity_type: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    description: str
    id: str = field(default_factory=_uid)
    weight: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """Container for entities and relations."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    # ── helpers ──

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def neighbours(self, entity_id: str) -> list[str]:
        """Return IDs of entities connected to *entity_id*."""
        ids: list[str] = []
        for r in self.relations:
            if r.source_id == entity_id:
                ids.append(r.target_id)
            elif r.target_id == entity_id:
                ids.append(r.source_id)
        return ids

    def entity_by_name(self, name: str) -> Entity | None:
        for e in self.entities.values():
            if e.name.lower() == name.lower():
                return e
        return None


# ── Attributed Communities ──────────────────────────────────────────────────

@dataclass
class Community:
    """An attributed community at a single hierarchical level."""

    id: str = field(default_factory=_uid)
    level: int = 0
    member_ids: list[str] = field(default_factory=list)  # entity or child-community ids
    summary: str = ""
    embedding: list[float] | None = None
    parent_id: str | None = None


@dataclass
class CommunityHierarchy:
    """The full multi-level hierarchy Δ of attributed communities."""

    levels: list[list[Community]] = field(default_factory=list)  # index = level

    @property
    def height(self) -> int:
        return len(self.levels)

    def communities_at(self, level: int) -> list[Community]:
        if 0 <= level < len(self.levels):
            return self.levels[level]
        return []

    def all_communities(self) -> list[Community]:
        return [c for level in self.levels for c in level]


# ── C-HNSW Index ────────────────────────────────────────────────────────────

@dataclass
class CHNSWNode:
    """A node inside the C-HNSW index."""

    id: str
    level: int
    embedding: list[float]
    label: str = ""  # entity name or community summary snippet
    intra_neighbours: list[str] = field(default_factory=list)  # same-layer links
    inter_link_down: str | None = None  # nearest neighbour in layer below


@dataclass
class CHNSWIndex:
    """Community-based Hierarchical Navigable Small World index."""

    nodes: dict[str, CHNSWNode] = field(default_factory=dict)  # id → node
    layers: list[list[str]] = field(default_factory=list)  # layer i → list of node ids
    M: int = 32  # max intra-layer connections per node
    ef_construction: int = 100

    @property
    def height(self) -> int:
        return len(self.layers)

    def nodes_at(self, level: int) -> list[CHNSWNode]:
        if 0 <= level < len(self.layers):
            return [self.nodes[nid] for nid in self.layers[level]]
        return []


# ── Search / Query results ──────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single retrieved element from the C-HNSW."""

    node_id: str
    level: int
    distance: float
    text: str = ""


@dataclass
class AnalysisPoint:
    """One point from the LLM's adaptive filtering report."""

    description: str
    score: float


@dataclass
class AnalysisReport:
    """Result of LLM adaptive filtering for one layer."""

    level: int
    points: list[AnalysisPoint] = field(default_factory=list)


# ── Memory Notes (A-Mem inspired) ────────────────────────────────────────────

@dataclass
class MemoryNote:
    """A memory note following the A-Mem Zettelkasten-inspired design.

    Implements the memory model from arXiv 2502.12110 (Section 3.1):
    - content (c_i): Original interaction content
    - keywords (K_i): LLM-generated key concepts
    - tags (G_i): LLM-generated categorization tags
    - embedding (e_i): Dense vector for similarity matching
    - embedding_model: Model used to generate the embedding
    - access_id: Hierarchical access scope for permission filtering
    """

    content: str
    id: str = field(default_factory=_uid)
    last_updated: str | None = None
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    category: str = ""
    retrieval_count: int = 0
    embedding: list[float] | None = None
    embedding_model: str = ""  # Model identifier (e.g., "text-embedding-3-small")
    access_id: str = ""  # Hierarchical access scope (e.g., "dept_engineering")

    def to_document(self) -> dict[str, Any]:
        """Convert to a document dict for KG construction pipeline."""
        return {
            "text": self.content,
            "title": f"Note {self.id}",
            "id": self.id,
            "access_id": self.access_id,
            "metadata": {
                "keywords": self.keywords,
                "tags": self.tags,
                "category": self.category,
                "access_id": self.access_id,
            },
        }

    def increment_retrieval(self) -> None:
        """Update access stats when retrieved."""
        self.retrieval_count += 1
        self.last_updated = _now_timestamp()


# ── External Database Connector Models ──────────────────────────────────────

@dataclass
class ColumnInfo:
    """Column/field metadata for a database table."""

    name: str
    data_type: str
    nullable: bool = True
    is_text: bool = False  # Whether to include in text extraction


@dataclass
class RelationshipInfo:
    """Foreign key / reference relationship between tables."""

    from_column: str
    to_table: str
    to_column: str
    relationship_type: str = "foreign_key"


@dataclass
class TableSchema:
    """Schema information for a database table/collection."""

    name: str
    database: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: str | None = None
    relationships: list[RelationshipInfo] = field(default_factory=list)

    def text_columns(self) -> list[str]:
        """Return names of columns marked for text extraction."""
        return [c.name for c in self.columns if c.is_text]


@dataclass
class ExternalRecord:
    """A record extracted from an external database."""

    id: str  # Primary key from source
    source_table: str  # Table/collection name
    source_database: str  # Database identifier
    content: dict[str, Any]  # All fields as key-value
    text_content: str  # Concatenated text for indexing
    access_id: str = ""  # Access scope for permission filtering
    metadata: dict[str, Any] = field(default_factory=dict)  # Schema info, types, constraints
    created_at: str | None = None  # Source timestamp if available
    updated_at: str | None = None  # Source timestamp if available

    def to_note_input(self) -> dict[str, Any]:
        """Convert to input format for NoteConstructionService."""
        return {
            "content": self.text_content,
            "category": self.source_table,
            "tags": [self.source_database, self.source_table],
            "access_id": self.access_id,
            "metadata": {
                "source_id": self.id,
                "source_table": self.source_table,
                "source_database": self.source_database,
                "original_content": self.content,
                "access_id": self.access_id,
            },
        }


@dataclass
class SyncState:
    """Tracks sync progress for incremental updates."""

    connector_id: str
    database_name: str
    table_name: str
    last_sync_at: str
    last_record_id: str | None = None
    last_updated_at: str | None = None
    record_count: int = 0


@dataclass
class SyncResult:
    """Result of a database sync operation."""

    tables_synced: list[str] = field(default_factory=list)
    records_added: int = 0
    records_updated: int = 0
    records_failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
