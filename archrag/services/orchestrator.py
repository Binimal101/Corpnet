"""Orchestrator: wires offline indexing and online retrieval pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from archrag.domain.models import CHNSWIndex, CommunityHierarchy, KnowledgeGraph, MemoryNote, SyncResult
from archrag.ports.clustering import ClusteringPort
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.llm import LLMPort
from archrag.ports.memory_note_store import MemoryNoteStorePort
from archrag.ports.vector_index import VectorIndexPort
from archrag.ports.external_database import ExternalDatabaseConnectorPort
from archrag.services.adaptive_filtering import AdaptiveFilteringService
from archrag.services.chnsw_build import CHNSWBuildService
from archrag.services.hierarchical_clustering import HierarchicalClusteringService
from archrag.services.hierarchical_search import HierarchicalSearchService
from archrag.services.kg_construction import KGConstructionService
from archrag.services.note_construction import NoteConstructionService

if TYPE_CHECKING:
    from archrag.services.database_sync import DatabaseSyncService
    from archrag.services.auto_sync_worker import AutoSyncWorker
    from archrag.services.sync_queue import SyncQueue

log = logging.getLogger(__name__)


class ArchRAGOrchestrator:
    """Top-level entry point for the ArchRAG pipeline."""

    def __init__(
        self,
        llm: LLMPort,
        embedding: EmbeddingPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        vector_index: VectorIndexPort,
        clustering: ClusteringPort,
        memory_note_store: MemoryNoteStorePort | None = None,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        max_levels: int = 5,
        similarity_threshold: float = 0.7,
        M: int = 32,
        ef_construction: int = 100,
        k_per_layer: int = 5,
        note_k_nearest: int = 10,
        note_enable_evolution: bool = True,
    ):
        self._llm = llm
        self._embedding = embedding
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._vector_index = vector_index
        self._clustering = clustering
        self._memory_note_store = memory_note_store

        # Sub-services
        self._kg_service = KGConstructionService(
            llm, embedding, graph_store, doc_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._cluster_service = HierarchicalClusteringService(
            llm, embedding, clustering, graph_store, doc_store,
            max_levels=max_levels,
            similarity_threshold=similarity_threshold,
        )
        self._chnsw_service = CHNSWBuildService(
            embedding, vector_index, graph_store, doc_store,
            M=M,
            ef_construction=ef_construction,
        )
        self._search_service = HierarchicalSearchService(
            embedding, vector_index, graph_store, doc_store,
            k_per_layer=k_per_layer,
        )
        self._filter_service = AdaptiveFilteringService(llm)

        # Note construction service (if memory note store is provided)
        self._note_service: NoteConstructionService | None = None
        if memory_note_store is not None:
            self._note_service = NoteConstructionService(
                llm, embedding, memory_note_store,
                k_nearest=note_k_nearest,
                enable_evolution=note_enable_evolution,
            )

        self._index: CHNSWIndex | None = None
        
        # External database connector (optional, set via connect_database)
        self._db_connector: ExternalDatabaseConnectorPort | None = None
        self._db_sync_service: "DatabaseSyncService | None" = None
        self._auto_sync_worker: "AutoSyncWorker | None" = None
        self._sync_queue: "SyncQueue | None" = None

    # ── External Database Sync ──

    def connect_database(
        self,
        connector_type: str,
        connection_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Connect to an external database for syncing.

        Args:
            connector_type: "sql" or "nosql"
            connection_config: Database-specific connection parameters.
                SQL: {"connection_string": "postgresql://..."}
                NoSQL: {"host": "localhost", "port": 27017, "database": "mydb"}

        Returns:
            Connection info and discovered tables.
        """
        if connector_type == "sql":
            from archrag.adapters.connectors.sql_connector import GenericSQLConnector
            self._db_connector = GenericSQLConnector()
        elif connector_type == "nosql":
            from archrag.adapters.connectors.nosql_connector import GenericNoSQLConnector
            self._db_connector = GenericNoSQLConnector()
        else:
            raise ValueError(f"Unknown connector_type: {connector_type}")

        self._db_connector.connect(connection_config)
        
        # Initialize sync service
        if self._note_service is not None and self._memory_note_store is not None:
            from archrag.services.database_sync import DatabaseSyncService
            self._db_sync_service = DatabaseSyncService(
                connector=self._db_connector,
                note_service=self._note_service,
                doc_store=self._doc_store,
                note_store=self._memory_note_store,
            )

        tables = self._db_connector.list_tables()
        info = self._db_connector.get_connection_info()

        log.info("Connected to %s database with %d tables", connector_type, len(tables))

        return {
            "connected": True,
            "connector_type": connector_type,
            "connection_info": info,
            "tables": tables,
        }

    def disconnect_database(self) -> bool:
        """Disconnect from the external database."""
        if self._db_connector is not None:
            self._db_connector.disconnect()
            self._db_connector = None
            self._db_sync_service = None
            log.info("Disconnected from external database")
            return True
        return False

    def get_database_schema(self, table: str | None = None) -> dict[str, Any]:
        """Get schema information for the connected database.

        Args:
            table: Specific table to get schema for. If None, returns all tables.

        Returns:
            Schema information including tables, columns, and relationships.
        """
        if self._db_connector is None or not self._db_connector.is_connected():
            raise RuntimeError("Not connected to any database")

        if table:
            schema = self._db_connector.get_table_schema(table)
            return {
                "table": schema.name,
                "database": schema.database,
                "primary_key": schema.primary_key,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.data_type,
                        "nullable": c.nullable,
                        "is_text": c.is_text,
                    }
                    for c in schema.columns
                ],
                "relationships": [
                    {
                        "from_column": r.from_column,
                        "to_table": r.to_table,
                        "to_column": r.to_column,
                    }
                    for r in schema.relationships
                ],
            }

        # Return overview of all tables
        tables = self._db_connector.list_tables()
        relationships = self._db_connector.discover_relationships()

        return {
            "database": self._db_connector.get_connection_info().get("database", ""),
            "tables": tables,
            "relationships": [
                {
                    "from_column": r.from_column,
                    "to_table": r.to_table,
                    "to_column": r.to_column,
                    "type": r.relationship_type,
                }
                for r in relationships
            ],
        }

    def sync_from_database(
        self,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
        *,
        incremental: bool = True,
        enable_linking: bool = True,
        enable_evolution: bool = True,
    ) -> dict[str, Any]:
        """Sync records from the connected external database.

        Args:
            tables: Tables to sync. If None, syncs all tables.
            text_columns_map: Map of table -> text columns for indexing.
            incremental: If True, only sync new/changed records.
            enable_linking: Whether to create links between notes.
            enable_evolution: Whether to evolve existing notes.

        Returns:
            Sync result with statistics.
        """
        if self._db_sync_service is None:
            raise RuntimeError("Database sync not available. Connect first and ensure note store is configured.")

        if incremental:
            result = self._db_sync_service.incremental_sync(
                tables=tables,
                text_columns_map=text_columns_map,
                enable_linking=enable_linking,
                enable_evolution=enable_evolution,
            )
        else:
            result = self._db_sync_service.full_sync(
                tables=tables,
                text_columns_map=text_columns_map,
                enable_linking=enable_linking,
                enable_evolution=enable_evolution,
            )

        return {
            "tables_synced": result.tables_synced,
            "records_added": result.records_added,
            "records_updated": result.records_updated,
            "records_failed": result.records_failed,
            "duration_seconds": result.duration_seconds,
            "errors": result.errors,
        }

    def get_sync_status(self) -> dict[str, Any]:
        """Get the current sync status for all tables."""
        if self._db_sync_service is None:
            return {"connected": False, "tables": {}}

        states = self._db_sync_service.get_all_sync_states()
        return {
            "connected": self._db_connector.is_connected() if self._db_connector else False,
            "tables": {
                table: {
                    "last_sync_at": state.last_sync_at,
                    "record_count": state.record_count,
                    "last_record_id": state.last_record_id,
                }
                for table, state in states.items()
            },
        }

    # ── Auto-Sync (Background Polling) ──

    def enable_auto_sync(
        self,
        poll_interval: float = 300.0,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
        enable_linking: bool = True,
        enable_evolution: bool = False,
    ) -> dict[str, Any]:
        """Enable automatic background syncing of database.

        The system will poll the connected database at regular intervals
        and automatically sync new records. This is connector-agnostic:
        works with SQL, NoSQL, or any future adapter.

        Args:
            poll_interval: Seconds between polls (default: 300 = 5 minutes).
            tables: Specific tables to monitor. None = all tables.
            text_columns_map: Map of table -> text columns for extraction.
            enable_linking: Whether to create links between notes.
            enable_evolution: Whether to evolve existing notes.

        Returns:
            Current auto-sync configuration.
        """
        if self._db_sync_service is None:
            raise RuntimeError("No database connected. Call connect_database() first.")

        # Create worker if not exists
        if self._auto_sync_worker is None:
            from archrag.services.auto_sync_worker import AutoSyncWorker
            self._auto_sync_worker = AutoSyncWorker(self._db_sync_service)

        # Configure and enable
        self._auto_sync_worker.configure(
            poll_interval=poll_interval,
            tables=tables,
            text_columns_map=text_columns_map,
            enable_linking=enable_linking,
            enable_evolution=enable_evolution,
        )
        self._auto_sync_worker.enable()

        return self._auto_sync_worker.get_config()

    def disable_auto_sync(self) -> bool:
        """Disable automatic background syncing.

        Returns:
            True if auto-sync was disabled, False if it wasn't enabled.
        """
        if self._auto_sync_worker is None:
            return False

        self._auto_sync_worker.disable()
        return True

    def configure_auto_sync(
        self,
        poll_interval: float | None = None,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
        enable_linking: bool | None = None,
        enable_evolution: bool | None = None,
    ) -> dict[str, Any]:
        """Update auto-sync configuration without enabling/disabling.

        Args:
            poll_interval: Seconds between polls.
            tables: Specific tables to monitor.
            text_columns_map: Map of table -> text columns.
            enable_linking: Whether to create links between notes.
            enable_evolution: Whether to evolve existing notes.

        Returns:
            Updated configuration.
        """
        if self._auto_sync_worker is None:
            raise RuntimeError("Auto-sync not initialized. Call enable_auto_sync() first.")

        self._auto_sync_worker.configure(
            poll_interval=poll_interval,
            tables=tables,
            text_columns_map=text_columns_map,
            enable_linking=enable_linking,
            enable_evolution=enable_evolution,
        )
        return self._auto_sync_worker.get_config()

    def get_auto_sync_status(self) -> dict[str, Any]:
        """Get auto-sync status including config and statistics.

        Returns:
            Dictionary with enabled status, config, and stats.
        """
        if self._auto_sync_worker is None:
            return {
                "enabled": False,
                "config": None,
                "stats": None,
            }

        return {
            "enabled": self._auto_sync_worker.is_enabled(),
            "syncing": self._auto_sync_worker.is_syncing(),
            "config": self._auto_sync_worker.get_config(),
            "stats": self._auto_sync_worker.get_stats(),
        }

    def trigger_sync_now(self) -> dict[str, Any]:
        """Manually trigger an immediate sync.

        Bypasses the poll interval and syncs now.

        Returns:
            Sync result dictionary.
        """
        if self._auto_sync_worker is not None:
            return self._auto_sync_worker.trigger_sync_now()
        elif self._db_sync_service is not None:
            # Fall back to direct sync
            result = self._db_sync_service.incremental_sync()
            return {
                "tables_synced": result.tables_synced,
                "records_added": result.records_added,
                "records_failed": result.records_failed,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
            }
        else:
            raise RuntimeError("No database connected.")

    # ── Debounced Sync Queue ──

    def request_sync(self, tables: list[str] | None = None) -> dict[str, Any]:
        """Request a debounced sync for tables.

        Multiple requests within the debounce window are batched together.
        This prevents redundant syncs during high-volume periods.

        Args:
            tables: Tables to sync. None = all tables.

        Returns:
            Queue status.
        """
        if self._db_sync_service is None:
            raise RuntimeError("No database connected.")

        # Create queue if not exists
        if self._sync_queue is None:
            from archrag.services.sync_queue import SyncQueue
            self._sync_queue = SyncQueue(
                sync_fn=self.sync_from_database,
                debounce_window=30.0,
                min_interval=60.0,
            )

        pending = self._sync_queue.request_sync(tables)
        return {
            "queued": True,
            "pending_tables": pending,
            "message": f"Sync requested. Will execute after debounce window.",
        }

    def flush_sync_queue(self) -> dict[str, Any]:
        """Immediately flush the sync queue.

        Returns:
            Sync result dictionary.
        """
        if self._sync_queue is None:
            return {"skipped": True, "reason": "No sync queue initialized"}

        return self._sync_queue.force_flush()

    # ── Offline indexing ──

    def index(self, corpus_path: str) -> None:
        """Run the full offline indexing pipeline.

        1. Load corpus (JSONL)
        2. KG construction
        3. Hierarchical clustering
        4. C-HNSW build
        """
        log.info("Starting offline indexing from %s", corpus_path)

        # Load documents
        documents = self._load_corpus(corpus_path)
        log.info("Loaded %d documents", len(documents))

        # Phase 1: KG construction
        kg = self._kg_service.build(documents)

        # Phase 2: Hierarchical clustering
        hierarchy = self._cluster_service.build(kg)

        # Phase 3: C-HNSW construction
        self._index = self._chnsw_service.build(hierarchy)

        log.info("Offline indexing complete.")

    # ── Online retrieval ──

    def query(self, question: str) -> str:
        """Answer a question using hierarchical search + adaptive filtering."""
        log.info("Query: %s", question)

        # Load persisted vector index if not already in memory
        if self._index is None:
            vec_path = Path("data/chnsw_vectors.json")
            if vec_path.exists():
                log.info("Loading vector index from %s", vec_path)
                self._vector_index.load(str(vec_path))
            else:
                log.warning("No persisted vector index found at %s", vec_path)

        # Hierarchical search
        results = self._search_service.search(question, self._index)

        total = sum(len(layer) for layer in results)
        log.info("Retrieved %d results across %d layers", total, len(results))

        # Adaptive filtering & answer generation
        answer = self._filter_service.generate_answer(question, results)

        return answer

    # ── Add / Remove / Search ──

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add new documents to an existing index.

        Runs KG extraction + embedding for the new docs, then rebuilds
        the hierarchy and C-HNSW index from scratch.
        """
        log.info("Adding %d documents to existing index", len(documents))

        # Build KG for new docs (merges into existing store)
        kg = self._kg_service.build(documents)

        # Rebuild hierarchy + C-HNSW over the full graph
        all_entities = self._graph_store.get_all_entities()
        full_kg = KnowledgeGraph()
        for e in all_entities:
            full_kg.add_entity(e)
        for r in self._graph_store.get_all_relations():
            full_kg.add_relation(r)

        hierarchy = self._cluster_service.build(full_kg)
        self._index = self._chnsw_service.build(hierarchy)
        log.info("Re-indexed with new documents.")

    def remove_entity(self, entity_name: str) -> bool:
        """Remove an entity by name and cascade-delete its relations.

        Returns True if found and removed, False otherwise.
        """
        entity = self._graph_store.get_entity_by_name(entity_name)
        if entity is None:
            log.warning("Entity not found: %s", entity_name)
            return False
        self._graph_store.delete_entity(entity.id)
        log.info("Deleted entity '%s' (id=%s) and its relations.", entity_name, entity.id)
        return True

    def search_entities(self, query: str) -> list[dict[str, str]]:
        """Search entities by name substring. Returns a simple dict list."""
        entities = self._graph_store.search_entities_by_name(query)
        return [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
            }
            for e in entities
        ]

    def search_chunks(self, query: str) -> list[dict[str, str]]:
        """Search chunks by text substring."""
        chunks = self._doc_store.search_chunks(query)
        return [
            {
                "id": c.id,
                "text": c.text[:200] + ("..." if len(c.text) > 200 else ""),
                "source": c.source_doc,
            }
            for c in chunks
        ]

    def stats(self) -> dict[str, Any]:
        """Return DB statistics."""
        entities = self._graph_store.get_all_entities()
        relations = self._graph_store.get_all_relations()
        chunks = self._doc_store.get_all_chunks()
        hierarchy = self._doc_store.load_hierarchy()
        note_count = 0
        if self._memory_note_store is not None:
            note_count = self._memory_note_store.count()
        return {
            "entities": len(entities),
            "relations": len(relations),
            "chunks": len(chunks),
            "hierarchy_levels": hierarchy.height if hierarchy else 0,
            "memory_notes": note_count,
        }

    # ── Memory Notes (A-Mem inspired) ──

    def add_memory_note(
        self,
        input_data: dict[str, Any],
        enable_linking: bool = True,
        enable_evolution: bool | None = None,
        add_to_kg: bool = True,
    ) -> dict[str, Any]:
        """Add a structured memory note with LLM-generated enrichment.

        This creates a MemoryNote with auto-generated keywords, context,
        tags, and links to related memories (following the A-Mem design).

        Args:
            input_data: Dict with 'content' or 'text', optional 'category', 'tags'.
            enable_linking: Whether to find and link related notes.
            enable_evolution: Whether to update related notes.
            add_to_kg: Whether to also add to the knowledge graph.

        Returns:
            Dict with note ID and generated metadata.
        """
        if self._note_service is None or self._memory_note_store is None:
            raise RuntimeError("MemoryNote system not configured")

        # Build enriched note
        note = self._note_service.build_note(
            input_data,
            enable_linking=enable_linking,
            enable_evolution=enable_evolution,
        )

        # Save to note store
        self._memory_note_store.save_note(note)
        log.info("Created memory note %s with %d links", note.id, len(note.links))

        # Optionally add to KG pipeline
        if add_to_kg:
            doc = note.to_document()
            self.add_documents([doc])

        return {
            "id": note.id,
            "content": note.content[:200] + ("..." if len(note.content) > 200 else ""),
            "keywords": note.keywords,
            "context": note.context,
            "tags": note.tags,
            "category": note.category,
            "links": note.links,
            "timestamp": note.timestamp,
        }

    def get_memory_note(self, note_id: str) -> dict[str, Any] | None:
        """Retrieve a memory note by ID."""
        if self._memory_note_store is None:
            return None

        note = self._memory_note_store.get_note(note_id)
        if note is None:
            return None

        # Update retrieval stats
        note.increment_retrieval()
        self._memory_note_store.update_note(note)

        return {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "context": note.context,
            "tags": note.tags,
            "category": note.category,
            "links": note.links,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "retrieval_count": note.retrieval_count,
        }

    def get_related_notes(self, note_id: str) -> list[dict[str, Any]]:
        """Get notes linked to a given note."""
        if self._memory_note_store is None:
            return []

        note = self._memory_note_store.get_note(note_id)
        if note is None:
            return []

        related: list[dict[str, Any]] = []
        for linked_id, relation_type in note.links.items():
            linked = self._memory_note_store.get_note(linked_id)
            if linked:
                related.append({
                    "id": linked.id,
                    "content": linked.content[:200] + ("..." if len(linked.content) > 200 else ""),
                    "context": linked.context,
                    "relation_type": relation_type,
                })

        return related

    def search_notes_by_content(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Semantic search for notes by content similarity."""
        if self._memory_note_store is None:
            return []

        # Embed the query
        embedding = self._embedding.embed(query)

        # Find nearest notes
        notes = self._memory_note_store.get_nearest_notes(embedding, k)

        return [
            {
                "id": n.id,
                "content": n.content[:200] + ("..." if len(n.content) > 200 else ""),
                "context": n.context,
                "tags": n.tags,
            }
            for n in notes
        ]

    def delete_memory_note(self, note_id: str) -> bool:
        """Delete a memory note by ID."""
        if self._memory_note_store is None:
            return False

        note = self._memory_note_store.get_note(note_id)
        if note is None:
            return False

        self._memory_note_store.delete_note(note_id)
        log.info("Deleted memory note %s", note_id)
        return True

    # ── helpers ──

    @staticmethod
    def _load_corpus(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        documents: list[dict[str, Any]] = []
        text = p.read_text(encoding="utf-8")

        # Try JSONL (one JSON object per line)
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError:
                pass

        # If JSONL failed, try as a single JSON array
        if not documents:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse corpus: {exc}") from exc

        return documents
