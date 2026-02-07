"""Unified Ingestion Pipeline: All data flows through MemoryNote → Chunks → KG → C-HNSW.

This service ensures consistent data treatment regardless of input format:
- JSONL/JSON files
- SQL database records
- NoSQL database documents
- Direct API input

Pipeline:
    Input (any format)
    → MemoryNote (LLM enrichment: keywords, context, tags, links)
    → TextChunks (split for entity extraction)
    → Knowledge Graph (entities, relations)
    → Community Hierarchy (Leiden clustering)
    → C-HNSW Index (hierarchical vector search)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from archrag.domain.models import MemoryNote, TextChunk, KnowledgeGraph, Entity, Relation

if TYPE_CHECKING:
    from archrag.ports.document_store import DocumentStorePort
    from archrag.ports.embedding import EmbeddingPort
    from archrag.ports.graph_store import GraphStorePort
    from archrag.ports.llm import LLMPort
    from archrag.ports.memory_note_store import MemoryNoteStorePort
    from archrag.services.note_construction import NoteConstructionService

log = logging.getLogger(__name__)


class UnifiedIngestionPipeline:
    """Unified pipeline that routes all input formats through MemoryNote → KG.
    
    This ensures:
    1. All data gets LLM-enriched metadata (keywords, context, tags)
    2. All data is stored as MemoryNotes (for semantic search & evolution)
    3. All data flows through the full KG pipeline (for hierarchical traversal)
    
    Usage:
        pipeline = UnifiedIngestionPipeline(...)
        
        # From JSONL file
        notes = pipeline.ingest_file("data.jsonl")
        
        # From database records
        notes = pipeline.ingest_records([ExternalRecord(...), ...])
        
        # From raw dict input
        note = pipeline.ingest_single({"content": "...", "category": "..."})
    """
    
    def __init__(
        self,
        llm: "LLMPort",
        embedding: "EmbeddingPort",
        graph_store: "GraphStorePort",
        doc_store: "DocumentStorePort",
        note_store: "MemoryNoteStorePort",
        note_service: "NoteConstructionService",
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialize the unified pipeline.
        
        Args:
            llm: LLM port for entity/relation extraction.
            embedding: Embedding port for vectorization.
            graph_store: Storage for KG entities and relations.
            doc_store: Storage for text chunks and metadata.
            note_store: Storage for MemoryNotes.
            note_service: Service for building enriched notes.
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap between chunks.
        """
        self._llm = llm
        self._embedding = embedding
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._note_store = note_store
        self._note_service = note_service
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
        # Import extraction prompts
        from archrag.prompts.extraction import (
            ENTITY_RELATION_EXTRACTION_PROMPT,
            ENTITY_RELATION_EXTRACTION_SYSTEM,
        )
        self._extraction_prompt = ENTITY_RELATION_EXTRACTION_PROMPT
        self._extraction_system = ENTITY_RELATION_EXTRACTION_SYSTEM

    # ── Public API ──────────────────────────────────────────────────────────

    def ingest_single(
        self,
        input_data: dict[str, Any],
        *,
        skip_kg: bool = False,
    ) -> MemoryNote:
        """Ingest a single item through the unified pipeline.
        
        Args:
            input_data: Dict with 'content' or 'text' key, optional metadata.
            skip_kg: If True, only create MemoryNote, skip KG pipeline.
            
        Returns:
            The created MemoryNote.
        """
        # Step 1: Build MemoryNote with LLM enrichment
        note = self._note_service.build_note(input_data)
        
        # Step 2: Save to note store
        self._note_store.save_note(note)
        log.info("Created MemoryNote %s: %s...", note.id, note.content[:50])
        
        # Step 3: If not skipping KG, run through chunking and entity extraction
        if not skip_kg:
            self._process_note_to_kg(note)
        
        return note

    def ingest_batch(
        self,
        items: list[dict[str, Any]],
        *,
        skip_kg: bool = False,
    ) -> list[MemoryNote]:
        """Ingest multiple items through the unified pipeline.
        
        Args:
            items: List of dicts with 'content' or 'text' keys.
            skip_kg: If True, only create MemoryNotes, skip KG pipeline.
            
        Returns:
            List of created MemoryNotes.
        """
        notes: list[MemoryNote] = []
        
        for item in items:
            try:
                note = self.ingest_single(item, skip_kg=skip_kg)
                notes.append(note)
            except Exception as e:
                log.error("Failed to ingest item: %s", e)
                continue
        
        log.info("Ingested %d/%d items through unified pipeline", len(notes), len(items))
        return notes

    def ingest_file(
        self,
        path: str,
    ) -> list[MemoryNote]:
        """Ingest documents from a file (JSONL or JSON).
        
        Args:
            path: Path to corpus file.
            
        Returns:
            List of created MemoryNotes.
        """
        documents = self._load_file(path)
        log.info("Loaded %d documents from %s", len(documents), path)
        
        return self.ingest_batch(documents)

    def ingest_from_external_record(
        self,
        record: Any,  # ExternalRecord
    ) -> MemoryNote:
        """Ingest an ExternalRecord (from database sync) through unified pipeline.
        
        Args:
            record: ExternalRecord from SQL/NoSQL connector.
            
        Returns:
            The created MemoryNote.
        """
        # Convert ExternalRecord to input dict
        input_data = record.to_note_input()
        
        return self.ingest_single(input_data)

    def get_kg_from_notes(self) -> KnowledgeGraph:
        """Build a KnowledgeGraph from all stored notes.
        
        Useful after batch ingestion to get the full graph.
        
        Returns:
            Complete KnowledgeGraph from all ingested data.
        """
        entities = self._graph_store.get_all_entities()
        relations = self._graph_store.get_all_relations()
        
        kg = KnowledgeGraph()
        for e in entities:
            kg.add_entity(e)
        for r in relations:
            kg.add_relation(r)
            
        return kg

    # ── Internal Pipeline Steps ─────────────────────────────────────────────

    def _process_note_to_kg(self, note: MemoryNote) -> None:
        """Process a MemoryNote through chunking and KG extraction.
        
        This is the core pipeline step that:
        1. Chunks the note content
        2. Extracts entities and relations from each chunk
        3. Computes embeddings for entities
        4. Persists to graph store
        """
        # Step 1: Create document representation with note metadata
        doc = note.to_document()
        
        # Step 2: Chunk the content
        chunks = self._chunk_note(note)
        if not chunks:
            log.warning("No chunks created for note %s", note.id)
            return
            
        self._doc_store.save_chunks(chunks)
        log.debug("Created %d chunks from note %s", len(chunks), note.id)
        
        # Step 3: Extract entities and relations from chunks
        name_to_entity: dict[str, Entity] = {}
        relations: list[Relation] = []
        
        for chunk in chunks:
            extracted = self._extract_entities_relations(chunk)
            
            # Process entities
            for ent_data in extracted.get("entities", []):
                name = ent_data.get("name", "").strip()
                if not name:
                    continue
                    
                key = name.lower()
                if key in name_to_entity:
                    # Merge with existing entity
                    existing = name_to_entity[key]
                    if ent_data.get("description", ""):
                        existing.description += " " + ent_data["description"]
                    existing.source_chunk_ids.append(chunk.id)
                else:
                    # Create new entity
                    entity = Entity(
                        name=name,
                        description=ent_data.get("description", ""),
                        entity_type=ent_data.get("type", ""),
                        source_chunk_ids=[chunk.id],
                    )
                    name_to_entity[key] = entity
            
            # Process relations
            for rel_data in extracted.get("relations", []):
                src_name = rel_data.get("source", "").strip().lower()
                tgt_name = rel_data.get("target", "").strip().lower()
                
                if src_name in name_to_entity and tgt_name in name_to_entity:
                    rel = Relation(
                        source_id=name_to_entity[src_name].id,
                        target_id=name_to_entity[tgt_name].id,
                        description=rel_data.get("description", ""),
                        source_chunk_ids=[chunk.id],
                    )
                    relations.append(rel)
        
        # Step 4: Compute entity embeddings
        entity_list = list(name_to_entity.values())
        if entity_list:
            texts = [f"{e.name}: {e.description}" for e in entity_list]
            embeddings = self._embedding.embed_batch(texts)
            for entity, emb in zip(entity_list, embeddings):
                entity.embedding = emb
        
        # Step 5: Persist to graph store
        if entity_list:
            self._graph_store.save_entities(entity_list)
        if relations:
            self._graph_store.save_relations(relations)
            
        log.info(
            "KG updated from note %s: %d entities, %d relations",
            note.id, len(entity_list), len(relations),
        )

    def _chunk_note(self, note: MemoryNote) -> list[TextChunk]:
        """Chunk a MemoryNote's content into TextChunks.
        
        The chunk includes note metadata for provenance.
        """
        chunks: list[TextChunk] = []
        content = note.content
        
        start = 0
        chunk_idx = 0
        
        while start < len(content):
            end = start + self._chunk_size
            chunk_text = content[start:end]
            
            chunk = TextChunk(
                text=chunk_text,
                source_doc=note.id,  # Use note ID as source
                metadata={
                    "note_id": note.id,
                    "note_category": note.category,
                    "note_tags": note.tags,
                    "note_keywords": note.keywords,
                    "chunk_index": chunk_idx,
                },
            )
            chunks.append(chunk)
            
            start += self._chunk_size - self._chunk_overlap
            chunk_idx += 1
        
        return chunks

    def _extract_entities_relations(self, chunk: TextChunk) -> dict[str, Any]:
        """Extract entities and relations from a text chunk via LLM."""
        prompt = self._extraction_prompt.format(text=chunk.text)
        
        try:
            result = self._llm.generate_json(
                prompt, system=self._extraction_system
            )
            return result
        except Exception as exc:
            log.warning("Extraction failed for chunk %s: %s", chunk.id, exc)
            return {"entities": [], "relations": []}

    def _load_file(self, path: str) -> list[dict[str, Any]]:
        """Load documents from a file (JSONL or JSON)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        text = p.read_text(encoding="utf-8")
        documents: list[dict[str, Any]] = []
        
        # Try JSONL first (one JSON object per line)
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
                raise ValueError(f"Cannot parse file: {exc}") from exc
        
        return documents
