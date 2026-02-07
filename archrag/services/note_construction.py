"""Service: MemoryNote construction with LLM-based enrichment.

Implements the A-Mem paper's note construction pipeline (Section 3):
1. Note Construction (P_s1) - Generate keywords, context, tags
2. Link Generation (P_s2) - Find and establish connections
3. Memory Evolution (P_s3) - Update related notes based on new info
"""

from __future__ import annotations

import json
import logging
from typing import Any

from archrag.domain.models import MemoryNote
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.llm import LLMPort
from archrag.ports.memory_note_store import MemoryNoteStorePort
from archrag.prompts.note_construction import (
    LINK_GENERATION_PROMPT,
    LINK_GENERATION_SYSTEM,
    MEMORY_EVOLUTION_PROMPT,
    MEMORY_EVOLUTION_SYSTEM,
    NOTE_CONSTRUCTION_PROMPT,
    NOTE_CONSTRUCTION_SYSTEM,
)

log = logging.getLogger(__name__)


class NoteConstructionService:
    """Build enriched MemoryNotes from raw input using LLM analysis."""

    def __init__(
        self,
        llm: LLMPort,
        embedding: EmbeddingPort,
        note_store: MemoryNoteStorePort,
        *,
        k_nearest: int = 10,
        enable_evolution: bool = True,
    ) -> None:
        """Initialize the service.

        Args:
            llm: LLM port for generating metadata.
            embedding: Embedding port for vectorization.
            note_store: Storage for memory notes.
            k_nearest: Number of nearest neighbors for link generation.
            enable_evolution: Whether to trigger memory evolution.
        """
        self._llm = llm
        self._embedding = embedding
        self._note_store = note_store
        self._k_nearest = k_nearest
        self._enable_evolution = enable_evolution

    def build_note(
        self,
        input_data: dict[str, Any],
        enable_linking: bool = True,
        enable_evolution: bool | None = None,
    ) -> MemoryNote:
        """Build an enriched MemoryNote from raw input.

        Args:
            input_data: Dict with at least 'content' or 'text' key.
                       Optional: 'category', 'tags', 'keywords'.
            enable_linking: Whether to find and establish links.
            enable_evolution: Override instance setting for evolution.

        Returns:
            Fully constructed MemoryNote with LLM-generated metadata.
        """
        # 1. Extract content from input
        content = self._extract_content(input_data)
        if not content:
            raise ValueError("Input must contain 'content' or 'text' field")

        log.info("Building note from content: %s...", content[:50])

        # 2. Generate metadata via LLM (P_s1)
        keywords, context, tags = self._generate_metadata(content)

        # Merge with any user-provided metadata
        if input_data.get("keywords"):
            keywords = list(set(keywords + input_data["keywords"]))
        if input_data.get("tags"):
            tags = list(set(tags + input_data["tags"]))

        category = input_data.get("category", "")

        # 3. Compute embedding
        embed_text = f"{content} {context} {' '.join(keywords)}"
        embedding = self._embedding.embed(embed_text)

        # 4. Create the note
        note = MemoryNote(
            content=content,
            keywords=keywords,
            context=context,
            tags=tags,
            category=category,
            embedding=embedding,
        )

        # 5. Link generation (P_s2)
        if enable_linking:
            candidates = self._note_store.get_nearest_notes(
                embedding, self._k_nearest, exclude_ids=[note.id]
            )
            if candidates:
                note.links = self._generate_links(note, candidates)
                log.info("Generated %d links for note %s", len(note.links), note.id)

        # 6. Memory evolution (P_s3)
        do_evolve = enable_evolution if enable_evolution is not None else self._enable_evolution
        if do_evolve and candidates:
            self._evolve_memories(note, candidates)

        return note

    def _extract_content(self, input_data: dict[str, Any]) -> str:
        """Extract content from various input formats."""
        if "content" in input_data:
            return str(input_data["content"]).strip()
        if "text" in input_data:
            return str(input_data["text"]).strip()
        if "context" in input_data:
            return str(input_data["context"]).strip()
        return ""

    def _generate_metadata(self, content: str) -> tuple[list[str], str, list[str]]:
        """Use LLM to generate keywords, context, and tags (P_s1)."""
        prompt = NOTE_CONSTRUCTION_PROMPT.format(content=content)

        try:
            result = self._llm.generate_json(prompt, system=NOTE_CONSTRUCTION_SYSTEM)
            keywords = result.get("keywords", [])
            context = result.get("context", "")
            tags = result.get("tags", [])

            # Ensure they're lists/strings
            if not isinstance(keywords, list):
                keywords = []
            if not isinstance(tags, list):
                tags = []
            if not isinstance(context, str):
                context = str(context) if context else ""

            return keywords, context, tags

        except Exception as exc:
            log.warning("Metadata generation failed: %s", exc)
            return [], "", []

    def _generate_links(
        self, note: MemoryNote, candidates: list[MemoryNote]
    ) -> dict[str, str]:
        """Use LLM to determine which candidates should be linked (P_s2)."""
        # Format candidates for the prompt
        candidates_text = "\n".join(
            f"- ID: {c.id}\n  Content: {c.content[:200]}...\n  Context: {c.context}\n  Keywords: {c.keywords}"
            for c in candidates
        )

        prompt = LINK_GENERATION_PROMPT.format(
            new_content=note.content[:500],
            new_context=note.context,
            new_keywords=note.keywords,
            candidates=candidates_text,
        )

        try:
            result = self._llm.generate_json(prompt, system=LINK_GENERATION_SYSTEM)
            links_data = result.get("links", [])

            # Build links dict: note_id -> relation_type
            links: dict[str, str] = {}
            valid_ids = {c.id for c in candidates}
            for link in links_data:
                nid = link.get("note_id", "")
                rel = link.get("relation_type", "related_to")
                if nid in valid_ids:
                    links[nid] = rel

            return links

        except Exception as exc:
            log.warning("Link generation failed: %s", exc)
            return {}

    def _evolve_memories(
        self, new_note: MemoryNote, neighbors: list[MemoryNote]
    ) -> None:
        """Update context/tags of related notes based on new info (P_s3)."""
        # Format neighbors for the prompt
        neighbors_text = "\n".join(
            f"- ID: {n.id}\n  Content: {n.content[:200]}...\n  Context: {n.context}\n  Tags: {n.tags}"
            for n in neighbors
        )

        prompt = MEMORY_EVOLUTION_PROMPT.format(
            new_content=new_note.content[:500],
            new_context=new_note.context,
            new_keywords=new_note.keywords,
            new_tags=new_note.tags,
            neighbors=neighbors_text,
        )

        try:
            result = self._llm.generate_json(prompt, system=MEMORY_EVOLUTION_SYSTEM)

            if not result.get("should_evolve", False):
                return

            updates = result.get("updates", [])
            neighbor_map = {n.id: n for n in neighbors}

            for update in updates:
                nid = update.get("note_id", "")
                if nid not in neighbor_map:
                    continue

                neighbor = neighbor_map[nid]
                changed = False

                # Update context if provided
                new_context = update.get("new_context")
                if new_context and isinstance(new_context, str):
                    old_context = neighbor.context
                    neighbor.context = new_context
                    neighbor.evolution_history.append({
                        "type": "context_update",
                        "triggered_by": new_note.id,
                        "old_value": old_context,
                        "new_value": new_context,
                    })
                    changed = True

                # Update tags if provided
                new_tags = update.get("new_tags")
                if new_tags and isinstance(new_tags, list):
                    old_tags = neighbor.tags
                    neighbor.tags = new_tags
                    neighbor.evolution_history.append({
                        "type": "tags_update",
                        "triggered_by": new_note.id,
                        "old_value": old_tags,
                        "new_value": new_tags,
                    })
                    changed = True

                if changed:
                    # Recompute embedding with updated metadata
                    embed_text = f"{neighbor.content} {neighbor.context} {' '.join(neighbor.keywords)}"
                    neighbor.embedding = self._embedding.embed(embed_text)
                    self._note_store.update_note(neighbor)
                    log.info("Evolved note %s based on new note %s", nid, new_note.id)

        except Exception as exc:
            log.warning("Memory evolution failed: %s", exc)
