"""Service: MemoryNote construction with LLM-based enrichment.

Implements the A-Mem paper's note construction pipeline (Section 3):
1. Note Construction (P_s1) - Generate keywords and tags
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
    ) -> None:
        """Initialize the service.

        Args:
            llm: LLM port for generating metadata.
            embedding: Embedding port for vectorization.
            note_store: Storage for memory notes.
        """
        self._llm = llm
        self._embedding = embedding
        self._note_store = note_store

    def build_note(
        self,
        input_data: dict[str, Any],
        access_id: str = "",
    ) -> MemoryNote:
        """Build an enriched MemoryNote from raw input.

        Args:
            input_data: Dict with at least 'content' or 'text' key.
                       Optional: 'category', 'tags', 'keywords', 'access_id'.
            access_id: Hierarchical access scope for the note.
                      If not provided, uses input_data['access_id'] or empty.

        Returns:
            Fully constructed MemoryNote with LLM-generated metadata.
        """
        # 1. Extract content from input
        content = self._extract_content(input_data)
        if not content:
            raise ValueError("Input must contain 'content' or 'text' field")

        log.info("Building note from content: %s...", content[:50])

        # 2. Generate metadata via LLM (P_s1) - only keywords and tags
        keywords, tags = self._generate_metadata(content)

        # Merge with any user-provided metadata
        if input_data.get("keywords"):
            keywords = list(set(keywords + input_data["keywords"]))
        if input_data.get("tags"):
            tags = list(set(tags + input_data["tags"]))

        category = input_data.get("category", "")
        retrieval_count = input_data.get("retrieval_count", 0)
        last_updated = input_data.get("last_updated")
        
        # Determine access_id: parameter > input_data > empty
        note_access_id = access_id or input_data.get("access_id", "")

        # 3. Compute embedding with structured format
        embed_text = self._build_embedding_text(
            content=content,
            last_updated=last_updated or "",
            tags=tags,
            keywords=keywords,
            category=category,
            retrieval_count=retrieval_count,
        )
        embedding = self._embedding.embed(embed_text)
        embedding_model = self._embedding.model_name()

        # 4. Create the note
        note = MemoryNote(
            content=content,
            keywords=keywords,
            tags=tags,
            category=category,
            retrieval_count=retrieval_count,
            last_updated=last_updated,
            embedding=embedding,
            embedding_model=embedding_model,
            access_id=note_access_id,
        )

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

    def _generate_metadata(self, content: str) -> tuple[list[str], list[str]]:
        """Use LLM to generate keywords and tags (P_s1)."""
        prompt = NOTE_CONSTRUCTION_PROMPT.format(content=content)

        try:
            result = self._llm.generate_json(prompt, system=NOTE_CONSTRUCTION_SYSTEM)
            keywords = result.get("keywords", [])
            tags = result.get("tags", [])

            # Ensure they're lists
            if not isinstance(keywords, list):
                keywords = []
            if not isinstance(tags, list):
                tags = []

            return keywords, tags

        except Exception as exc:
            log.warning("Metadata generation failed: %s", exc)
            return [], []
    
    def _build_embedding_text(
        self,
        content: str,
        last_updated: str,
        tags: list[str],
        keywords: list[str],
        category: str,
        retrieval_count: int,
    ) -> str:
        """Build the embedding text with structured delimiters.
        
        Format: key:value|key:value|...
        """
        parts = []
        
        # Content
        parts.append(f"content:{content}")
        
        # Last updated
        parts.append(f"last_updated:{last_updated}")
        
        # Tags (comma-separated)
        tags_str = ",".join(tags) if tags else ""
        parts.append(f"tags:{tags_str}")
        
        # Keywords (comma-separated)
        keywords_str = ",".join(keywords) if keywords else ""
        parts.append(f"keywords:{keywords_str}")
        
        # Category
        parts.append(f"category:{category}")
        
        # Retrieval count
        parts.append(f"retrieval_count:{retrieval_count}")
        
        return "|".join(parts)

