"""Port: persistent storage for MemoryNotes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from archrag.domain.models import MemoryNote


class MemoryNoteStorePort(ABC):
    """Abstract interface for MemoryNote persistence.

    Supports CRUD operations and similarity-based retrieval
    for the A-Mem inspired memory system.
    """

    # ── CRUD ──

    @abstractmethod
    def save_note(self, note: MemoryNote) -> None:
        """Persist a new memory note."""
        ...

    @abstractmethod
    def get_note(self, note_id: str) -> MemoryNote | None:
        """Retrieve a note by ID."""
        ...

    @abstractmethod
    def get_all_notes(self) -> list[MemoryNote]:
        """Retrieve all stored notes."""
        ...

    @abstractmethod
    def update_note(self, note: MemoryNote) -> None:
        """Update an existing note (matched by ID)."""
        ...

    @abstractmethod
    def delete_note(self, note_id: str) -> None:
        """Delete a note by ID."""
        ...

    # ── Similarity search ──

    @abstractmethod
    def get_nearest_notes(
        self, embedding: list[float], k: int, exclude_ids: list[str] | None = None
    ) -> list[MemoryNote]:
        """Find k nearest notes by cosine similarity.

        Args:
            embedding: Query vector.
            k: Number of results.
            exclude_ids: Optional list of note IDs to exclude.
        """
        ...

    # ── Tag-based search ──

    @abstractmethod
    def search_by_tags(self, tags: list[str]) -> list[MemoryNote]:
        """Find notes matching any of the given tags."""
        ...

    @abstractmethod
    def search_by_keywords(self, keywords: list[str]) -> list[MemoryNote]:
        """Find notes matching any of the given keywords."""
        ...

    # ── Lifecycle ──

    @abstractmethod
    def clear(self) -> None:
        """Remove all notes."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of notes."""
        ...
