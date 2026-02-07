"""MemoryNote store adapter: SQLite."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from archrag.domain.models import MemoryNote
from archrag.ports.memory_note_store import MemoryNoteStorePort


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteMemoryNoteStore(MemoryNoteStorePort):
    """Store MemoryNotes in SQLite with JSON serialization for complex fields."""

    def __init__(self, db_path: str = "data/archrag.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_notes (
                id                TEXT PRIMARY KEY,
                content           TEXT NOT NULL,
                timestamp         TEXT NOT NULL,
                last_accessed     TEXT,
                keywords          TEXT NOT NULL DEFAULT '[]',
                context           TEXT NOT NULL DEFAULT '',
                tags              TEXT NOT NULL DEFAULT '[]',
                category          TEXT NOT NULL DEFAULT '',
                links             TEXT NOT NULL DEFAULT '{}',
                retrieval_count   INTEGER NOT NULL DEFAULT 0,
                evolution_history TEXT NOT NULL DEFAULT '[]',
                embedding         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_notes_category ON memory_notes(category);
            CREATE INDEX IF NOT EXISTS idx_notes_timestamp ON memory_notes(timestamp);
            """
        )
        self._conn.commit()

    def _row_to_note(self, row: tuple) -> MemoryNote:
        """Convert a database row to a MemoryNote."""
        return MemoryNote(
            id=row[0],
            content=row[1],
            timestamp=row[2],
            last_accessed=row[3],
            keywords=json.loads(row[4]) if row[4] else [],
            context=row[5] or "",
            tags=json.loads(row[6]) if row[6] else [],
            category=row[7] or "",
            links=json.loads(row[8]) if row[8] else {},
            retrieval_count=row[9] or 0,
            evolution_history=json.loads(row[10]) if row[10] else [],
            embedding=json.loads(row[11]) if row[11] else None,
        )

    def _note_to_row(self, note: MemoryNote) -> tuple:
        """Convert a MemoryNote to a database row tuple."""
        return (
            note.id,
            note.content,
            note.timestamp,
            note.last_accessed,
            json.dumps(note.keywords),
            note.context,
            json.dumps(note.tags),
            note.category,
            json.dumps(note.links),
            note.retrieval_count,
            json.dumps(note.evolution_history),
            json.dumps(note.embedding) if note.embedding else None,
        )

    # ── CRUD ──

    def save_note(self, note: MemoryNote) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_notes 
               (id, content, timestamp, last_accessed, keywords, context, 
                tags, category, links, retrieval_count, evolution_history, embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            self._note_to_row(note),
        )
        self._conn.commit()

    def get_note(self, note_id: str) -> MemoryNote | None:
        cur = self._conn.execute(
            "SELECT * FROM memory_notes WHERE id=?", (note_id,)
        )
        row = cur.fetchone()
        return self._row_to_note(row) if row else None

    def get_all_notes(self) -> list[MemoryNote]:
        cur = self._conn.execute("SELECT * FROM memory_notes ORDER BY timestamp DESC")
        return [self._row_to_note(row) for row in cur.fetchall()]

    def update_note(self, note: MemoryNote) -> None:
        self.save_note(note)  # INSERT OR REPLACE handles update

    def delete_note(self, note_id: str) -> None:
        self._conn.execute("DELETE FROM memory_notes WHERE id=?", (note_id,))
        self._conn.commit()

    # ── Similarity search ──

    def get_nearest_notes(
        self,
        embedding: list[float],
        k: int,
        exclude_ids: list[str] | None = None,
    ) -> list[MemoryNote]:
        """Find k nearest notes by cosine similarity.

        Note: This is a brute-force implementation. For large datasets,
        consider using a vector database or FAISS.
        """
        exclude_set = set(exclude_ids or [])

        cur = self._conn.execute(
            "SELECT * FROM memory_notes WHERE embedding IS NOT NULL"
        )

        scored: list[tuple[float, MemoryNote]] = []
        for row in cur.fetchall():
            if row[0] in exclude_set:
                continue
            note = self._row_to_note(row)
            if note.embedding:
                sim = _cosine_similarity(embedding, note.embedding)
                scored.append((sim, note))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [note for _, note in scored[:k]]

    # ── Tag-based search ──

    def search_by_tags(self, tags: list[str]) -> list[MemoryNote]:
        if not tags:
            return []

        # SQLite JSON search: check if any tag is in the tags array
        # Using LIKE for simplicity; for better performance use JSON functions
        conditions = []
        params = []
        for tag in tags:
            conditions.append("LOWER(tags) LIKE LOWER(?)")
            params.append(f'%"{tag}"%')

        query = f"SELECT * FROM memory_notes WHERE {' OR '.join(conditions)}"
        cur = self._conn.execute(query, params)
        return [self._row_to_note(row) for row in cur.fetchall()]

    def search_by_keywords(self, keywords: list[str]) -> list[MemoryNote]:
        if not keywords:
            return []

        conditions = []
        params = []
        for kw in keywords:
            conditions.append("LOWER(keywords) LIKE LOWER(?)")
            params.append(f'%"{kw}"%')

        query = f"SELECT * FROM memory_notes WHERE {' OR '.join(conditions)}"
        cur = self._conn.execute(query, params)
        return [self._row_to_note(row) for row in cur.fetchall()]

    # ── Lifecycle ──

    def clear(self) -> None:
        self._conn.execute("DELETE FROM memory_notes")
        self._conn.commit()

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM memory_notes")
        return cur.fetchone()[0]
