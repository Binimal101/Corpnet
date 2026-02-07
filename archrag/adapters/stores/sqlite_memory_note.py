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
        self._migrate_schema()
        self._create_tables()

    def _migrate_schema(self) -> None:
        """Migrate existing database schema to new format."""
        try:
            # Check if table exists
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_notes'"
            )
            if not cur.fetchone():
                return  # Table doesn't exist, no migration needed

            # Check current schema
            cur = self._conn.execute("PRAGMA table_info(memory_notes)")
            columns = {row[1]: row[2] for row in cur.fetchall()}

            # If schema already matches, no migration needed
            if "last_updated" in columns and "embedding_model" in columns:
                if "timestamp" not in columns and "last_accessed" not in columns:
                    return  # Schema is already up to date

            # Need to recreate table with new schema
            self._conn.execute("BEGIN TRANSACTION")
            try:
                # Create new table with correct schema
                self._conn.execute("""
                    CREATE TABLE memory_notes_new (
                        id                TEXT PRIMARY KEY,
                        content           TEXT NOT NULL,
                        last_updated      TEXT,
                        keywords          TEXT NOT NULL DEFAULT '[]',
                        tags              TEXT NOT NULL DEFAULT '[]',
                        category          TEXT NOT NULL DEFAULT '',
                        retrieval_count   INTEGER NOT NULL DEFAULT 0,
                        embedding         TEXT,
                        embedding_model   TEXT NOT NULL DEFAULT ''
                    )
                """)
                
                # Copy data from old table, mapping columns
                select_cols = []
                insert_cols = ["id", "content", "last_updated", "keywords", "tags", 
                              "category", "retrieval_count", "embedding", "embedding_model"]
                
                # Map old columns to new
                if "id" in columns:
                    select_cols.append("id")
                if "content" in columns:
                    select_cols.append("content")
                # Map last_accessed or timestamp to last_updated
                if "last_updated" in columns:
                    select_cols.append("last_updated")
                elif "last_accessed" in columns:
                    select_cols.append("last_accessed AS last_updated")
                elif "timestamp" in columns:
                    select_cols.append("timestamp AS last_updated")
                else:
                    select_cols.append("NULL AS last_updated")
                
                if "keywords" in columns:
                    select_cols.append("keywords")
                else:
                    select_cols.append("'[]' AS keywords")
                
                if "tags" in columns:
                    select_cols.append("tags")
                else:
                    select_cols.append("'[]' AS tags")
                
                if "category" in columns:
                    select_cols.append("category")
                else:
                    select_cols.append("'' AS category")
                
                if "retrieval_count" in columns:
                    select_cols.append("retrieval_count")
                else:
                    select_cols.append("0 AS retrieval_count")
                
                if "embedding" in columns:
                    select_cols.append("embedding")
                else:
                    select_cols.append("NULL AS embedding")
                
                # embedding_model gets default empty string
                select_cols.append("'' AS embedding_model")
                
                self._conn.execute(
                    f"INSERT INTO memory_notes_new ({', '.join(insert_cols)}) "
                    f"SELECT {', '.join(select_cols)} FROM memory_notes"
                )
                
                # Drop old table and rename new
                self._conn.execute("DROP TABLE memory_notes")
                self._conn.execute("ALTER TABLE memory_notes_new RENAME TO memory_notes")
                self._conn.commit()
                
            except Exception as e:
                self._conn.rollback()
                import logging
                log = logging.getLogger(__name__)
                log.warning("Schema migration failed: %s", e)
                # Recreate table from scratch if migration fails
                self._conn.execute("DROP TABLE IF EXISTS memory_notes")
                self._conn.commit()

        except Exception as e:
            # If migration fails, log but don't crash
            import logging
            log = logging.getLogger(__name__)
            log.warning("Schema migration check failed: %s", e)

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_notes (
                id                TEXT PRIMARY KEY,
                content           TEXT NOT NULL,
                last_updated      TEXT,
                keywords          TEXT NOT NULL DEFAULT '[]',
                tags              TEXT NOT NULL DEFAULT '[]',
                category          TEXT NOT NULL DEFAULT '',
                retrieval_count   INTEGER NOT NULL DEFAULT 0,
                embedding         TEXT,
                embedding_model   TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_notes_category ON memory_notes(category);
            CREATE INDEX IF NOT EXISTS idx_notes_last_updated ON memory_notes(last_updated);
            """
        )
        self._conn.commit()

    def _row_to_note(self, row: tuple) -> MemoryNote:
        """Convert a database row to a MemoryNote."""
        return MemoryNote(
            id=row[0],
            content=row[1],
            last_updated=row[2],
            keywords=json.loads(row[3]) if row[3] else [],
            tags=json.loads(row[4]) if row[4] else [],
            category=row[5] or "",
            retrieval_count=row[6] or 0,
            embedding=json.loads(row[7]) if row[7] else None,
            embedding_model=row[8] or "",
        )

    def _note_to_row(self, note: MemoryNote) -> tuple:
        """Convert a MemoryNote to a database row tuple."""
        return (
            note.id,
            note.content,
            note.last_updated,
            json.dumps(note.keywords),
            json.dumps(note.tags),
            note.category,
            note.retrieval_count,
            json.dumps(note.embedding) if note.embedding else None,
            note.embedding_model,
        )

    # ── CRUD ──

    def save_note(self, note: MemoryNote) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_notes 
               (id, content, last_updated, keywords, tags, category, 
                retrieval_count, embedding, embedding_model)
               VALUES (?,?,?,?,?,?,?,?,?)""",
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
        cur = self._conn.execute("SELECT * FROM memory_notes ORDER BY last_updated DESC")
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
