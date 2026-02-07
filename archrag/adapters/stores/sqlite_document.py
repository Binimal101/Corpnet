"""Document store adapter: SQLite."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from archrag.domain.models import Community, CommunityHierarchy, TextChunk
from archrag.ports.document_store import DocumentStorePort


class SQLiteDocumentStore(DocumentStorePort):
    """Store chunks, communities, and hierarchy metadata in SQLite."""

    def __init__(self, db_path: str = "data/archrag.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id              TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                last_updated    TEXT NOT NULL DEFAULT '',
                keywords        TEXT NOT NULL DEFAULT '[]',
                tags            TEXT NOT NULL DEFAULT '[]',
                category        TEXT NOT NULL DEFAULT '',
                retrieval_count INTEGER NOT NULL DEFAULT 0,
                embedding_model TEXT NOT NULL DEFAULT '',
                embedding       TEXT
            );

            CREATE TABLE IF NOT EXISTS communities (
                id         TEXT PRIMARY KEY,
                level      INTEGER NOT NULL DEFAULT 0,
                member_ids TEXT NOT NULL DEFAULT '[]',
                summary    TEXT NOT NULL DEFAULT '',
                embedding  TEXT,
                parent_id  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_comm_level ON communities(level);

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ── chunks ──

    def save_chunk(self, chunk: TextChunk) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, content, last_updated, keywords, tags, category,
                retrieval_count, embedding_model, embedding)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            self._chunk_row(chunk),
        )
        self._conn.commit()

    def save_chunks(self, chunks: list[TextChunk]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, content, last_updated, keywords, tags, category,
                retrieval_count, embedding_model, embedding)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [self._chunk_row(c) for c in chunks],
        )
        self._conn.commit()

    @staticmethod
    def _chunk_row(c: TextChunk) -> tuple:
        return (
            c.id,
            c.content,
            c.last_updated,
            json.dumps(c.keywords),
            json.dumps(c.tags),
            c.category,
            c.retrieval_count,
            c.embedding_model,
            json.dumps(c.embedding) if c.embedding else None,
        )

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        cur = self._conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_chunk(row)

    def get_all_chunks(self) -> list[TextChunk]:
        cur = self._conn.execute("SELECT * FROM chunks")
        return [self._row_to_chunk(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_chunk(r: tuple) -> TextChunk:
        return TextChunk(
            id=r[0],
            content=r[1],
            last_updated=r[2],
            keywords=json.loads(r[3]),
            tags=json.loads(r[4]),
            category=r[5],
            retrieval_count=r[6],
            embedding_model=r[7],
            embedding=json.loads(r[8]) if r[8] else None,
        )

    # ── communities ──

    def save_community(self, community: Community) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO communities
               (id, level, member_ids, summary, embedding, parent_id)
               VALUES (?,?,?,?,?,?)""",
            (
                community.id,
                community.level,
                json.dumps(community.member_ids),
                community.summary,
                json.dumps(community.embedding) if community.embedding else None,
                community.parent_id,
            ),
        )
        self._conn.commit()

    def save_communities(self, communities: list[Community]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO communities
               (id, level, member_ids, summary, embedding, parent_id)
               VALUES (?,?,?,?,?,?)""",
            [
                (
                    c.id,
                    c.level,
                    json.dumps(c.member_ids),
                    c.summary,
                    json.dumps(c.embedding) if c.embedding else None,
                    c.parent_id,
                )
                for c in communities
            ],
        )
        self._conn.commit()

    def get_community(self, community_id: str) -> Community | None:
        cur = self._conn.execute(
            "SELECT * FROM communities WHERE id=?", (community_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return Community(
            id=row[0],
            level=row[1],
            member_ids=json.loads(row[2]),
            summary=row[3],
            embedding=json.loads(row[4]) if row[4] else None,
            parent_id=row[5],
        )

    def get_communities_at_level(self, level: int) -> list[Community]:
        cur = self._conn.execute(
            "SELECT * FROM communities WHERE level=?", (level,)
        )
        return [
            Community(
                id=r[0],
                level=r[1],
                member_ids=json.loads(r[2]),
                summary=r[3],
                embedding=json.loads(r[4]) if r[4] else None,
                parent_id=r[5],
            )
            for r in cur.fetchall()
        ]

    # ── hierarchy ──

    def save_hierarchy(self, hierarchy: CommunityHierarchy) -> None:
        # Persist all communities and store the structure as meta
        for level_comms in hierarchy.levels:
            self.save_communities(level_comms)
        structure = {
            "height": hierarchy.height,
            "level_ids": [
                [c.id for c in level_comms] for level_comms in hierarchy.levels
            ],
        }
        self.put_meta("hierarchy_structure", json.dumps(structure))

    def load_hierarchy(self) -> CommunityHierarchy | None:
        raw = self.get_meta("hierarchy_structure")
        if raw is None:
            return None
        structure = json.loads(raw)
        levels: list[list[Community]] = []
        for id_list in structure["level_ids"]:
            comms = [self.get_community(cid) for cid in id_list]
            levels.append([c for c in comms if c is not None])
        return CommunityHierarchy(levels=levels)

    # ── meta ──

    def put_meta(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (key, str(value)),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Any:
        cur = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def clear(self) -> None:
        self._conn.executescript(
            "DELETE FROM chunks; DELETE FROM communities; DELETE FROM meta;"
        )
        self._conn.commit()

    def delete_chunk(self, chunk_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
        self._conn.commit()

    def search_chunks(self, query: str) -> list[TextChunk]:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE LOWER(content) LIKE LOWER(?)",
            (f"%{query}%",),
        )
        return [self._row_to_chunk(r) for r in cur.fetchall()]

    def clone(self) -> "SQLiteDocumentStore":
        """Create an independent copy via sqlite3 backup API."""
        fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="archrag_doc_")
        import os
        os.close(fd)
        dst_conn = sqlite3.connect(tmp_path)
        self._conn.backup(dst_conn)
        dst_conn.close()
        return SQLiteDocumentStore(db_path=tmp_path)
