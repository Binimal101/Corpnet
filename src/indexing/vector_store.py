"""Vector store implementations for chunk storage and similarity search.

Supports:
- PgVectorStore: PostgreSQL + pgvector/pgvectorscale (production)
- InMemoryVectorStore: In-memory with numpy (testing/development)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from src.core.types import DocumentChunk, SearchResult

if TYPE_CHECKING:
    from src.core.config import DatabaseConfig

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract interface for vector storage and search."""
    
    @abstractmethod
    def insert(self, chunk: DocumentChunk) -> str:
        """Insert a chunk into the store. Returns chunk_id."""
        ...
    
    @abstractmethod
    def bulk_insert(self, chunks: list[DocumentChunk]) -> list[str]:
        """Insert multiple chunks. Returns list of chunk_ids."""
        ...
    
    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        """Top-k similarity search."""
        ...
    
    @abstractmethod
    def search_by_threshold(
        self,
        embedding: list[float],
        threshold: float,
        max_results: int = 20,
    ) -> list[SearchResult]:
        """Threshold-based search. Returns all results with similarity >= threshold."""
        ...
    
    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by ID."""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks."""
        ...


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using numpy for similarity search."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
    
    def insert(self, chunk: DocumentChunk) -> str:
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
        
        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = np.array(chunk.embedding)
        return chunk.chunk_id
    
    def bulk_insert(self, chunks: list[DocumentChunk]) -> list[str]:
        return [self.insert(c) for c in chunks]
    
    def search(self, embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        if not self._chunks:
            return []
        
        query = np.array(embedding)
        query = query / (np.linalg.norm(query) + 1e-10)
        
        scores = []
        for chunk_id, vec in self._vectors.items():
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
            score = float(np.dot(query, vec_norm))
            scores.append((chunk_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in scores[:top_k]:
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=chunk.text,
                score=score,
                source=chunk.doc_id,
                peer_id="local",
                metadata=chunk.metadata,
            ))
        
        return results
    
    def search_by_threshold(
        self,
        embedding: list[float],
        threshold: float,
        max_results: int = 20,
    ) -> list[SearchResult]:
        if not self._chunks:
            return []
        
        query = np.array(embedding)
        query = query / (np.linalg.norm(query) + 1e-10)
        
        scores = []
        for chunk_id, vec in self._vectors.items():
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
            score = float(np.dot(query, vec_norm))
            if score >= threshold:
                scores.append((chunk_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Always return at least 1 result (best match) even if below threshold
        if not scores and self._chunks:
            best_id = None
            best_score = -float("inf")
            for chunk_id, vec in self._vectors.items():
                vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
                score = float(np.dot(query, vec_norm))
                if score > best_score:
                    best_score = score
                    best_id = chunk_id
            if best_id:
                scores = [(best_id, best_score)]
        
        results = []
        for chunk_id, score in scores[:max_results]:
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=chunk.text,
                score=score,
                source=chunk.doc_id,
                peer_id="local",
                metadata=chunk.metadata,
            ))
        
        return results
    
    def delete(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)
        self._vectors.pop(chunk_id, None)
    
    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore(VectorStore):
    """PostgreSQL vector store using pgvector/pgvectorscale."""
    
    def __init__(self, database_url: str, table_name: str = "vectors"):
        self.database_url = database_url
        self.table_name = table_name
        self._conn = None
    
    def _get_connection(self):
        if self._conn is None:
            import psycopg2
            self._conn = psycopg2.connect(self.database_url)
        return self._conn
    
    def insert(self, chunk: DocumentChunk) -> str:
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
        
        import json
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table_name} (chunk_id, doc_id, embedding, text_content, labels, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    text_content = EXCLUDED.text_content,
                    metadata = EXCLUDED.metadata
                RETURNING chunk_id
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.embedding,
                    chunk.text,
                    list(chunk.labels),
                    json.dumps(chunk.metadata),
                ),
            )
            conn.commit()
        return chunk.chunk_id
    
    def bulk_insert(self, chunks: list[DocumentChunk]) -> list[str]:
        return [self.insert(c) for c in chunks]
    
    def search(self, embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, doc_id, text_content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            
            results = []
            for row in cur.fetchall():
                results.append(SearchResult(
                    chunk_id=row[0],
                    text=row[2],
                    score=float(row[4]),
                    source=row[1],
                    peer_id="local",
                    metadata=row[3] or {},
                ))
            return results
    
    def search_by_threshold(
        self,
        embedding: list[float],
        threshold: float,
        max_results: int = 20,
    ) -> list[SearchResult]:
        conn = self._get_connection()
        max_distance = 1.0 - threshold
        
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, doc_id, text_content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table_name}
                WHERE (embedding <=> %s::vector) <= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, max_distance, embedding, max_results),
            )
            
            results = []
            for row in cur.fetchall():
                results.append(SearchResult(
                    chunk_id=row[0],
                    text=row[2],
                    score=float(row[4]),
                    source=row[1],
                    peer_id="local",
                    metadata=row[3] or {},
                ))
            
            # Guarantee at least 1 result
            if not results:
                cur.execute(
                    f"""
                    SELECT chunk_id, doc_id, text_content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (embedding, embedding),
                )
                for row in cur.fetchall():
                    results.append(SearchResult(
                        chunk_id=row[0],
                        text=row[2],
                        score=float(row[4]),
                        source=row[1],
                        peer_id="local",
                        metadata=row[3] or {},
                    ))
            
            return results
    
    def delete(self, chunk_id: str) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE chunk_id = %s", (chunk_id,))
            conn.commit()
    
    def count(self) -> int:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cur.fetchone()
            return result[0] if result else 0
    
    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
