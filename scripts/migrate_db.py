"""Database migration script.

Creates the database schema for DAC-HRAG.
Uses pgvector/pgvectorscale for vector storage.
"""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SCHEMA = """
-- DAC-HRAG Database Schema
-- PostgreSQL + pgvector/pgvectorscale
-- Default embedding dimension: 768 (nomic-embed-text)

CREATE EXTENSION IF NOT EXISTS vector;
-- CREATE EXTENSION IF NOT EXISTS vectorscale;  -- Enable for TimescaleDB

-- Vectors table (document chunks)
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id UUID PRIMARY KEY,
    doc_id TEXT NOT NULL,
    embedding vector(768),
    text_content TEXT,
    labels TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for similarity search
-- For pgvectorscale: CREATE INDEX ON vectors USING diskann (embedding);
CREATE INDEX IF NOT EXISTS vectors_embedding_idx
    ON vectors
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS vectors_labels_idx ON vectors USING gin (labels);
CREATE INDEX IF NOT EXISTS vectors_doc_id_idx ON vectors (doc_id);

-- Communities table
CREATE TABLE IF NOT EXISTS communities (
    community_id UUID PRIMARY KEY,
    level INT NOT NULL,
    summary TEXT NOT NULL,
    summary_embedding vector(768) NOT NULL,
    labels TEXT[] NOT NULL DEFAULT '{}',
    member_ids UUID[] NOT NULL DEFAULT '{}',
    parent_id UUID REFERENCES communities(community_id),
    children_ids UUID[] NOT NULL DEFAULT '{}',
    peer_id TEXT NOT NULL DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- For pgvectorscale: CREATE INDEX ON communities USING diskann (summary_embedding);
CREATE INDEX IF NOT EXISTS communities_embedding_idx
    ON communities
    USING ivfflat (summary_embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS communities_level_idx ON communities (level);
CREATE INDEX IF NOT EXISTS communities_labels_idx ON communities USING gin (labels);

-- Peer registry
CREATE TABLE IF NOT EXISTS peers (
    peer_id TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    port INT NOT NULL,
    communities TEXT[] NOT NULL DEFAULT '{}',
    is_super_peer BOOLEAN DEFAULT FALSE,
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Query log for analytics
CREATE TABLE IF NOT EXISTS query_log (
    query_id UUID PRIMARY KEY,
    query_text TEXT NOT NULL,
    answer TEXT,
    result_count INT DEFAULT 0,
    latency_ms FLOAT,
    routing_path TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS query_log_created_idx ON query_log (created_at DESC);
"""


def migrate():
    """Run database migration."""
    from src.core.config import get_settings
    
    settings = get_settings()
    db_url = settings.database.url
    
    print(f"Connecting to: {db_url}")
    
    import psycopg2
    
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Execute schema
            cur.execute(SCHEMA)
        
        print("Migration complete!")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
