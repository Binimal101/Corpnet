"""Seed sample data for demos."""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def seed():
    """Seed the system with sample data."""
    from src.core.config import get_settings
    from src.core.embeddings import create_embedding_provider
    from src.core.llm import create_llm_provider
    from src.ingestion.chunker import RecursiveChunker
    from src.ingestion.entity_extractor import EntityExtractor
    from src.indexing.vector_store import InMemoryVectorStore
    
    settings = get_settings()
    embedder = create_embedding_provider(settings.embeddings)
    llm = create_llm_provider(settings.llm)
    
    print("Seeding sample data...")
    
    # Sample documents
    sample_docs = [
        ("engineering", """
        # Engineering Documentation
        
        Our backend is built with Python and FastAPI. We use PostgreSQL
        with pgvector for vector storage. The system implements a 
        distributed RAG architecture with peer-to-peer communication.
        
        ## Key Components
        - Ingestion Pipeline: Chunks documents and extracts entities
        - Clustering: Uses Leiden algorithm for community detection
        - Routing: Hierarchical descent through community levels
        """),
        ("marketing", """
        # Marketing Guidelines
        
        Brand voice should be professional yet approachable. Our target
        audience is technical decision-makers in enterprise companies.
        
        ## Campaigns
        - Q1: Developer conference sponsorships
        - Q2: Content marketing push
        - Q3: Product launch campaign
        """),
    ]
    
    chunker = RecursiveChunker(chunk_size=256, chunk_overlap=32)
    extractor = EntityExtractor(llm)
    store = InMemoryVectorStore(dimension=settings.embeddings.dimension)
    
    total_chunks = 0
    for doc_id, text in sample_docs:
        print(f"  Processing {doc_id}...")
        
        # Chunk
        chunks = chunker.chunk(text, doc_id, {"source": f"sample/{doc_id}"})
        
        # Extract entities (skip for speed in demo)
        # for chunk in chunks:
        #     extractor.extract_sync(chunk)
        
        # Embed
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        
        # Store
        for chunk in chunks:
            store.insert(chunk)
        
        total_chunks += len(chunks)
        print(f"    → {len(chunks)} chunks")
    
    print(f"\nSeeded {total_chunks} chunks total.")
    print(f"Store contains {store.count()} chunks.")


if __name__ == "__main__":
    seed()
