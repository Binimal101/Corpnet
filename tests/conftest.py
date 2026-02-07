"""Shared test fixtures for DAC-HRAG tests."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.core.config import reset_settings
from src.core.embeddings import MockEmbedding
from src.core.llm import MockLLM
from src.core.types import DocumentChunk, Community, SearchResult
from src.indexing.vector_store import InMemoryVectorStore


@pytest.fixture(autouse=True)
def reset_config():
    """Reset settings before each test."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def embedder() -> MockEmbedding:
    """Mock embedding provider for tests."""
    return MockEmbedding(dim=768)


@pytest.fixture
def llm() -> MockLLM:
    """Mock LLM provider for tests."""
    return MockLLM()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """In-memory vector store for tests."""
    return InMemoryVectorStore(dimension=768)


@pytest.fixture
def sample_embedding() -> list[float]:
    """A sample 768-dimensional embedding."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(768)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def sample_chunk(sample_embedding: list[float]) -> DocumentChunk:
    """A sample document chunk."""
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        doc_id="test-doc",
        text="This is a test chunk about Python programming.",
        embedding=sample_embedding,
        labels=set(),
        metadata={"source": "test"},
        entities=["Python", "programming"],
        relations=[("Python", "is_a", "programming language")],
    )


@pytest.fixture
def sample_chunks(embedder: MockEmbedding) -> list[DocumentChunk]:
    """Multiple sample chunks for testing."""
    texts = [
        "Python is a programming language used for web development.",
        "FastAPI is a modern web framework for Python.",
        "PostgreSQL is a powerful relational database.",
        "Docker containers provide isolated environments.",
        "Kubernetes orchestrates container deployments.",
    ]
    
    chunks = []
    for i, text in enumerate(texts):
        embedding = embedder.embed_text(text)
        chunks.append(DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=f"doc-{i}",
            text=text,
            embedding=embedding,
            labels=set(),
            metadata={"index": i},
            entities=[],
            relations=[],
        ))
    
    return chunks


@pytest.fixture
def populated_store(
    vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
) -> InMemoryVectorStore:
    """Vector store pre-populated with sample chunks."""
    for chunk in sample_chunks:
        vector_store.insert(chunk)
    return vector_store


@pytest.fixture
def sample_community(sample_embedding: list[float]) -> Community:
    """A sample community."""
    return Community(
        community_id=str(uuid.uuid4()),
        level=0,
        summary="A community about Python and web development.",
        summary_embedding=sample_embedding,
        labels=set(),
        member_ids=["entity-1", "entity-2"],
        parent_id=None,
        children_ids=[],
        peer_id="peer-001",
    )
