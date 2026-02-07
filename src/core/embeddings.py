"""Embedding provider implementations.

Supports:
- sentence-transformers (local, default)
- openai (API-based)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.config import EmbeddingsConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
    
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using sentence-transformers (local inference)."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", dim: int = 768):
        self.model_name = model_name
        self._dimension = dim
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using random embeddings")
                self._model = "mock"
        return self._model
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> list[float]:
        model = self._get_model()
        if model == "mock":
            # Deterministic mock based on text hash
            rng = np.random.default_rng(hash(text) % (2**32))
            vec = rng.standard_normal(self._dimension)
            vec = vec / np.linalg.norm(vec)
            return vec.tolist()
        return model.encode(text, normalize_embeddings=True).tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        if model == "mock":
            return [self.embed_text(t) for t in texts]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI API."""
    
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        self.model = model
        self._dimension = dim
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI()
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class MockEmbedding(EmbeddingProvider):
    """Mock embedding provider for testing (deterministic random vectors)."""
    
    def __init__(self, dim: int = 768):
        self._dimension = dim
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> list[float]:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dimension)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def create_embedding_provider(config: "EmbeddingsConfig") -> EmbeddingProvider:
    """Factory function to create the appropriate embedding provider."""
    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbedding(
            model_name=config.model,
            dim=config.dimension,
        )
    elif config.provider == "openai":
        return OpenAIEmbedding(
            model=config.model,
            dim=config.dimension,
        )
    elif config.provider == "mock":
        return MockEmbedding(dim=config.dimension)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
