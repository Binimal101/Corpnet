"""Orchestrator: wires offline indexing and online retrieval pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from archrag.domain.models import CHNSWIndex, CommunityHierarchy, KnowledgeGraph
from archrag.ports.clustering import ClusteringPort
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.llm import LLMPort
from archrag.ports.vector_index import VectorIndexPort
from archrag.services.adaptive_filtering import AdaptiveFilteringService
from archrag.services.chnsw_build import CHNSWBuildService
from archrag.services.hierarchical_clustering import HierarchicalClusteringService
from archrag.services.hierarchical_search import HierarchicalSearchService
from archrag.services.kg_construction import KGConstructionService

log = logging.getLogger(__name__)


class ArchRAGOrchestrator:
    """Top-level entry point for the ArchRAG pipeline."""

    def __init__(
        self,
        llm: LLMPort,
        embedding: EmbeddingPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        vector_index: VectorIndexPort,
        clustering: ClusteringPort,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        max_levels: int = 5,
        similarity_threshold: float = 0.7,
        M: int = 32,
        ef_construction: int = 100,
        k_per_layer: int = 5,
    ):
        self._llm = llm
        self._embedding = embedding
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._vector_index = vector_index
        self._clustering = clustering

        # Sub-services
        self._kg_service = KGConstructionService(
            llm, embedding, graph_store, doc_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._cluster_service = HierarchicalClusteringService(
            llm, embedding, clustering, graph_store, doc_store,
            max_levels=max_levels,
            similarity_threshold=similarity_threshold,
        )
        self._chnsw_service = CHNSWBuildService(
            embedding, vector_index, graph_store, doc_store,
            M=M,
            ef_construction=ef_construction,
        )
        self._search_service = HierarchicalSearchService(
            embedding, vector_index, graph_store, doc_store,
            k_per_layer=k_per_layer,
        )
        self._filter_service = AdaptiveFilteringService(llm)

        self._index: CHNSWIndex | None = None

    # ── Offline indexing ──

    def index(self, corpus_path: str) -> None:
        """Run the full offline indexing pipeline.

        1. Load corpus (JSONL)
        2. KG construction
        3. Hierarchical clustering
        4. C-HNSW build
        """
        log.info("Starting offline indexing from %s", corpus_path)

        # Load documents
        documents = self._load_corpus(corpus_path)
        log.info("Loaded %d documents", len(documents))

        # Phase 1: KG construction
        kg = self._kg_service.build(documents)

        # Phase 2: Hierarchical clustering
        hierarchy = self._cluster_service.build(kg)

        # Phase 3: C-HNSW construction
        self._index = self._chnsw_service.build(hierarchy)

        log.info("Offline indexing complete.")

    # ── Online retrieval ──

    def query(self, question: str) -> str:
        """Answer a question using hierarchical search + adaptive filtering."""
        log.info("Query: %s", question)

        # Hierarchical search
        results = self._search_service.search(question, self._index)

        total = sum(len(layer) for layer in results)
        log.info("Retrieved %d results across %d layers", total, len(results))

        # Adaptive filtering & answer generation
        answer = self._filter_service.generate_answer(question, results)

        return answer

    # ── helpers ──

    @staticmethod
    def _load_corpus(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        documents: list[dict[str, Any]] = []
        text = p.read_text(encoding="utf-8")

        # Try JSONL (one JSON object per line)
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError:
                pass

        # If JSONL failed, try as a single JSON array
        if not documents:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse corpus: {exc}") from exc

        return documents
