"""k-NN graph augmentation for clustering.

Adds k-NN edges based on embedding similarity to densify the graph
before community detection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.clustering.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def knn_augment(
    graph: "KnowledgeGraph",
    embeddings: dict[str, list[float]],
    k: int = 10,
    threshold: float = 0.5,
) -> "KnowledgeGraph":
    """Augment graph with k-NN edges based on embedding similarity.
    
    Args:
        graph: The knowledge graph to augment.
        embeddings: Mapping from entity name to embedding vector.
        k: Number of nearest neighbors to consider.
        threshold: Minimum similarity threshold for adding edges.
    
    Returns:
        The augmented graph (modified in-place).
    """
    # Get entities with embeddings
    entities_with_emb = [(name, embeddings[name]) for name in graph.nodes if name in embeddings]
    
    if len(entities_with_emb) < 2:
        logger.warning("Not enough entities with embeddings for k-NN augmentation")
        return graph
    
    names = [e[0] for e in entities_with_emb]
    vectors = np.array([e[1] for e in entities_with_emb])
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    
    # Compute similarity matrix
    similarity = vectors @ vectors.T
    
    edges_added = 0
    for i, name_i in enumerate(names):
        # Get top-k similar entities
        sims = similarity[i]
        top_indices = np.argsort(sims)[::-1][1:k+1]  # Exclude self
        
        for j in top_indices:
            if sims[j] >= threshold:
                name_j = names[j]
                # Add edge if not already present
                if not any(
                    (e.source == name_i and e.target == name_j) or
                    (e.source == name_j and e.target == name_i)
                    for e in graph.edges
                ):
                    graph._add_edge(name_i, name_j, "knn_similar", weight=float(sims[j]))
                    edges_added += 1
    
    logger.info(f"Added {edges_added} k-NN edges")
    return graph


def compute_entity_embeddings(
    graph: "KnowledgeGraph",
    chunk_embeddings: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Compute entity embeddings by averaging chunk embeddings.
    
    Args:
        graph: The knowledge graph.
        chunk_embeddings: Mapping from chunk_id to embedding.
    
    Returns:
        Mapping from entity name to embedding.
    """
    entity_embeddings = {}
    
    for name, node in graph.nodes.items():
        # Get embeddings of chunks this entity appears in
        embs = [chunk_embeddings[cid] for cid in node.chunk_ids if cid in chunk_embeddings]
        
        if embs:
            # Average the embeddings
            avg = np.mean(embs, axis=0)
            # Normalize
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            entity_embeddings[name] = avg.tolist()
    
    logger.info(f"Computed embeddings for {len(entity_embeddings)} entities")
    return entity_embeddings
