"""Leiden community detection for knowledge graph clustering.

Implements iterative Leiden clustering with per-level resolution schedule.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.clustering.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def leiden_cluster(
    graph: "KnowledgeGraph",
    resolution: float = 1.0,
) -> list[list[str]]:
    """Run Leiden community detection on a knowledge graph.
    
    Args:
        graph: The knowledge graph to cluster.
        resolution: Leiden resolution parameter. Higher = more, smaller communities.
    
    Returns:
        List of communities, where each community is a list of entity names.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError("igraph and leidenalg are required for Leiden clustering")
    
    if graph.node_count() < 2:
        # Return all nodes as one community
        return [list(graph.nodes.keys())]
    
    # Convert to igraph
    ig_graph, vertex_names = graph.to_igraph()
    
    if ig_graph.ecount() == 0:
        # No edges - each node is its own community
        return [[name] for name in vertex_names]
    
    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
    )
    
    # Convert partition to list of communities
    communities = []
    for community_idx in range(len(partition)):
        members = [vertex_names[v] for v in partition[community_idx]]
        if members:
            communities.append(members)
    
    logger.info(f"Leiden found {len(communities)} communities at resolution {resolution}")
    return communities


def hierarchical_cluster(
    graph: "KnowledgeGraph",
    resolution_schedule: list[float] | None = None,
    min_community_size: int = 2,
    max_levels: int = 5,
) -> list[list[list[str]]]:
    """Build a hierarchical clustering using Leiden with decreasing resolution.
    
    Args:
        graph: The knowledge graph to cluster.
        resolution_schedule: List of resolution values for each level.
                            Default: [2.0, 1.0, 0.5] (fine to coarse).
        min_community_size: Minimum entities to form a community.
        max_levels: Maximum hierarchy depth.
    
    Returns:
        List of levels, where each level is a list of communities,
        and each community is a list of entity names.
    """
    if resolution_schedule is None:
        resolution_schedule = [2.0, 1.0, 0.5]
    
    levels = []
    current_entities = set(graph.nodes.keys())
    
    for level_idx, resolution in enumerate(resolution_schedule):
        if level_idx >= max_levels:
            break
        
        if len(current_entities) < min_community_size:
            logger.info(f"Stopping at level {level_idx}: too few entities")
            break
        
        # Build subgraph for current entities
        subgraph = _build_subgraph(graph, current_entities)
        
        # Cluster at this level
        communities = leiden_cluster(subgraph, resolution=resolution)
        
        # Filter by minimum size
        communities = [c for c in communities if len(c) >= min_community_size]
        
        if not communities:
            logger.info(f"Stopping at level {level_idx}: no valid communities")
            break
        
        levels.append(communities)
        logger.info(f"Level {level_idx}: {len(communities)} communities")
        
        # For next level, entities are the community centroids
        # (in practice, we'd create synthetic nodes for communities)
        if len(communities) <= 1:
            logger.info(f"Stopping at level {level_idx}: converged to single community")
            break
    
    return levels


def _build_subgraph(graph: "KnowledgeGraph", entity_names: set[str]) -> "KnowledgeGraph":
    """Build a subgraph containing only the specified entities."""
    from src.clustering.knowledge_graph import KnowledgeGraph, GraphNode
    
    subgraph = KnowledgeGraph()
    
    # Copy nodes
    for name in entity_names:
        if name in graph.nodes:
            node = graph.nodes[name]
            subgraph.nodes[name] = GraphNode(
                name=name,
                chunk_ids=list(node.chunk_ids),
                labels=set(node.labels),
                embedding=node.embedding,
            )
    
    # Copy edges where both endpoints are in the subgraph
    for edge in graph.edges:
        if edge.source in entity_names and edge.target in entity_names:
            subgraph._add_edge(edge.source, edge.target, edge.relation_type, edge.weight)
    
    return subgraph
