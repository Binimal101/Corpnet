"""Knowledge graph construction from DocumentChunks.

The KG is built FROM chunks after ingestion, not during it.
Nodes are entities, edges are relations.
Each node stores which chunks it appears in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import DocumentChunk, Entity

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph representing an entity."""
    name: str
    chunk_ids: list[str] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)
    embedding: list[float] | None = None
    
    def add_chunk(self, chunk_id: str, labels: set[str] | None = None):
        """Add a chunk reference to this node."""
        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)
        if labels:
            self.labels.update(labels)


@dataclass
class GraphEdge:
    """An edge in the knowledge graph representing a relation."""
    source: str  # Entity name
    target: str  # Entity name
    relation_type: str
    weight: float = 1.0


class KnowledgeGraph:
    """In-memory knowledge graph built from DocumentChunks.
    
    Nodes are entities extracted from chunks.
    Edges are relations between entities.
    """
    
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._edge_set: set[tuple[str, str, str]] = set()  # For dedup
    
    def add_chunk(self, chunk: "DocumentChunk"):
        """Add a chunk's entities and relations to the graph."""
        # Add entity nodes
        for entity_name in chunk.entities:
            self._get_or_create_node(entity_name).add_chunk(chunk.chunk_id, chunk.labels)
        
        # Add relation edges
        for source, rel_type, target in chunk.relations:
            self._add_edge(source, target, rel_type)
    
    def add_node(self, entity: "Entity"):
        """Add an entity directly as a node."""
        if entity.name in self.nodes:
            node = self.nodes[entity.name]
            node.chunk_ids.extend(entity.chunk_ids)
            node.labels.update(entity.labels)
            if entity.embedding:
                node.embedding = entity.embedding
        else:
            self.nodes[entity.name] = GraphNode(
                name=entity.name,
                chunk_ids=list(entity.chunk_ids),
                labels=set(entity.labels),
                embedding=entity.embedding,
            )
    
    def _get_or_create_node(self, name: str) -> GraphNode:
        """Get or create a node by entity name."""
        if name not in self.nodes:
            self.nodes[name] = GraphNode(name=name)
        return self.nodes[name]
    
    def _add_edge(self, source: str, target: str, rel_type: str, weight: float = 1.0):
        """Add an edge, ensuring both endpoint nodes exist."""
        # Ensure nodes exist
        self._get_or_create_node(source)
        self._get_or_create_node(target)
        
        # Deduplicate edges
        edge_key = (source, rel_type, target)
        if edge_key not in self._edge_set:
            self._edge_set.add(edge_key)
            self.edges.append(GraphEdge(
                source=source,
                target=target,
                relation_type=rel_type,
                weight=weight,
            ))
    
    def get_node(self, name: str) -> GraphNode | None:
        """Get a node by name."""
        return self.nodes.get(name)
    
    def get_edges_for(self, node_name: str) -> list[GraphEdge]:
        """Get all edges involving a node."""
        return [e for e in self.edges if e.source == node_name or e.target == node_name]
    
    def node_count(self) -> int:
        """Return the number of nodes."""
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """Return the number of edges."""
        return len(self.edges)
    
    def to_igraph(self):
        """Convert to igraph format for Leiden clustering.
        
        Returns:
            Tuple of (igraph.Graph, list[str]) where the list maps
            vertex indices to entity names.
        """
        try:
            import igraph as ig
        except ImportError:
            raise ImportError("igraph is required for Leiden clustering")
        
        # Build vertex list (entity names)
        vertex_names = list(self.nodes.keys())
        name_to_idx = {name: i for i, name in enumerate(vertex_names)}
        
        # Build edge list
        edges = []
        weights = []
        for edge in self.edges:
            if edge.source in name_to_idx and edge.target in name_to_idx:
                edges.append((name_to_idx[edge.source], name_to_idx[edge.target]))
                weights.append(edge.weight)
        
        # Create graph
        g = ig.Graph(n=len(vertex_names), edges=edges, directed=False)
        g.vs["name"] = vertex_names
        g.es["weight"] = weights if weights else [1.0] * len(edges)
        
        return g, vertex_names
    
    def from_chunks(self, chunks: list["DocumentChunk"]) -> "KnowledgeGraph":
        """Build graph from a list of chunks."""
        for chunk in chunks:
            self.add_chunk(chunk)
        logger.info(f"Built KG with {self.node_count()} nodes, {self.edge_count()} edges")
        return self
