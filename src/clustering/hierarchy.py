"""Community hierarchy data structure.

Organizes communities into levels from fine (leaf) to coarse (root).
Provides bidirectional navigation (parent ↔ children).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.core.types import Community

logger = logging.getLogger(__name__)


@dataclass
class CommunityHierarchy:
    """Hierarchical community structure with bidirectional links.
    
    Level 0 = leaf entities
    Higher levels = progressively coarser community groupings
    """
    
    levels: list[list[Community]] = field(default_factory=list)
    _community_index: dict[str, Community] = field(default_factory=dict)
    
    @property
    def top_level(self) -> int:
        """Return the index of the top (coarsest) level."""
        return len(self.levels) - 1 if self.levels else 0
    
    @property
    def num_levels(self) -> int:
        """Return the number of levels."""
        return len(self.levels)
    
    def add_level(self, communities: list[Community]) -> None:
        """Add a new level of communities."""
        self.levels.append(communities)
        for community in communities:
            self._community_index[community.community_id] = community
    
    def get_level(self, n: int) -> list[Community]:
        """Get all communities at level n."""
        if 0 <= n < len(self.levels):
            return self.levels[n]
        return []
    
    def get_community(self, community_id: str) -> Community | None:
        """Get a community by ID."""
        return self._community_index.get(community_id)
    
    def get_children(self, community_id: str) -> list[Community]:
        """Get child communities for a community."""
        community = self._community_index.get(community_id)
        if not community:
            return []
        
        children = []
        for child_id in community.children_ids:
            child = self._community_index.get(child_id)
            if child:
                children.append(child)
        return children
    
    def get_parent(self, community_id: str) -> Community | None:
        """Get the parent community."""
        community = self._community_index.get(community_id)
        if community and community.parent_id:
            return self._community_index.get(community.parent_id)
        return None
    
    def set_parent(self, child_id: str, parent_id: str) -> None:
        """Set parent-child relationship."""
        child = self._community_index.get(child_id)
        parent = self._community_index.get(parent_id)
        
        if child:
            child.parent_id = parent_id
        if parent and child_id not in parent.children_ids:
            parent.children_ids.append(child_id)
    
    def get_path_to_root(self, community_id: str) -> list[str]:
        """Get the path from a community to the root."""
        path = [community_id]
        current = self._community_index.get(community_id)
        
        while current and current.parent_id:
            path.append(current.parent_id)
            current = self._community_index.get(current.parent_id)
        
        return path
    
    def total_communities(self) -> int:
        """Return total number of communities across all levels."""
        return sum(len(level) for level in self.levels)
    
    def to_dict(self) -> dict:
        """Serialize hierarchy for storage/transport."""
        return {
            "levels": [
                [c.to_dict() for c in level]
                for level in self.levels
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CommunityHierarchy":
        """Deserialize hierarchy."""
        hierarchy = cls()
        for level_data in d.get("levels", []):
            communities = [Community.from_dict(c) for c in level_data]
            hierarchy.add_level(communities)
        return hierarchy
