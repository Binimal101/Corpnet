"""Animated tree visualization of the hierarchy flow.

Shows: Input Data -> Level 0 (Entities) -> Communities -> Chunks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class AnimatedTreeVisualizer:
    """Create animated tree visualization of the hierarchy."""
    
    def __init__(
        self,
        figsize: tuple[int, int] = (16, 12),
        style: str = "dark_background",
        colormap: str = "Spectral",
        output_dir: str = "data/visualizations",
        fps: int = 2,
    ):
        """Initialize the animated tree visualizer.
        
        Args:
            figsize: Figure size
            style: Matplotlib style
            colormap: Colormap for nodes
            output_dir: Output directory
            fps: Frames per second for animation
        """
        self._figsize = figsize
        self._style = style
        self._colormap = colormap
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
    
    def _load_hierarchy_with_chunks(
        self,
        db_path: str = "data/archrag.db",
    ) -> dict[str, Any]:
        """Load hierarchy data including chunk relationships."""
        import json
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        
        # Load entities
        cur = conn.execute("SELECT id, name, description, embedding, source_chunk_ids FROM entities")
        entities = cur.fetchall()
        
        entity_data = {}
        for eid, name, desc, emb_json, chunk_ids_json in entities:
            entity_data[eid] = {
                "id": eid,
                "name": name,
                "description": desc,
                "embedding": json.loads(emb_json) if emb_json else None,
                "chunk_ids": json.loads(chunk_ids_json) if chunk_ids_json else [],
            }
        
        # Load chunks
        cur = conn.execute("SELECT id, text, source_doc FROM chunks")
        chunks = cur.fetchall()
        
        chunk_data = {}
        for cid, text, source_doc in chunks:
            chunk_data[cid] = {
                "id": cid,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "source_doc": source_doc,
            }
        
        # Load hierarchy structure
        cur = conn.execute("SELECT value FROM meta WHERE key='hierarchy_structure'")
        row = cur.fetchone()
        
        hierarchy_levels = []
        
        if row:
            structure = json.loads(row[0])
            level_ids = structure.get("level_ids", [])
            
            for level_idx, comm_ids in enumerate(level_ids):
                cur = conn.execute(
                    "SELECT id, level, member_ids, summary, embedding FROM communities WHERE level=?",
                    (level_idx,)
                )
                communities = cur.fetchall()
                
                level_communities = []
                for cid, level, member_ids_json, summary, emb_json in communities:
                    member_ids = json.loads(member_ids_json)
                    level_communities.append({
                        "id": cid,
                        "level": level,
                        "member_ids": member_ids,
                        "summary": summary[:60] if summary else f"Community {cid[:8]}",
                        "embedding": json.loads(emb_json) if emb_json else None,
                    })
                
                hierarchy_levels.append(level_communities)
        
        conn.close()
        
        return {
            "entities": entity_data,
            "chunks": chunk_data,
            "hierarchy_levels": hierarchy_levels,
        }
    
    def _build_tree_structure(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build tree structure for visualization.
        
        Returns:
            {
                "root": {"type": "input", "children": [...]},
                "levels": [
                    {"type": "entities", "nodes": [...]},
                    {"type": "communities", "level": 0, "nodes": [...]},
                    ...
                ],
                "chunks": {"type": "chunks", "nodes": [...]}
            }
        """
        entities = data["entities"]
        chunks = data["chunks"]
        hierarchy_levels = data["hierarchy_levels"]
        
        # Build entity to chunk mapping
        entity_chunks = {}
        for eid, entity in entities.items():
            entity_chunks[eid] = entity.get("chunk_ids", [])
        
        # Build community to entity mapping
        community_entities = {}
        if hierarchy_levels:
            # Level 0 communities contain entities
            for comm in hierarchy_levels[0]:
                community_entities[comm["id"]] = comm["member_ids"]
        
        # Build tree
        tree = {
            "root": {
                "type": "input",
                "label": "Input Data",
                "children": [],
            },
            "levels": [],
        }
        
        # Level 0: Entities
        entity_nodes = []
        for eid, entity in entities.items():
            entity_nodes.append({
                "id": eid,
                "label": entity["name"],
                "type": "entity",
                "chunk_ids": entity.get("chunk_ids", []),
            })
        tree["levels"].append({
            "type": "entities",
            "level": 0,
            "nodes": entity_nodes,
        })
        
        # Community levels
        for level_idx, level_comms in enumerate(hierarchy_levels):
            comm_nodes = []
            for comm in level_comms:
                comm_nodes.append({
                    "id": comm["id"],
                    "label": comm["summary"],
                    "type": "community",
                    "level": comm["level"],
                    "member_ids": comm["member_ids"],
                })
            tree["levels"].append({
                "type": "communities",
                "level": level_idx,
                "nodes": comm_nodes,
            })
        
        # Chunks (leaf nodes)
        chunk_nodes = []
        for cid, chunk in chunks.items():
            chunk_nodes.append({
                "id": cid,
                "label": chunk["text"][:30] + "..." if len(chunk["text"]) > 30 else chunk["text"],
                "type": "chunk",
            })
        tree["chunks"] = {
            "type": "chunks",
            "nodes": chunk_nodes,
        }
        
        return tree
    
    def visualize_animated_tree(
        self,
        db_path: str = "data/archrag.db",
        save: bool = True,
        show: bool = False,
    ) -> None:
        """Create animated tree visualization.
        
        Args:
            db_path: Path to database
            save: If True, save animation as GIF/MP4
            show: If True, display animation
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import seaborn as sns
        
        # Load data
        data = self._load_hierarchy_with_chunks(db_path)
        tree = self._build_tree_structure(data)
        
        # Prepare figure
        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=self._figsize)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")
            
            # Color palette
            palette = sns.color_palette(self._colormap, n_colors=20)
            
            # Animation frames
            frames = []
            
            # Frame 0: Root (Input Data)
            frames.append({
                "title": "Input Data",
                "nodes": [{"x": 5, "y": 9, "label": "Input Data", "type": "root", "size": 200}],
                "edges": [],
            })
            
            # Frame 1: Entities appear
            entity_nodes = tree["levels"][0]["nodes"]
            n_entities = len(entity_nodes)
            entity_positions = self._layout_circular(5, 7, n_entities, radius=3.5)
            
            frames.append({
                "title": "Level 0: Entities",
                "nodes": [
                    {
                        "x": pos[0],
                        "y": pos[1],
                        "label": node["label"][:15] + "..." if len(node["label"]) > 15 else node["label"],
                        "type": "entity",
                        "size": 80,
                        "id": node["id"],
                    }
                    for node, pos in zip(entity_nodes[:30], entity_positions[:30])  # Limit to 30 for clarity
                ],
                "edges": [
                    {"from": (5, 9), "to": (pos[0], pos[1])}
                    for pos in entity_positions[:30]
                ],
            })
            
            # Frame 2: Communities appear (level 0)
            if len(tree["levels"]) > 1:
                comm_nodes = tree["levels"][1]["nodes"]
                n_comms = len(comm_nodes)
                comm_positions = self._layout_circular(5, 4, n_comms, radius=2.5)
                
                frames.append({
                    "title": "Level 1: Communities",
                    "nodes": [
                        {
                            "x": pos[0],
                            "y": pos[1],
                            "label": node["label"][:20] + "..." if len(node["label"]) > 20 else node["label"],
                            "type": "community",
                            "size": 100,
                            "id": node["id"],
                        }
                        for node, pos in zip(comm_nodes[:20], comm_positions[:20])
                    ],
                    "edges": [
                        {"from": (5, 7), "to": (pos[0], pos[1])}
                        for pos in comm_positions[:20]
                    ],
                })
            
            # Frame 3: Chunks appear
            chunk_nodes = tree["chunks"]["nodes"]
            n_chunks = min(len(chunk_nodes), 20)  # Limit chunks
            chunk_positions = self._layout_circular(5, 1.5, n_chunks, radius=4)
            
            frames.append({
                "title": "Chunks (Leaf Nodes)",
                "nodes": [
                    {
                        "x": pos[0],
                        "y": pos[1],
                        "label": node["label"][:12] + "..." if len(node["label"]) > 12 else node["label"],
                        "type": "chunk",
                        "size": 50,
                        "id": node["id"],
                    }
                    for node, pos in zip(chunk_nodes[:n_chunks], chunk_positions[:n_chunks])
                ],
                "edges": [
                    {"from": (5, 4), "to": (pos[0], pos[1])}
                    for pos in chunk_positions[:n_chunks]
                ],
            })
            
            # Animation function
            def animate(frame_idx):
                ax.clear()
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.axis("off")
                
                frame = frames[frame_idx % len(frames)]
                
                # Draw title
                ax.text(
                    5, 9.5,
                    frame["title"],
                    ha="center",
                    va="top",
                    fontsize=16,
                    fontweight="bold",
                )
                
                # Draw nodes
                node_colors = {
                    "root": palette[0],
                    "entity": palette[5],
                    "community": palette[10],
                    "chunk": palette[15],
                }
                
                for node in frame["nodes"]:
                    color = node_colors.get(node["type"], palette[0])
                    circle = plt.Circle(
                        (node["x"], node["y"]),
                        node["size"] / 200,
                        color=color,
                        alpha=0.8,
                        ec="white",
                        linewidth=2,
                    )
                    ax.add_patch(circle)
                    
                    # Label
                    ax.text(
                        node["x"], node["y"] - node["size"] / 150,
                        node["label"],
                        ha="center",
                        va="top",
                        fontsize=7,
                        wrap=True,
                    )
                
                # Draw edges
                for edge in frame["edges"]:
                    ax.plot(
                        [edge["from"][0], edge["to"][0]],
                        [edge["from"][1], edge["to"][1]],
                        color="gray",
                        alpha=0.4,
                        linewidth=1,
                        zorder=0,
                    )
            
            # Create animation
            anim = animation.FuncAnimation(
                fig, animate,
                frames=len(frames),
                interval=1000 / self._fps,
                repeat=True,
            )
            
            if save:
                # Save as GIF
                gif_path = self._output_dir / "hierarchy_animated_tree.gif"
                try:
                    anim.save(
                        str(gif_path),
                        writer="pillow",
                        fps=self._fps,
                    )
                    log.info("Saved animated tree to %s", gif_path)
                except Exception as e:
                    log.warning("Failed to save GIF: %s", e)
                    # Fallback: save as static frames
                    for i, frame in enumerate(frames):
                        animate(i)
                        static_path = self._output_dir / f"tree_frame_{i}.png"
                        fig.savefig(static_path, dpi=150, bbox_inches="tight")
                    log.info("Saved static frames instead")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def _layout_circular(
        self,
        center_x: float,
        center_y: float,
        n_nodes: int,
        radius: float,
    ) -> list[tuple[float, float]]:
        """Layout nodes in a circle."""
        if n_nodes == 0:
            return []
        if n_nodes == 1:
            return [(center_x, center_y)]
        
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = []
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions.append((x, y))
        
        return positions
    
    def visualize_static_tree(
        self,
        db_path: str = "data/archrag.db",
        save: bool = True,
        show: bool = True,
    ) -> None:
        """Create a static tree visualization (all levels at once).
        
        Args:
            db_path: Path to database
            save: If True, save figure
            show: If True, display figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        import seaborn as sns
        
        # Load data
        data = self._load_hierarchy_with_chunks(db_path)
        tree = self._build_tree_structure(data)
        
        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=self._figsize)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")
            
            palette = sns.color_palette(self._colormap, n_colors=20)
            
            y_levels = [9, 7, 5, 3, 1.5]  # Y positions for each level
            
            # Root
            ax.add_patch(plt.Circle((5, y_levels[0]), 0.3, color=palette[0], alpha=0.8, ec="white", linewidth=2))
            ax.text(5, y_levels[0], "Input Data", ha="center", va="center", fontsize=10, fontweight="bold")
            
            # Entities (Level 0)
            entity_nodes = tree["levels"][0]["nodes"][:25]  # Limit for clarity
            n_entities = len(entity_nodes)
            entity_x = np.linspace(1, 9, n_entities) if n_entities > 1 else [5]
            
            for i, (node, x) in enumerate(zip(entity_nodes, entity_x)):
                ax.add_patch(plt.Circle((x, y_levels[1]), 0.15, color=palette[5], alpha=0.7, ec="white", linewidth=1))
                ax.plot([5, x], [y_levels[0] - 0.3, y_levels[1] + 0.15], color="gray", alpha=0.3, linewidth=0.5)
                label = node["label"][:10] + "..." if len(node["label"]) > 10 else node["label"]
                ax.text(x, y_levels[1] - 0.25, label, ha="center", va="top", fontsize=5, rotation=45)
            
            # Communities
            if len(tree["levels"]) > 1:
                comm_nodes = tree["levels"][1]["nodes"][:15]
                n_comms = len(comm_nodes)
                comm_x = np.linspace(1.5, 8.5, n_comms) if n_comms > 1 else [5]
                
                for i, (node, x) in enumerate(zip(comm_nodes, comm_x)):
                    ax.add_patch(plt.Circle((x, y_levels[2]), 0.2, color=palette[10], alpha=0.7, ec="white", linewidth=1))
                    # Connect to entities (simplified)
                    ax.plot([entity_x[i % len(entity_x)], x], [y_levels[1] - 0.15, y_levels[2] + 0.2], 
                           color="gray", alpha=0.2, linewidth=0.5)
                    label = node["label"][:12] + "..." if len(node["label"]) > 12 else node["label"]
                    ax.text(x, y_levels[2] - 0.3, label, ha="center", va="top", fontsize=5)
            
            # Chunks
            chunk_nodes = tree["chunks"]["nodes"][:20]
            n_chunks = len(chunk_nodes)
            chunk_x = np.linspace(1, 9, n_chunks) if n_chunks > 1 else [5]
            
            for i, (node, x) in enumerate(zip(chunk_nodes, chunk_x)):
                ax.add_patch(plt.Circle((x, y_levels[4]), 0.1, color=palette[15], alpha=0.6, ec="white", linewidth=1))
                # Connect to entities
                entity_idx = i % len(entity_x)
                ax.plot([entity_x[entity_idx], x], [y_levels[1] - 0.15, y_levels[4] + 0.1],
                       color="gray", alpha=0.2, linewidth=0.3)
            
            ax.set_title(
                "ArchRAG Hierarchy Tree\nInput Data → Entities → Communities → Chunks",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / "hierarchy_static_tree.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved static tree to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
