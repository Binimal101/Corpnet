"""Dendrogram visualization for hierarchical clustering.

Creates a tree-like visualization showing the nested structure
of communities across hierarchy levels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class DendrogramVisualizer:
    """Visualizes the hierarchy as a dendrogram (tree structure)."""
    
    def __init__(
        self,
        figsize: tuple[int, int] = (16, 12),
        style: str = "dark_background",
        colormap: str = "Spectral",
        output_dir: str = "data/visualizations",
    ):
        self._figsize = figsize
        self._style = style
        self._colormap = colormap
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_dendrogram(
        self,
        hierarchy_data: list[dict[str, Any]],
        save: bool = True,
        show: bool = True,
    ) -> None:
        """Create a dendrogram showing the hierarchy structure.
        
        Args:
            hierarchy_data: List of level data from load_hierarchy_from_db
            save: If True, save the figure
            show: If True, display the figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import FancyBboxPatch, ConnectionPatch
        import matplotlib.colors as mcolors
        
        if not hierarchy_data:
            log.warning("No hierarchy data to visualize")
            return
        
        n_levels = len(hierarchy_data)
        
        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=self._figsize)
            
            # Calculate positions
            level_height = 0.8 / n_levels
            y_positions = {}  # node_id -> (x, y)
            
            # Color palette for clusters
            all_clusters = sum(
                len(set(d["cluster_assignments"])) for d in hierarchy_data
            )
            palette = sns.color_palette(self._colormap, n_colors=max(all_clusters, 1))
            color_idx = 0
            
            for level in range(n_levels - 1, -1, -1):
                data = hierarchy_data[level]
                node_ids = data["node_ids"]
                labels = data["labels"]
                cluster_assignments = data["cluster_assignments"]
                
                n_nodes = len(node_ids)
                if n_nodes == 0:
                    continue
                
                # Y position for this level (top to bottom)
                y = 1.0 - (level + 0.5) * level_height
                
                # X positions spread across the width
                x_positions = np.linspace(0.05, 0.95, n_nodes)
                
                # Group by cluster for coloring
                unique_clusters = sorted(set(cluster_assignments))
                cluster_colors = {}
                for c in unique_clusters:
                    cluster_colors[c] = palette[color_idx % len(palette)]
                    color_idx += 1
                
                # Draw nodes
                for i, (nid, label, cluster) in enumerate(zip(node_ids, labels, cluster_assignments)):
                    x = x_positions[i]
                    y_positions[nid] = (x, y)
                    
                    color = cluster_colors[cluster]
                    
                    # Draw node circle
                    circle = plt.Circle(
                        (x, y), 0.015,
                        color=color,
                        ec="white",
                        linewidth=1,
                        zorder=3,
                    )
                    ax.add_patch(circle)
                    
                    # Add label (truncated)
                    display_label = label[:20] + "..." if len(label) > 20 else label
                    ax.text(
                        x, y - 0.03,
                        display_label,
                        fontsize=6,
                        ha="center",
                        va="top",
                        rotation=45,
                        alpha=0.8,
                    )
                
                # Draw level label
                level_name = "Entities" if level == 0 else f"Level {level}"
                ax.text(
                    0.01, y,
                    f"{level_name}\n({n_nodes} nodes)",
                    fontsize=9,
                    fontweight="bold",
                    va="center",
                )
            
            # Draw connections between levels
            for level in range(1, n_levels):
                data = hierarchy_data[level]
                node_ids = data["node_ids"]
                
                # Get parent level data
                if level + 1 < n_levels:
                    parent_data = hierarchy_data[level + 1]
                    parent_cluster_assignments = parent_data["cluster_assignments"]
                    parent_node_ids = parent_data["node_ids"]
                    
                    # Draw connections based on cluster assignments
                    for i, cluster_id in enumerate(parent_cluster_assignments):
                        if cluster_id < len(node_ids):
                            child_id = node_ids[cluster_id] if cluster_id < len(node_ids) else None
                            parent_id = parent_node_ids[i]
                            
                            if child_id in y_positions and parent_id in y_positions:
                                x1, y1 = y_positions[child_id]
                                x2, y2 = y_positions[parent_id]
                                
                                ax.plot(
                                    [x1, x2], [y1, y2],
                                    color="gray",
                                    alpha=0.3,
                                    linewidth=0.5,
                                    zorder=1,
                                )
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect("equal")
            ax.axis("off")
            
            ax.set_title(
                "ArchRAG Community Hierarchy\n"
                f"{n_levels} levels, {sum(len(d['node_ids']) for d in hierarchy_data)} total nodes",
                fontsize=14,
                fontweight="bold",
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / "hierarchy_dendrogram.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved dendrogram to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def visualize_sunburst(
        self,
        hierarchy_data: list[dict[str, Any]],
        save: bool = True,
        show: bool = True,
    ) -> None:
        """Create a sunburst chart showing the hierarchy.
        
        Args:
            hierarchy_data: List of level data
            save: If True, save the figure
            show: If True, display the figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Wedge
        
        if not hierarchy_data:
            log.warning("No hierarchy data to visualize")
            return
        
        n_levels = len(hierarchy_data)
        
        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=self._figsize, subplot_kw={"projection": "polar"})
            
            # Each level is a ring
            ring_width = 0.8 / n_levels
            
            for level in range(n_levels):
                data = hierarchy_data[level]
                cluster_assignments = data["cluster_assignments"]
                labels = data["labels"]
                
                unique_clusters = sorted(set(cluster_assignments))
                n_clusters = len(unique_clusters)
                
                if n_clusters == 0:
                    continue
                
                # Color palette
                palette = sns.color_palette(self._colormap, n_colors=n_clusters)
                
                # Calculate cluster sizes
                cluster_sizes = []
                for c in unique_clusters:
                    size = sum(1 for ca in cluster_assignments if ca == c)
                    cluster_sizes.append(size)
                
                total = sum(cluster_sizes)
                
                # Draw wedges
                inner_radius = level * ring_width + 0.1
                outer_radius = inner_radius + ring_width * 0.9
                
                start_angle = 0
                for i, (cluster_id, size) in enumerate(zip(unique_clusters, cluster_sizes)):
                    angle_span = 2 * np.pi * size / total
                    
                    theta = np.linspace(start_angle, start_angle + angle_span, 50)
                    
                    # Draw the wedge
                    ax.fill_between(
                        theta,
                        inner_radius,
                        outer_radius,
                        color=palette[i],
                        alpha=0.7,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    
                    # Add label at center of wedge
                    mid_angle = start_angle + angle_span / 2
                    mid_radius = (inner_radius + outer_radius) / 2
                    
                    if angle_span > 0.15:  # Only label if big enough
                        cluster_labels = [
                            labels[j][:15] for j, ca in enumerate(cluster_assignments) 
                            if ca == cluster_id
                        ]
                        display_text = f"{size}" if size > 1 else cluster_labels[0][:10]
                        ax.text(
                            mid_angle, mid_radius,
                            display_text,
                            fontsize=7,
                            ha="center",
                            va="center",
                        )
                    
                    start_angle += angle_span
            
            ax.set_ylim(0, 1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            
            # Add level labels
            for level in range(n_levels):
                inner_radius = level * ring_width + 0.1
                level_name = "Entities" if level == 0 else f"L{level}"
                ax.text(
                    0, inner_radius + ring_width / 2,
                    level_name,
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
            
            ax.set_title(
                "ArchRAG Hierarchy Sunburst",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            
            if save:
                out_path = self._output_dir / "hierarchy_sunburst.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved sunburst to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def visualize_treemap(
        self,
        hierarchy_data: list[dict[str, Any]],
        level: int = 0,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """Create a treemap visualization for a specific level.
        
        Args:
            hierarchy_data: List of level data
            level: Which level to visualize as treemap
            save: If True, save the figure
            show: If True, display the figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        try:
            import squarify
        except ImportError:
            log.warning("squarify not installed, using basic rectangles")
            squarify = None
        
        if level >= len(hierarchy_data):
            log.warning("Level %d not found in hierarchy", level)
            return
        
        data = hierarchy_data[level]
        cluster_assignments = data["cluster_assignments"]
        labels = data["labels"]
        
        unique_clusters = sorted(set(cluster_assignments))
        n_clusters = len(unique_clusters)
        
        # Calculate cluster sizes and labels
        sizes = []
        cluster_labels = []
        for c in unique_clusters:
            members = [labels[i] for i, ca in enumerate(cluster_assignments) if ca == c]
            sizes.append(len(members))
            # Use first few member names as label
            label = ", ".join(members[:3])
            if len(label) > 40:
                label = label[:37] + "..."
            if len(members) > 3:
                label += f" (+{len(members) - 3})"
            cluster_labels.append(label)
        
        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=self._figsize)
            
            palette = sns.color_palette(self._colormap, n_colors=n_clusters)
            
            if squarify is not None:
                squarify.plot(
                    sizes=sizes,
                    label=cluster_labels,
                    color=palette,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=2,
                    text_kwargs={"fontsize": 8, "wrap": True},
                    ax=ax,
                )
            else:
                # Simple grid layout fallback
                n_cols = int(np.ceil(np.sqrt(n_clusters)))
                n_rows = int(np.ceil(n_clusters / n_cols))
                
                for i, (size, label) in enumerate(zip(sizes, cluster_labels)):
                    row = i // n_cols
                    col = i % n_cols
                    
                    rect = plt.Rectangle(
                        (col / n_cols, 1 - (row + 1) / n_rows),
                        1 / n_cols * 0.95,
                        1 / n_rows * 0.95,
                        color=palette[i],
                        alpha=0.7,
                        ec="white",
                        linewidth=2,
                    )
                    ax.add_patch(rect)
                    
                    ax.text(
                        col / n_cols + 0.5 / n_cols,
                        1 - row / n_rows - 0.5 / n_rows,
                        f"{label}\n({size})",
                        fontsize=7,
                        ha="center",
                        va="center",
                        wrap=True,
                    )
            
            ax.axis("off")
            
            level_name = "Entities" if level == 0 else f"Communities Level {level}"
            ax.set_title(
                f"ArchRAG Treemap — {level_name}\n"
                f"{sum(sizes)} nodes in {n_clusters} clusters",
                fontsize=14,
                fontweight="bold",
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / f"hierarchy_treemap_level_{level}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved treemap to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
