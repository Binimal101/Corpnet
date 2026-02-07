"""Hierarchy visualization with reverse embedding projection.

Visualizes the hierarchical clustering in real-time by projecting 
high-dimensional embeddings to 2D using UMAP or t-SNE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VisualizationState:
    """Holds the current state of the visualization."""
    
    level: int
    node_ids: list[str]
    labels: list[str]
    embeddings_2d: np.ndarray
    cluster_assignments: list[int]
    community_labels: list[str]


class HierarchyVisualizer:
    """Visualizes hierarchical clustering with reverse embedding projection.
    
    Supports real-time updates during clustering and static visualization
    of existing hierarchies.
    """
    
    def __init__(
        self,
        method: Literal["umap", "tsne", "pca"] = "umap",
        figsize: tuple[int, int] = (14, 10),
        style: str = "dark_background",
        colormap: str = "Spectral",
        interactive: bool = True,
        output_dir: str = "data/visualizations",
    ):
        """Initialize the visualizer.
        
        Args:
            method: Dimensionality reduction method ('umap', 'tsne', 'pca')
            figsize: Figure size as (width, height)
            style: Matplotlib style to use
            colormap: Colormap for cluster colors
            interactive: If True, use interactive mode for real-time updates
            output_dir: Directory to save visualization outputs
        """
        self._method = method
        self._figsize = figsize
        self._style = style
        self._colormap = colormap
        self._interactive = interactive
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self._fig = None
        self._axes = None
        self._reducer = None
        self._states: list[VisualizationState] = []
        
    def _ensure_imports(self) -> None:
        """Lazy import visualization dependencies."""
        global plt, sns, UMAP, TSNE, PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self._method == "umap":
            try:
                from umap import UMAP
            except ImportError:
                log.warning("UMAP not available, falling back to t-SNE")
                self._method = "tsne"
        
        if self._method == "tsne":
            from sklearn.manifold import TSNE
        
        if self._method == "pca":
            from sklearn.decomposition import PCA
    
    def _create_reducer(self, n_components: int = 2) -> Any:
        """Create a dimensionality reduction model."""
        if self._method == "umap":
            from umap import UMAP
            return UMAP(
                n_components=n_components,
                n_neighbors=min(15, max(2, n_components * 2)),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
        elif self._method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(
                n_components=n_components,
                perplexity=min(30, max(5, n_components)),
                random_state=42,
                max_iter=1000,
            )
        else:  # pca
            from sklearn.decomposition import PCA
            return PCA(n_components=n_components, random_state=42)
    
    def project_embeddings(
        self,
        embeddings: np.ndarray,
        fit: bool = True,
    ) -> np.ndarray:
        """Project high-dimensional embeddings to 2D.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            fit: If True, fit a new reducer; otherwise use existing
            
        Returns:
            Array of shape (n_samples, 2)
        """
        self._ensure_imports()
        
        if len(embeddings) < 2:
            # Can't reduce with less than 2 points
            if len(embeddings) == 1:
                return np.array([[0.0, 0.0]])
            return np.array([])
        
        # Adjust perplexity/neighbors for small datasets
        n_samples = len(embeddings)
        
        if fit or self._reducer is None:
            if self._method == "umap":
                from umap import UMAP
                n_neighbors = min(15, max(2, n_samples - 1))
                self._reducer = UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
            elif self._method == "tsne":
                from sklearn.manifold import TSNE
                perplexity = min(30, max(2, (n_samples - 1) // 3))
                self._reducer = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=1000,
                )
            else:
                from sklearn.decomposition import PCA
                self._reducer = PCA(n_components=2, random_state=42)
        
        try:
            if hasattr(self._reducer, "fit_transform"):
                return self._reducer.fit_transform(embeddings)
            else:
                return self._reducer.fit(embeddings).transform(embeddings)
        except Exception as e:
            log.warning("Projection failed: %s, using PCA fallback", e)
            from sklearn.decomposition import PCA
            fallback = PCA(n_components=2, random_state=42)
            return fallback.fit_transform(embeddings)
    
    def visualize_level(
        self,
        level: int,
        node_ids: list[str],
        labels: list[str],
        embeddings: np.ndarray,
        cluster_assignments: list[int],
        community_labels: list[str] | None = None,
        save: bool = True,
        show: bool = True,
    ) -> VisualizationState:
        """Visualize a single level of the hierarchy.
        
        Args:
            level: The hierarchy level (0 = entities, 1+ = communities)
            node_ids: List of node IDs
            labels: Display labels for each node
            embeddings: High-dimensional embeddings (n_nodes, n_dims)
            cluster_assignments: Cluster ID for each node
            community_labels: Optional labels for each cluster
            save: If True, save the figure to disk
            show: If True, display the figure
            
        Returns:
            VisualizationState containing the projected data
        """
        self._ensure_imports()
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Project embeddings to 2D
        embeddings_2d = self.project_embeddings(embeddings, fit=True)
        
        if community_labels is None:
            community_labels = [f"Cluster {i}" for i in set(cluster_assignments)]
        
        state = VisualizationState(
            level=level,
            node_ids=node_ids,
            labels=labels,
            embeddings_2d=embeddings_2d,
            cluster_assignments=cluster_assignments,
            community_labels=community_labels,
        )
        self._states.append(state)
        
        # Create figure
        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=self._figsize)
            
            # Get unique clusters and create color palette
            unique_clusters = sorted(set(cluster_assignments))
            n_clusters = len(unique_clusters)
            palette = sns.color_palette(self._colormap, n_colors=max(n_clusters, 1))
            cluster_to_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}
            
            # Plot each cluster
            for cluster_id in unique_clusters:
                mask = np.array(cluster_assignments) == cluster_id
                points = embeddings_2d[mask]
                cluster_labels = [labels[i] for i, m in enumerate(mask) if m]
                
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=[cluster_to_color[cluster_id]],
                    s=120,
                    alpha=0.7,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"Cluster {cluster_id}" if cluster_id < 10 else None,
                )
                
                # Add labels for a subset of points to avoid clutter
                for i, (x, y) in enumerate(points):
                    if len(points) <= 20 or i % max(1, len(points) // 10) == 0:
                        label = cluster_labels[i]
                        if len(label) > 25:
                            label = label[:22] + "..."
                        ax.annotate(
                            label,
                            (x, y),
                            fontsize=7,
                            alpha=0.8,
                            xytext=(5, 5),
                            textcoords="offset points",
                        )
            
            # Draw convex hulls around clusters
            from scipy.spatial import ConvexHull
            for cluster_id in unique_clusters:
                mask = np.array(cluster_assignments) == cluster_id
                points = embeddings_2d[mask]
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        hull_points = np.append(hull.vertices, hull.vertices[0])
                        ax.fill(
                            points[hull_points, 0],
                            points[hull_points, 1],
                            alpha=0.15,
                            color=cluster_to_color[cluster_id],
                        )
                        ax.plot(
                            points[hull_points, 0],
                            points[hull_points, 1],
                            color=cluster_to_color[cluster_id],
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.5,
                        )
                    except Exception:
                        pass  # Skip if hull computation fails
            
            # Styling
            level_name = "Entities" if level == 0 else f"Communities (Level {level})"
            ax.set_title(
                f"ArchRAG Hierarchy — {level_name}\n"
                f"{len(node_ids)} nodes in {n_clusters} clusters",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel(f"{self._method.upper()} Dimension 1", fontsize=10)
            ax.set_ylabel(f"{self._method.upper()} Dimension 2", fontsize=10)
            
            if n_clusters <= 15:
                ax.legend(
                    loc="upper right",
                    framealpha=0.9,
                    fontsize=8,
                )
            
            # Add stats box
            stats_text = (
                f"Method: {self._method.upper()}\n"
                f"Level: {level}\n"
                f"Nodes: {len(node_ids)}\n"
                f"Clusters: {n_clusters}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / f"hierarchy_level_{level}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved visualization to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        return state
    
    def visualize_full_hierarchy(
        self,
        hierarchy_data: list[dict[str, Any]],
        save: bool = True,
        show: bool = True,
    ) -> list[VisualizationState]:
        """Visualize all levels of the hierarchy.
        
        Args:
            hierarchy_data: List of dicts with keys:
                - node_ids: list[str]
                - labels: list[str]
                - embeddings: np.ndarray
                - cluster_assignments: list[int]
            save: If True, save figures to disk
            show: If True, display figures
            
        Returns:
            List of VisualizationState for each level
        """
        states = []
        for level, data in enumerate(hierarchy_data):
            state = self.visualize_level(
                level=level,
                node_ids=data["node_ids"],
                labels=data["labels"],
                embeddings=data["embeddings"],
                cluster_assignments=data["cluster_assignments"],
                community_labels=data.get("community_labels"),
                save=save,
                show=show,
            )
            states.append(state)
        return states
    
    def visualize_hierarchy_overview(
        self,
        hierarchy_data: list[dict[str, Any]],
        save: bool = True,
        show: bool = True,
    ) -> None:
        """Create an overview figure showing all hierarchy levels.
        
        Args:
            hierarchy_data: List of level data dicts
            save: If True, save the figure
            show: If True, display the figure
        """
        self._ensure_imports()
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec
        
        n_levels = len(hierarchy_data)
        if n_levels == 0:
            log.warning("No hierarchy data to visualize")
            return
        
        # Determine grid layout
        if n_levels <= 2:
            rows, cols = 1, n_levels
        elif n_levels <= 4:
            rows, cols = 2, 2
        else:
            cols = 3
            rows = (n_levels + cols - 1) // cols
        
        with plt.style.context(self._style):
            fig = plt.figure(figsize=(self._figsize[0] * 1.2, self._figsize[1] * rows / 2))
            gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.25)
            
            for level, data in enumerate(hierarchy_data):
                row = level // cols
                col = level % cols
                ax = fig.add_subplot(gs[row, col])
                
                embeddings = data["embeddings"]
                labels = data["labels"]
                cluster_assignments = data["cluster_assignments"]
                
                # Project to 2D
                if len(embeddings) >= 2:
                    embeddings_2d = self.project_embeddings(embeddings, fit=True)
                elif len(embeddings) == 1:
                    embeddings_2d = np.array([[0.0, 0.0]])
                else:
                    continue
                
                # Plot
                unique_clusters = sorted(set(cluster_assignments))
                n_clusters = len(unique_clusters)
                palette = sns.color_palette(self._colormap, n_colors=max(n_clusters, 1))
                cluster_to_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}
                
                for cluster_id in unique_clusters:
                    mask = np.array(cluster_assignments) == cluster_id
                    points = embeddings_2d[mask]
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        c=[cluster_to_color[cluster_id]],
                        s=60,
                        alpha=0.7,
                        edgecolors="white",
                        linewidths=0.3,
                    )
                
                level_name = "Entities" if level == 0 else f"Level {level}"
                ax.set_title(
                    f"{level_name}: {len(labels)} nodes, {n_clusters} clusters",
                    fontsize=10,
                    fontweight="bold",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            
            fig.suptitle(
                "ArchRAG Hierarchical Clustering Overview",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / "hierarchy_overview.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved overview to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def create_callback(
        self,
        show: bool = False,
        save: bool = True,
    ) -> Callable[[int, list[str], list[str], np.ndarray, list[int]], None]:
        """Create a callback function for real-time visualization during clustering.
        
        Args:
            show: If True, display each level as it's computed
            save: If True, save each level to disk
            
        Returns:
            A callback function compatible with HierarchicalClusteringService
        """
        def callback(
            level: int,
            node_ids: list[str],
            labels: list[str],
            embeddings: np.ndarray,
            cluster_assignments: list[int],
        ) -> None:
            log.info("Visualizing level %d: %d nodes", level, len(node_ids))
            self.visualize_level(
                level=level,
                node_ids=node_ids,
                labels=labels,
                embeddings=embeddings,
                cluster_assignments=cluster_assignments,
                save=save,
                show=show,
            )
        
        return callback
    
    def visualize_with_aggregated_clusters(
        self,
        level: int,
        node_ids: list[str],
        labels: list[str],
        embeddings: np.ndarray,
        cluster_assignments: list[int],
        aggregation_method: str = "mean",
        save: bool = True,
        show: bool = True,
    ) -> VisualizationState:
        """Visualize a level with aggregated cluster embeddings as summary nodes.
        
        Args:
            level: The hierarchy level
            node_ids: List of node IDs
            labels: Display labels
            embeddings: High-dimensional embeddings
            cluster_assignments: Cluster ID for each node
            aggregation_method: Method to aggregate ('mean', 'weighted_mean', 'centroid')
            save: If True, save the figure
            show: If True, display the figure
            
        Returns:
            VisualizationState with aggregated data
        """
        self._ensure_imports()
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.spatial import ConvexHull
        
        # Aggregate embeddings for each cluster
        unique_clusters = sorted(set(cluster_assignments))
        aggregated_embeddings = []
        aggregated_labels = []
        aggregated_cluster_ids = []
        
        for cluster_id in unique_clusters:
            mask = np.array(cluster_assignments) == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_labels = [labels[i] for i, m in enumerate(mask) if m]
            
            # Aggregate
            if aggregation_method == "mean":
                agg_emb = np.mean(cluster_embeddings, axis=0)
            elif aggregation_method == "weighted_mean":
                center = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                weights = 1.0 / (distances + 1e-10)
                weights = weights / weights.sum()
                agg_emb = np.average(cluster_embeddings, axis=0, weights=weights)
            elif aggregation_method == "centroid":
                agg_emb = np.mean(cluster_embeddings, axis=0)
            else:
                agg_emb = np.mean(cluster_embeddings, axis=0)
            
            aggregated_embeddings.append(agg_emb)
            aggregated_labels.append(f"Cluster {cluster_id} ({len(cluster_labels)} nodes)")
            aggregated_cluster_ids.append(cluster_id)
        
        aggregated_embeddings = np.array(aggregated_embeddings)
        
        # Project aggregated embeddings to 2D
        embeddings_2d = self.project_embeddings(aggregated_embeddings, fit=True)
        
        # Create visualization
        with plt.style.context(self._style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self._figsize[0] * 1.5, self._figsize[1]))
            
            # Left: Original nodes
            palette = sns.color_palette(self._colormap, n_colors=max(len(unique_clusters), 1))
            cluster_to_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}
            
            original_2d = self.project_embeddings(embeddings, fit=True)
            
            for cluster_id in unique_clusters:
                mask = np.array(cluster_assignments) == cluster_id
                points = original_2d[mask]
                
                ax1.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=[cluster_to_color[cluster_id]],
                    s=80,
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.5,
                )
            
            ax1.set_title("Original Nodes", fontsize=12, fontweight="bold")
            ax1.set_xlabel(f"{self._method.upper()} Dim 1", fontsize=9)
            ax1.set_ylabel(f"{self._method.upper()} Dim 2", fontsize=9)
            
            # Right: Aggregated cluster nodes
            for i, (cluster_id, label) in enumerate(zip(aggregated_cluster_ids, aggregated_labels)):
                point = embeddings_2d[i]
                color = cluster_to_color[cluster_id]
                
                ax2.scatter(
                    point[0], point[1],
                    c=[color],
                    s=300,
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=2,
                )
                
                ax2.annotate(
                    label,
                    (point[0], point[1]),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
            
            ax2.set_title(f"Aggregated Clusters ({aggregation_method})", fontsize=12, fontweight="bold")
            ax2.set_xlabel(f"{self._method.upper()} Dim 1", fontsize=9)
            ax2.set_ylabel(f"{self._method.upper()} Dim 2", fontsize=9)
            
            fig.suptitle(
                f"Level {level}: Original vs Aggregated Clusters",
                fontsize=14,
                fontweight="bold",
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / f"level_{level}_aggregated_{aggregation_method}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved aggregated visualization to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        state = VisualizationState(
            level=level,
            node_ids=[f"cluster_{c}" for c in aggregated_cluster_ids],
            labels=aggregated_labels,
            embeddings_2d=embeddings_2d,
            cluster_assignments=list(range(len(aggregated_cluster_ids))),
            community_labels=aggregated_labels,
        )
        return state
    
    def export_hierarchy_text(
        self,
        hierarchy_data: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> str:
        """Export the hierarchy as a text representation.
        
        Args:
            hierarchy_data: List of level data dicts
            output_path: Optional path to save the text file
            
        Returns:
            Text representation of the hierarchy
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ArchRAG Hierarchical Clustering Summary")
        lines.append("=" * 60)
        lines.append("")
        
        for level, data in enumerate(hierarchy_data):
            level_name = "Entities (Base Level)" if level == 0 else f"Community Level {level}"
            lines.append(f"### {level_name} ###")
            lines.append("-" * 40)
            
            cluster_assignments = data["cluster_assignments"]
            labels = data["labels"]
            unique_clusters = sorted(set(cluster_assignments))
            
            lines.append(f"Total Nodes: {len(labels)}")
            lines.append(f"Clusters: {len(unique_clusters)}")
            lines.append("")
            
            for cluster_id in unique_clusters:
                cluster_labels = [
                    labels[i] for i, c in enumerate(cluster_assignments) if c == cluster_id
                ]
                lines.append(f"  Cluster {cluster_id} ({len(cluster_labels)} members):")
                for label in cluster_labels[:10]:  # Show first 10
                    display_label = label if len(label) <= 60 else label[:57] + "..."
                    lines.append(f"    • {display_label}")
                if len(cluster_labels) > 10:
                    lines.append(f"    ... and {len(cluster_labels) - 10} more")
                lines.append("")
            
            lines.append("")
        
        text = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(text)
            log.info("Saved hierarchy text to %s", output_path)
        
        return text


def load_hierarchy_from_db(
    db_path: str = "data/archrag.db",
) -> list[dict[str, Any]]:
    """Load hierarchy data from the SQLite database.
    
    Args:
        db_path: Path to the archrag database
        
    Returns:
        List of level data dicts ready for visualization
    """
    import json
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    
    # Load entities (level 0)
    cur = conn.execute("SELECT id, name, description, embedding FROM entities")
    entities = cur.fetchall()
    
    entity_ids = []
    entity_labels = []
    entity_embeddings = []
    
    for eid, name, desc, emb_json in entities:
        entity_ids.append(eid)
        entity_labels.append(name)
        if emb_json:
            entity_embeddings.append(json.loads(emb_json))
        else:
            entity_embeddings.append([0.0] * 768)  # placeholder
    
    # Load hierarchy structure
    cur = conn.execute("SELECT value FROM meta WHERE key='hierarchy_structure'")
    row = cur.fetchone()
    
    hierarchy_data = []
    
    if entities:
        # For level 0, we need to figure out cluster assignments from level 1 communities
        entity_cluster_map = {}
        
        if row:
            structure = json.loads(row[0])
            level_ids = structure.get("level_ids", [])
            
            if level_ids:
                # Level 0 communities tell us entity clusters
                cur = conn.execute(
                    "SELECT id, member_ids FROM communities WHERE level=0"
                )
                for idx, (cid, member_ids_json) in enumerate(cur.fetchall()):
                    member_ids = json.loads(member_ids_json)
                    for mid in member_ids:
                        entity_cluster_map[mid] = idx
        
        # Assign entities to clusters
        cluster_assignments = [
            entity_cluster_map.get(eid, 0) for eid in entity_ids
        ]
        
        hierarchy_data.append({
            "node_ids": entity_ids,
            "labels": entity_labels,
            "embeddings": np.array(entity_embeddings, dtype=np.float32),
            "cluster_assignments": cluster_assignments,
        })
    
    # Load communities by level
    if row:
        structure = json.loads(row[0])
        level_ids = structure.get("level_ids", [])
        
        for level_idx, comm_ids in enumerate(level_ids):
            comm_ids_list = []
            comm_labels = []
            comm_embeddings = []
            comm_cluster_map = {}
            
            cur = conn.execute(
                f"SELECT id, summary, embedding FROM communities WHERE level=?",
                (level_idx,)
            )
            communities = cur.fetchall()
            
            for idx, (cid, summary, emb_json) in enumerate(communities):
                comm_ids_list.append(cid)
                comm_labels.append(summary[:60] if summary else f"Community {cid[:8]}")
                if emb_json:
                    comm_embeddings.append(json.loads(emb_json))
                else:
                    comm_embeddings.append([0.0] * 768)
            
            # Get cluster assignments from next level
            cluster_assignments = list(range(len(comm_ids_list)))  # default: each is own cluster
            
            if level_idx + 1 < len(level_ids):
                # Check next level for parent assignments
                cur = conn.execute(
                    "SELECT id, member_ids FROM communities WHERE level=?",
                    (level_idx + 1,)
                )
                for parent_idx, (pid, member_ids_json) in enumerate(cur.fetchall()):
                    member_ids = json.loads(member_ids_json)
                    for mid in member_ids:
                        if mid in comm_ids_list:
                            idx = comm_ids_list.index(mid)
                            cluster_assignments[idx] = parent_idx
            
            if comm_ids_list:
                hierarchy_data.append({
                    "node_ids": comm_ids_list,
                    "labels": comm_labels,
                    "embeddings": np.array(comm_embeddings, dtype=np.float32),
                    "cluster_assignments": cluster_assignments,
                })
    
    conn.close()
    return hierarchy_data
