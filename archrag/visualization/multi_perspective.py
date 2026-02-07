"""Multi-perspective visualization of level 0 clustering.

Shows the same data from different dimensionality reduction perspectives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class MultiPerspectiveVisualizer:
    """Visualize level 0 clustering from multiple algorithm perspectives."""
    
    def __init__(
        self,
        methods: list[str] = None,
        figsize: tuple[int, int] = (18, 6),
        style: str = "dark_background",
        colormap: str = "Spectral",
        output_dir: str = "data/visualizations",
    ):
        """Initialize the multi-perspective visualizer.
        
        Args:
            methods: List of methods to use (default: ['umap', 'tsne', 'pca'])
            figsize: Figure size as (width, height)
            style: Matplotlib style
            colormap: Colormap for clusters
            output_dir: Output directory
        """
        self._methods = methods or ["umap", "tsne", "pca"]
        self._figsize = figsize
        self._style = style
        self._colormap = colormap
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def _project_with_method(
        self,
        embeddings: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Project embeddings using a specific method."""
        if len(embeddings) < 2:
            if len(embeddings) == 1:
                return np.array([[0.0, 0.0]])
            return np.array([])
        
        n_samples = len(embeddings)
        
        if method == "umap":
            from umap import UMAP
            n_neighbors = min(15, max(2, n_samples - 1))
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
        elif method == "tsne":
            from sklearn.manifold import TSNE
            perplexity = min(30, max(2, (n_samples - 1) // 3))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                max_iter=1000,
            )
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        try:
            return reducer.fit_transform(embeddings)
        except Exception as e:
            log.warning("Projection with %s failed: %s, using PCA fallback", method, e)
            from sklearn.decomposition import PCA
            fallback = PCA(n_components=2, random_state=42)
            return fallback.fit_transform(embeddings)
    
    def visualize_multi_perspective(
        self,
        node_ids: list[str],
        labels: list[str],
        embeddings: np.ndarray,
        cluster_assignments: list[int],
        save: bool = True,
        show: bool = True,
    ) -> None:
        """Visualize level 0 from multiple algorithm perspectives.
        
        Args:
            node_ids: List of node IDs
            labels: Display labels
            embeddings: High-dimensional embeddings
            cluster_assignments: Cluster ID for each node
            save: If True, save the figure
            show: If True, display the figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.spatial import ConvexHull
        
        n_methods = len(self._methods)
        
        with plt.style.context(self._style):
            fig, axes = plt.subplots(1, n_methods, figsize=self._figsize)
            if n_methods == 1:
                axes = [axes]
            
            # Get unique clusters and color palette
            unique_clusters = sorted(set(cluster_assignments))
            n_clusters = len(unique_clusters)
            palette = sns.color_palette(self._colormap, n_colors=max(n_clusters, 1))
            cluster_to_color = {
                c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)
            }
            
            # Project with each method and plot
            for method_idx, method in enumerate(self._methods):
                ax = axes[method_idx]
                
                # Project embeddings
                embeddings_2d = self._project_with_method(embeddings, method)
                
                if len(embeddings_2d) == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.set_title(f"{method.upper()}\n(No data)", fontsize=12, fontweight="bold")
                    continue
                
                # Plot each cluster
                for cluster_id in unique_clusters:
                    mask = np.array(cluster_assignments) == cluster_id
                    points = embeddings_2d[mask]
                    cluster_labels = [labels[i] for i, m in enumerate(mask) if m]
                    
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        c=[cluster_to_color[cluster_id]],
                        s=100,
                        alpha=0.7,
                        edgecolors="white",
                        linewidths=0.5,
                        label=f"Cluster {cluster_id}" if cluster_id < 10 else None,
                    )
                    
                    # Add labels for subset of points
                    for i, (x, y) in enumerate(points):
                        if len(points) <= 15 or i % max(1, len(points) // 8) == 0:
                            label = cluster_labels[i]
                            if len(label) > 20:
                                label = label[:17] + "..."
                            ax.annotate(
                                label,
                                (x, y),
                                fontsize=6,
                                alpha=0.8,
                                xytext=(3, 3),
                                textcoords="offset points",
                            )
                    
                    # Draw convex hull
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
                            pass
                
                # Styling
                ax.set_title(
                    f"{method.upper()}\n{len(node_ids)} nodes, {n_clusters} clusters",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xlabel(f"{method.upper()} Dim 1", fontsize=9)
                ax.set_ylabel(f"{method.upper()} Dim 2", fontsize=9)
                
                if n_clusters <= 12:
                    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
            
            fig.suptitle(
                "Level 0 Clustering — Multiple Perspectives",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            
            plt.tight_layout()
            
            if save:
                out_path = self._output_dir / "level0_multi_perspective.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                log.info("Saved multi-perspective visualization to %s", out_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def aggregate_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        cluster_assignments: list[int],
        method: str = "mean",
    ) -> tuple[np.ndarray, list[int]]:
        """Aggregate embeddings for each cluster.
        
        Args:
            embeddings: Original embeddings (n_nodes, n_dims)
            cluster_assignments: Cluster ID for each node
            method: Aggregation method ('mean', 'weighted_mean', 'centroid')
            
        Returns:
            Tuple of (aggregated_embeddings, unique_cluster_ids)
        """
        unique_clusters = sorted(set(cluster_assignments))
        aggregated = []
        
        for cluster_id in unique_clusters:
            mask = np.array(cluster_assignments) == cluster_id
            cluster_embeddings = embeddings[mask]
            
            if method == "mean":
                agg_emb = np.mean(cluster_embeddings, axis=0)
            elif method == "weighted_mean":
                # Weight by inverse distance from cluster center
                center = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                weights = 1.0 / (distances + 1e-10)
                weights = weights / weights.sum()
                agg_emb = np.average(cluster_embeddings, axis=0, weights=weights)
            elif method == "centroid":
                agg_emb = np.mean(cluster_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            aggregated.append(agg_emb)
        
        return np.array(aggregated), unique_clusters
