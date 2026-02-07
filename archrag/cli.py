"""ArchRAG command-line interface."""

from __future__ import annotations

import json
import logging

import click


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, config: str, verbose: bool) -> None:
    """ArchRAG: Attributed Community-based Hierarchical RAG."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@main.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.pass_context
def index(ctx: click.Context, corpus: str) -> None:
    """Build the full index from a CORPUS file (JSONL or JSON array)."""
    from archrag.config import build_orchestrator

    orch = build_orchestrator(ctx.obj["config"])
    orch.index(corpus)
    click.echo("Indexing complete.")


@main.command()
@click.argument("question")
@click.pass_context
def query(ctx: click.Context, question: str) -> None:
    """Answer a QUESTION using the built index."""
    from archrag.config import build_orchestrator

    orch = build_orchestrator(ctx.obj["config"])
    answer = orch.query(question)
    click.echo(answer)


@main.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.pass_context
def add(ctx: click.Context, corpus: str) -> None:
    """Add new documents from CORPUS file to an existing index."""
    from archrag.config import build_orchestrator

    orch = build_orchestrator(ctx.obj["config"])
    documents = orch._load_corpus(corpus)
    orch.add_documents(documents)
    click.echo(f"Added {len(documents)} documents and re-indexed.")


@main.command()
@click.argument("entity_name")
@click.pass_context
def remove(ctx: click.Context, entity_name: str) -> None:
    """Remove an entity by NAME from the knowledge graph."""
    from archrag.config import build_orchestrator

    orch = build_orchestrator(ctx.obj["config"])
    if orch.remove_entity(entity_name):
        click.echo(f"Removed entity: {entity_name}")
    else:
        click.echo(f"Entity not found: {entity_name}")


@main.command()
@click.argument("query_str")
@click.option("--type", "-t", "search_type", default="entities",
              type=click.Choice(["entities", "chunks", "all"]),
              help="What to search: entities, chunks, or all.")
@click.pass_context
def search(ctx: click.Context, query_str: str, search_type: str) -> None:
    """Search the knowledge graph for QUERY_STR (substring match)."""
    from archrag.config import build_orchestrator

    orch = build_orchestrator(ctx.obj["config"])

    if search_type in ("entities", "all"):
        entities = orch.search_entities(query_str)
        if entities:
            click.echo(f"\n--- Entities matching '{query_str}' ({len(entities)}) ---")
            for e in entities:
                click.echo(f"  [{e['type']}] {e['name']}: {e['description'][:120]}")
        else:
            click.echo(f"No entities matching '{query_str}'.")

    if search_type in ("chunks", "all"):
        chunks = orch.search_chunks(query_str)
        if chunks:
            click.echo(f"\n--- Chunks matching '{query_str}' ({len(chunks)}) ---")
            for c in chunks:
                click.echo(f"  {c['id']}: {c['text'][:120]}...")
        else:
            click.echo(f"No chunks matching '{query_str}'.")


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show database stats and current configuration."""
    from archrag.config import build_orchestrator, load_config

    import yaml as _yaml

    cfg = load_config(ctx.obj["config"])
    orch = build_orchestrator(ctx.obj["config"])
    st = orch.stats()

    click.echo("=== Database Stats ===")
    click.echo(f"  Entities:         {st['entities']}")
    click.echo(f"  Relations:        {st['relations']}")
    click.echo(f"  Chunks:           {st['chunks']}")
    click.echo(f"  Hierarchy levels: {st['hierarchy_levels']}")
    click.echo("\n=== Config ===")
    click.echo(_yaml.dump(cfg, default_flow_style=False))


@main.command()
@click.option("--method", "-m", default="umap",
              type=click.Choice(["umap", "tsne", "pca"]),
              help="Dimensionality reduction method.")
@click.option("--output", "-o", default="data/visualizations",
              help="Output directory for visualizations.")
@click.option("--no-show", is_flag=True, help="Don't display figures (just save).")
@click.option("--overview", is_flag=True, help="Generate overview figure with all levels.")
@click.option("--text", is_flag=True, help="Export hierarchy as text summary.")
@click.pass_context
def visualize(ctx: click.Context, method: str, output: str, no_show: bool, overview: bool, text: bool) -> None:
    """Visualize the hierarchical clustering with reverse embedding projection."""
    from archrag.config import load_config
    from archrag.visualization.hierarchy_viz import HierarchyVisualizer, load_hierarchy_from_db
    
    cfg = load_config(ctx.obj["config"])
    db_path = cfg.get("document_store", {}).get("path", "data/archrag.db")
    
    click.echo(f"Loading hierarchy from {db_path}...")
    hierarchy_data = load_hierarchy_from_db(db_path)
    
    if not hierarchy_data:
        click.echo("No hierarchy data found. Run 'archrag index' first.")
        return
    
    click.echo(f"Found {len(hierarchy_data)} hierarchy levels.")
    
    viz = HierarchyVisualizer(
        method=method,
        output_dir=output,
        interactive=not no_show,
    )
    
    if text:
        text_output = viz.export_hierarchy_text(
            hierarchy_data,
            output_path=f"{output}/hierarchy_summary.txt",
        )
        click.echo("\n" + text_output)
    
    if overview:
        click.echo("Generating overview visualization...")
        viz.visualize_hierarchy_overview(
            hierarchy_data,
            save=True,
            show=not no_show,
        )
    else:
        click.echo("Generating per-level visualizations...")
        for level, data in enumerate(hierarchy_data):
            level_name = "entities" if level == 0 else f"level {level}"
            click.echo(f"  Visualizing {level_name}: {len(data['node_ids'])} nodes...")
            viz.visualize_level(
                level=level,
                node_ids=data["node_ids"],
                labels=data["labels"],
                embeddings=data["embeddings"],
                cluster_assignments=data["cluster_assignments"],
                save=True,
                show=not no_show,
            )
    
    click.echo(f"\nVisualizations saved to {output}/")


@main.command("index-viz")
@click.argument("corpus", type=click.Path(exists=True))
@click.option("--method", "-m", default="umap",
              type=click.Choice(["umap", "tsne", "pca"]),
              help="Dimensionality reduction method.")
@click.option("--output", "-o", default="data/visualizations",
              help="Output directory for visualizations.")
@click.option("--show", is_flag=True, help="Display figures in real-time during indexing.")
@click.pass_context
def index_with_visualization(ctx: click.Context, corpus: str, method: str, output: str, show: bool) -> None:
    """Build the full index with real-time visualization of clustering."""
    from archrag.config import build_orchestrator
    from archrag.visualization.hierarchy_viz import HierarchyVisualizer
    
    click.echo("Building index with real-time visualization...")
    
    viz = HierarchyVisualizer(
        method=method,
        output_dir=output,
        interactive=show,
    )
    
    # Build orchestrator and attach visualization callback
    orch = build_orchestrator(ctx.obj["config"])
    
    # Get the cluster service from the snapshot and attach callback
    callback = viz.create_callback(show=show, save=True)
    orch._snapshot.cluster_service.set_visualization_callback(callback)
    
    # Run indexing
    orch.index(corpus)
    
    click.echo("Indexing complete.")
    click.echo(f"Visualizations saved to {output}/")


@main.command("viz-dendrogram")
@click.option("--output", "-o", default="data/visualizations",
              help="Output directory for visualizations.")
@click.option("--no-show", is_flag=True, help="Don't display figures (just save).")
@click.option("--sunburst", is_flag=True, help="Generate sunburst chart instead of dendrogram.")
@click.option("--treemap", is_flag=True, help="Generate treemap visualization.")
@click.option("--level", "-l", default=0, type=int, help="Level for treemap (default: 0).")
@click.pass_context
def visualize_dendrogram(ctx: click.Context, output: str, no_show: bool, sunburst: bool, treemap: bool, level: int) -> None:
    """Visualize the hierarchy as a dendrogram, sunburst, or treemap."""
    from archrag.config import load_config
    from archrag.visualization.hierarchy_viz import load_hierarchy_from_db
    from archrag.visualization.dendrogram import DendrogramVisualizer
    
    cfg = load_config(ctx.obj["config"])
    db_path = cfg.get("document_store", {}).get("path", "data/archrag.db")
    
    click.echo(f"Loading hierarchy from {db_path}...")
    hierarchy_data = load_hierarchy_from_db(db_path)
    
    if not hierarchy_data:
        click.echo("No hierarchy data found. Run 'archrag index' first.")
        return
    
    viz = DendrogramVisualizer(output_dir=output)
    
    if sunburst:
        click.echo("Generating sunburst visualization...")
        viz.visualize_sunburst(hierarchy_data, save=True, show=not no_show)
    elif treemap:
        click.echo(f"Generating treemap visualization for level {level}...")
        viz.visualize_treemap(hierarchy_data, level=level, save=True, show=not no_show)
    else:
        click.echo("Generating dendrogram visualization...")
        viz.visualize_dendrogram(hierarchy_data, save=True, show=not no_show)
    
    click.echo(f"Visualization saved to {output}/")


@main.command("viz-multi-perspective")
@click.option("--output", "-o", default="data/visualizations",
              help="Output directory for visualizations.")
@click.option("--no-show", is_flag=True, help="Don't display figures (just save).")
@click.option("--methods", "-m", multiple=True, default=["umap", "tsne", "pca"],
              type=click.Choice(["umap", "tsne", "pca"]),
              help="Dimensionality reduction methods to use (can specify multiple).")
@click.pass_context
def visualize_multi_perspective(ctx: click.Context, output: str, no_show: bool, methods: tuple[str, ...]) -> None:
    """Visualize level 0 clustering from multiple algorithm perspectives."""
    from archrag.config import load_config
    from archrag.visualization.hierarchy_viz import load_hierarchy_from_db
    from archrag.visualization.multi_perspective import MultiPerspectiveVisualizer
    
    cfg = load_config(ctx.obj["config"])
    db_path = cfg.get("document_store", {}).get("path", "data/archrag.db")
    
    click.echo(f"Loading hierarchy from {db_path}...")
    hierarchy_data = load_hierarchy_from_db(db_path)
    
    if not hierarchy_data or len(hierarchy_data) == 0:
        click.echo("No hierarchy data found. Run 'archrag index' first.")
        return
    
    level_0_data = hierarchy_data[0]
    
    click.echo(f"Visualizing level 0 with {len(methods)} perspectives: {', '.join(methods)}...")
    
    viz = MultiPerspectiveVisualizer(
        methods=list(methods),
        output_dir=output,
    )
    
    viz.visualize_multi_perspective(
        node_ids=level_0_data["node_ids"],
        labels=level_0_data["labels"],
        embeddings=level_0_data["embeddings"],
        cluster_assignments=level_0_data["cluster_assignments"],
        save=True,
        show=not no_show,
    )
    
    click.echo(f"Multi-perspective visualization saved to {output}/")


@main.command("viz-tree")
@click.option("--output", "-o", default="data/visualizations",
              help="Output directory for visualizations.")
@click.option("--no-show", is_flag=True, help="Don't display figures (just save).")
@click.option("--animated", is_flag=True, help="Generate animated tree (GIF).")
@click.option("--fps", default=2, type=int, help="Frames per second for animation.")
@click.pass_context
def visualize_tree(ctx: click.Context, output: str, no_show: bool, animated: bool, fps: int) -> None:
    """Visualize the hierarchy as a tree: Input Data -> Entities -> Communities -> Chunks."""
    from archrag.config import load_config
    from archrag.visualization.animated_tree import AnimatedTreeVisualizer
    
    cfg = load_config(ctx.obj["config"])
    db_path = cfg.get("document_store", {}).get("path", "data/archrag.db")
    
    viz = AnimatedTreeVisualizer(output_dir=output, fps=fps)
    
    if animated:
        click.echo("Generating animated tree visualization...")
        viz.visualize_animated_tree(db_path=db_path, save=True, show=not no_show)
    else:
        click.echo("Generating static tree visualization...")
        viz.visualize_static_tree(db_path=db_path, save=True, show=not no_show)
    
    click.echo(f"Tree visualization saved to {output}/")


@main.command("viz-aggregated")
@click.option("--output", "-o", default="data/visualizations",
              help="Output directory for visualizations.")
@click.option("--no-show", is_flag=True, help="Don't display figures (just save).")
@click.option("--level", "-l", default=0, type=int, help="Hierarchy level to visualize (default: 0).")
@click.option("--method", "-m", default="mean",
              type=click.Choice(["mean", "weighted_mean", "centroid"]),
              help="Aggregation method for cluster embeddings.")
@click.pass_context
def visualize_aggregated(ctx: click.Context, output: str, no_show: bool, level: int, method: str) -> None:
    """Visualize hierarchy level with aggregated cluster embeddings as summary nodes."""
    from archrag.config import load_config
    from archrag.visualization.hierarchy_viz import HierarchyVisualizer, load_hierarchy_from_db
    
    cfg = load_config(ctx.obj["config"])
    db_path = cfg.get("document_store", {}).get("path", "data/archrag.db")
    
    click.echo(f"Loading hierarchy from {db_path}...")
    hierarchy_data = load_hierarchy_from_db(db_path)
    
    if not hierarchy_data or level >= len(hierarchy_data):
        click.echo(f"Level {level} not found. Available levels: 0-{len(hierarchy_data)-1}")
        return
    
    level_data = hierarchy_data[level]
    
    click.echo(f"Visualizing level {level} with aggregated clusters ({method})...")
    
    viz = HierarchyVisualizer(
        method="umap",
        output_dir=output,
    )
    
    viz.visualize_with_aggregated_clusters(
        level=level,
        node_ids=level_data["node_ids"],
        labels=level_data["labels"],
        embeddings=level_data["embeddings"],
        cluster_assignments=level_data["cluster_assignments"],
        aggregation_method=method,
        save=True,
        show=not no_show,
    )
    
    click.echo(f"Aggregated visualization saved to {output}/")


if __name__ == "__main__":
    main()
