"""ArchRAG command-line interface."""

from __future__ import annotations

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
    """Run the offline indexing pipeline on CORPUS (JSONL file)."""
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
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show the current configuration."""
    from archrag.config import load_config

    import yaml as _yaml

    cfg = load_config(ctx.obj["config"])
    click.echo(_yaml.dump(cfg, default_flow_style=False))


if __name__ == "__main__":
    main()
