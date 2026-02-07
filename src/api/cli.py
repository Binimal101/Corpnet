"""Command-line interface for DAC-HRAG.

Commands:
- serve: Start the API server
- query: Run a query from the command line
- ingest: Ingest data from a file or directory
- peer: Start a peer node
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="DAC-HRAG: Distributed Attributed Community-Hierarchical RAG")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn
    console.print(f"[bold green]Starting DAC-HRAG API server on {host}:{port}[/bold green]")
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(10, help="Number of results"),
    threshold: float = typer.Option(0.35, help="Similarity threshold"),
):
    """Run a query from the command line."""
    console.print(f"[bold blue]Query:[/bold blue] {text}")
    console.print(f"  Top-k: {top_k}, Threshold: {threshold}")
    
    # TODO: Initialize query engine
    console.print("[yellow]Query engine not initialized. Run 'serve' first and use HTTP API.[/yellow]")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory"),
    source: str = typer.Option("file", help="Source identifier"),
    chunk_size: int = typer.Option(512, help="Chunk size"),
):
    """Ingest data from a file or directory."""
    console.print(f"[bold blue]Ingesting from:[/bold blue] {path}")
    console.print(f"  Source: {source}, Chunk size: {chunk_size}")
    
    import os
    if not os.path.exists(path):
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)
    
    # TODO: Initialize pipeline
    console.print("[yellow]Ingestion pipeline not initialized.[/yellow]")


@app.command()
def peer(
    peer_id: str = typer.Option("peer-001", help="Peer ID"),
    port: int = typer.Option(50051, help="gRPC port"),
    coordinator: str = typer.Option("localhost:50050", help="Coordinator address"),
    super_peer: bool = typer.Option(False, help="Run as super-peer"),
):
    """Start a peer node."""
    role = "super-peer" if super_peer else "leaf peer"
    console.print(f"[bold green]Starting {role}: {peer_id}[/bold green]")
    console.print(f"  Port: {port}, Coordinator: {coordinator}")
    
    # TODO: Initialize peer
    console.print("[yellow]Peer networking not yet implemented.[/yellow]")


@app.command()
def status():
    """Show system status."""
    table = Table(title="DAC-HRAG Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    table.add_row("API", "Not Running", "Run 'dachrag serve' to start")
    table.add_row("Database", "Unknown", "Check DATABASE_URL config")
    table.add_row("Peers", "0", "No peers connected")
    table.add_row("Hierarchy", "Empty", "Run clustering to build")
    
    console.print(table)


@app.command()
def config():
    """Show current configuration."""
    from src.core.config import get_settings
    
    settings = get_settings()
    
    console.print("[bold]Database[/bold]")
    console.print(f"  URL: {settings.database.url}")
    
    console.print("[bold]Embeddings[/bold]")
    console.print(f"  Provider: {settings.embeddings.provider}")
    console.print(f"  Model: {settings.embeddings.model}")
    console.print(f"  Dimension: {settings.embeddings.dimension}")
    
    console.print("[bold]LLM[/bold]")
    console.print(f"  Provider: {settings.llm.provider}")
    console.print(f"  Model: {settings.llm.model}")
    
    console.print("[bold]Routing[/bold]")
    console.print(f"  Threshold: {settings.routing.similarity_threshold}")
    console.print(f"  Top-k: {settings.routing.top_k_results}")


def main():
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
