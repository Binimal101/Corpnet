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
    click.echo(f"  Memory notes:     {st.get('memory_notes', 0)}")
    click.echo("\n=== Config ===")
    click.echo(_yaml.dump(cfg, default_flow_style=False))


@main.command()
@click.option("--no-llm", is_flag=True, help="Use guided prompts without LLM (form-based).")
@click.option("--no-greeting", is_flag=True, help="Skip the greeting message.")
@click.pass_context
def agent(ctx: click.Context, no_llm: bool, no_greeting: bool) -> None:
    """Interactive agent for guided data ingestion.
    
    The agent helps you:
    - Connect to SQL/NoSQL databases
    - Save connection details for future use
    - Select which data to index
    - Sync data to ArchRAG
    - Query your indexed data
    
    Examples:
        archrag agent                   # Full conversational agent
        archrag agent --no-llm          # Form-based guided setup
    """
    from archrag.config import build_orchestrator
    from archrag.agent.connection_store import ConnectionStore
    from archrag.agent.ingestion_agent import IngestionAgent, GuidedSetup

    orch = build_orchestrator(ctx.obj["config"])
    store = ConnectionStore()

    if no_llm:
        # Use form-based guided setup
        setup = GuidedSetup(orch, store)
        setup.run()
    else:
        # Use full conversational agent
        agent_instance = IngestionAgent(orch, orch._llm, store)
        agent_instance.run_interactive(greeting=not no_greeting)


@main.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the MCP server for AI agent integration.
    
    This starts a local MCP server that can be connected to by
    MCP-compatible clients like Claude Desktop or Cursor.
    
    The server runs on stdio transport by default.
    
    Examples:
        archrag serve                   # Start MCP server
        
    To configure Claude Desktop, add to config.json:
        {
            "mcpServers": {
                "archrag": {
                    "command": "archrag",
                    "args": ["serve"]
                }
            }
        }
    """
    import os
    os.environ.setdefault("ARCHRAG_CONFIG", ctx.obj["config"])
    
    from archrag.mcp_server import mcp
    mcp.run(transport="stdio")


@main.command()
@click.pass_context
def connections(ctx: click.Context) -> None:
    """List saved database connections."""
    from archrag.agent.connection_store import ConnectionStore

    store = ConnectionStore()
    conns = store.list_connections()

    if not conns:
        click.echo("No saved connections.")
        click.echo("Use 'archrag agent' to connect a database and save it.")
        return

    click.echo(f"Saved connections ({len(conns)}):\n")
    for conn in conns:
        click.echo(f"  {conn.name} ({conn.connector_type})")
        if conn.description:
            click.echo(f"    Description: {conn.description}")
        click.echo(f"    Tables: {', '.join(conn.tables) if conn.tables else 'not configured'}")
        click.echo(f"    Last used: {conn.last_used_at or 'never'}")
        click.echo(f"    Total syncs: {conn.total_syncs} ({conn.total_records_synced} records)")
        click.echo()


@main.command("connect")
@click.argument("name")
@click.pass_context
def connect_saved(ctx: click.Context, name: str) -> None:
    """Connect to a saved database by NAME and sync."""
    from archrag.config import build_orchestrator
    from archrag.agent.connection_store import ConnectionStore
    from archrag.agent.tools import AgentTools

    orch = build_orchestrator(ctx.obj["config"])
    store = ConnectionStore()
    tools = AgentTools(orch, store)

    # Connect
    result = tools.execute_tool("connect_database", {"saved_name": name})
    if not result.success:
        click.echo(f"❌ {result.message}")
        return
    
    click.echo(f"✅ {result.message}")
    
    # Get saved connection to check for configured tables
    conn = store.get_connection(name)
    if conn and conn.tables:
        click.echo(f"Syncing tables: {', '.join(conn.tables)}")
        sync_result = tools.execute_tool("sync_database", {
            "tables": conn.tables,
            "incremental": True,
        })
        click.echo(f"{'✅' if sync_result.success else '❌'} {sync_result.message}")
    else:
        click.echo("No tables configured. Use 'archrag agent' to configure sync settings.")


if __name__ == "__main__":
    main()
