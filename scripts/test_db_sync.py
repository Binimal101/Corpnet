#!/usr/bin/env python3
"""Test script for external database sync functionality.

This script demonstrates the full pipeline:
1. Connect to the mock SQL database
2. Discover schema and tables
3. Sync records into ArchRAG as MemoryNotes
4. Query the indexed data

Prerequisites:
1. Create the mock database: python scripts/create_mock_db.py
2. Ensure config.yaml exists with LLM and embedding settings
3. Set OPENAI_API_KEY in .env

Run with: python scripts/test_db_sync.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def test_database_sync():
    """Run the full database sync test."""
    from archrag.config import build_orchestrator

    # Check for mock database
    mock_db_path = PROJECT_ROOT / "data" / "mock_company.db"
    if not mock_db_path.exists():
        print("❌ Mock database not found. Run first:")
        print("   python scripts/create_mock_db.py")
        return False

    print("=" * 60)
    print("ArchRAG External Database Sync Test")
    print("=" * 60)

    # 1. Build orchestrator
    print("\n1. Initializing ArchRAG orchestrator...")
    try:
        orch = build_orchestrator("config.yaml")
        print("   ✅ Orchestrator initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False

    # 2. Connect to mock database
    print("\n2. Connecting to mock SQLite database...")
    connection_string = f"sqlite:///{mock_db_path}"
    try:
        result = orch.connect_database("sql", {"connection_string": connection_string})
        print(f"   ✅ Connected to database")
        print(f"   Tables found: {result['tables']}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

    # 3. Explore schema
    print("\n3. Exploring database schema...")
    tables_to_sync = ["employees", "projects", "research_papers", "meeting_notes"]

    for table in tables_to_sync:
        try:
            schema = orch.get_database_schema(table=table)
            text_cols = [c["name"] for c in schema["columns"] if c["is_text"]]
            print(f"   📋 {table}: {len(schema['columns'])} columns, text columns: {text_cols}")
        except Exception as e:
            print(f"   ⚠️ Could not get schema for {table}: {e}")

    # 4. Configure text columns for each table
    print("\n4. Configuring text extraction columns...")
    text_columns_map = {
        "employees": ["name", "role", "bio", "expertise"],
        "projects": ["name", "description", "tech_stack", "goals"],
        "research_papers": ["title", "abstract", "authors", "keywords", "findings"],
        "meeting_notes": ["title", "attendees", "agenda", "discussion", "action_items", "decisions"],
    }
    for table, cols in text_columns_map.items():
        print(f"   {table}: {cols}")

    # 5. Sync database
    print("\n5. Syncing database records to ArchRAG...")
    print("   (This will convert each record to a MemoryNote with LLM-generated metadata)")
    try:
        result = orch.sync_from_database(
            tables=tables_to_sync,
            text_columns_map=text_columns_map,
            incremental=False,  # Full sync
            enable_linking=True,
            enable_evolution=False,  # Disable for faster testing
        )
        print(f"   ✅ Sync completed in {result['duration_seconds']:.2f}s")
        print(f"   Records added: {result['records_added']}")
        print(f"   Records failed: {result['records_failed']}")
        if result['errors']:
            print(f"   Errors: {result['errors'][:3]}")
    except Exception as e:
        print(f"   ❌ Sync failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Check sync status
    print("\n6. Checking sync status...")
    status = orch.get_sync_status()
    for table, info in status.get("tables", {}).items():
        print(f"   {table}: {info['record_count']} records synced")

    # 7. Search notes
    print("\n7. Testing semantic search on synced data...")
    test_queries = [
        "machine learning and neural networks",
        "infrastructure and distributed systems",
        "product roadmap and planning",
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = orch.search_notes_by_content(query, k=3)
        if results:
            for i, r in enumerate(results[:2], 1):
                print(f"   {i}. {r['context'][:80]}...")
        else:
            print("   No results found")

    # 8. Get stats
    print("\n8. Final statistics...")
    stats = orch.stats()
    print(f"   Entities:     {stats.get('entities', 0)}")
    print(f"   Relations:    {stats.get('relations', 0)}")
    print(f"   Chunks:       {stats.get('chunks', 0)}")
    print(f"   Memory notes: {stats.get('memory_notes', 0)}")

    # 9. Disconnect
    print("\n9. Disconnecting from database...")
    orch.disconnect_database()
    print("   ✅ Disconnected")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return True


def test_with_mcp_tools():
    """Demonstrate using MCP tools for database sync."""
    print("\n" + "=" * 60)
    print("MCP Tools Usage Examples")
    print("=" * 60)

    mock_db_path = PROJECT_ROOT / "data" / "mock_company.db"
    if not mock_db_path.exists():
        print("❌ Run create_mock_db.py first")
        return

    print("""
To test via MCP tools (using an MCP client or agent):

1. Connect to database:
   connect_database(
       connector_type="sql",
       connection_string="sqlite:///data/mock_company.db"
   )

2. List tables:
   list_tables()

3. View table schema:
   get_table_schema(table_name="employees")

4. Sync data:
   sync_database(
       tables=["employees", "projects", "research_papers"],
       text_columns={
           "employees": ["name", "role", "bio", "expertise"],
           "projects": ["name", "description", "goals"],
           "research_papers": ["title", "abstract", "findings"]
       },
       incremental=False
   )

5. Search synced notes:
   search_notes(query_str="machine learning infrastructure")

6. Get sync status:
   get_sync_status()

7. Disconnect:
   disconnect_database()
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test database sync functionality")
    parser.add_argument("--mcp-examples", action="store_true", help="Show MCP tool usage examples")
    args = parser.parse_args()

    if args.mcp_examples:
        test_with_mcp_tools()
    else:
        success = test_database_sync()
        sys.exit(0 if success else 1)
