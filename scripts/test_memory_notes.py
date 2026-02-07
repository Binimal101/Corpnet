#!/usr/bin/env python3
"""Test script for MemoryNote functionality.

This script tests the MemoryNote system by:
1. Adding sample notes with LLM enrichment
2. Retrieving notes by ID
3. Finding related notes
4. Semantic search
5. Deleting notes

Usage:
    python scripts/test_memory_notes.py

Requires the ArchRAG environment to be set up with a valid config.yaml.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from archrag.config import build_orchestrator


def main():
    print("=" * 60)
    print("MemoryNote System Test")
    print("=" * 60)

    # Load sample notes
    sample_path = Path(__file__).parent.parent / "archrag" / "dataset" / "sample_notes.json"
    if not sample_path.exists():
        print(f"Error: Sample notes file not found at {sample_path}")
        return 1

    with open(sample_path) as f:
        sample_notes = json.load(f)

    print(f"\nLoaded {len(sample_notes)} sample notes")

    # Build orchestrator
    print("\nInitializing orchestrator...")
    try:
        orch = build_orchestrator("config.yaml")
    except Exception as e:
        print(f"Error building orchestrator: {e}")
        return 1

    # Test 1: Add notes
    print("\n" + "-" * 40)
    print("Test 1: Adding Memory Notes")
    print("-" * 40)

    note_ids = []
    for i, note_data in enumerate(sample_notes[:3]):  # Test with first 3
        print(f"\nAdding note {i+1}...")
        print(f"  Content: {note_data['content'][:60]}...")

        try:
            result = orch.add_memory_note(
                note_data,
                enable_linking=True,
                enable_evolution=False,  # Disable evolution for faster testing
                add_to_kg=False,  # Don't add to KG for this test
            )
            note_ids.append(result["id"])
            print(f"  Created: {result['id']}")
            print(f"  Keywords: {result['keywords']}")
            print(f"  Context: {result['context'][:80]}...")
            print(f"  Tags: {result['tags']}")
            print(f"  Links: {len(result['links'])} related notes")
        except Exception as e:
            print(f"  Error: {e}")

    if not note_ids:
        print("No notes created, cannot continue testing")
        return 1

    # Test 2: Retrieve a note
    print("\n" + "-" * 40)
    print("Test 2: Retrieving Memory Note")
    print("-" * 40)

    test_id = note_ids[0]
    print(f"\nRetrieving note {test_id}...")
    result = orch.get_memory_note(test_id)
    if result:
        print(f"  ID: {result['id']}")
        print(f"  Content: {result['content'][:60]}...")
        print(f"  Context: {result['context']}")
        print(f"  Keywords: {result['keywords']}")
        print(f"  Tags: {result['tags']}")
        print(f"  Retrieval count: {result['retrieval_count']}")
    else:
        print(f"  Note not found!")

    # Test 3: Get related notes
    print("\n" + "-" * 40)
    print("Test 3: Getting Related Notes")
    print("-" * 40)

    print(f"\nFinding notes related to {test_id}...")
    related = orch.get_related_notes(test_id)
    if related:
        for r in related:
            print(f"  [{r['relation_type']}] {r['id']}: {r['content'][:50]}...")
    else:
        print("  No related notes found (this is expected with few notes)")

    # Test 4: Semantic search
    print("\n" + "-" * 40)
    print("Test 4: Semantic Search")
    print("-" * 40)

    queries = ["quantum physics experiments", "Nobel Prize winners", "machine learning AI"]
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = orch.search_notes_by_content(query, k=3)
        if results:
            for r in results:
                print(f"  {r['id']}: {r['content'][:50]}... [{', '.join(r['tags'][:2])}]")
        else:
            print("  No results found")

    # Test 5: Stats
    print("\n" + "-" * 40)
    print("Test 5: Database Stats")
    print("-" * 40)

    stats = orch.stats()
    print(f"\n  Entities: {stats['entities']}")
    print(f"  Relations: {stats['relations']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Memory Notes: {stats.get('memory_notes', 0)}")

    # Test 6: Delete a note
    print("\n" + "-" * 40)
    print("Test 6: Deleting Memory Note")
    print("-" * 40)

    if len(note_ids) > 0:
        delete_id = note_ids[-1]
        print(f"\nDeleting note {delete_id}...")
        if orch.delete_memory_note(delete_id):
            print("  Deleted successfully")
        else:
            print("  Delete failed")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
