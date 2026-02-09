#!/usr/bin/env python3
"""Generate Notes from topic JSONL files.

This script:
1. Loads facts from data/topics/*.jsonl
2. Converts them to simplified Note format
3. Saves Notes to data/notes/*.jsonl (one file per topic)
4. Reserves some notes for single add_data() testing
5. Outputs notes for initial_upload() bulk upload

Note format (simplified):
{
    "content": "...",
    "tags": [...],
    "category": "...",
    "retrieval_count": 0
}

Removed fields: id, access_id, keywords, last_updated, embedding_model, embedding
"""

import json
import os
import random
from pathlib import Path

# Topics with their category names
TOPICS = {
    "law": "Law",
    "computer_science": "Computer Science",
    "finance": "Finance",
    "geology": "Geology",
    "physics": "Physics",
}

# Number of notes to reserve for single add_data() testing
RESERVED_PER_TOPIC = 1  # Total 5 reserved notes (1 per topic)


def load_facts(topic: str) -> list[dict]:
    """Load facts from a topic JSONL file."""
    path = Path(f"data/topics/{topic}.jsonl")
    facts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


def generate_note_from_fact(fact: dict, category: str) -> dict:
    """Generate a simplified Note from a fact.
    
    Simplified Note format:
    - content: the fact text
    - tags: derived from category
    - category: the topic category
    - retrieval_count: always 0
    
    No: id, access_id, keywords, last_updated, embedding_model, embedding
    """
    content = fact.get("text", fact.get("content", ""))
    
    # Basic tags derived from category
    tags = [category.lower().replace(" ", "_")]
    
    note = {
        "content": content,
        "tags": tags,
        "category": category,
        "retrieval_count": 0,
    }
    
    return note


def save_notes(notes: list[dict], output_path: Path) -> None:
    """Save notes to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for note in notes:
            f.write(json.dumps(note) + "\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Note Generation Script")
    print("=" * 60)
    
    # Create output directories
    notes_dir = Path("data/notes")
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    all_notes = []
    reserved_notes = []
    bulk_notes = []
    
    for topic, category in TOPICS.items():
        print(f"\nProcessing {topic}...")
        
        # Load facts
        facts = load_facts(topic)
        print(f"  Loaded {len(facts)} facts")
        
        # Generate notes
        notes = []
        for fact in facts:
            note = generate_note_from_fact(fact=fact, category=category)
            notes.append(note)
        
        # Save all notes for this topic
        output_path = notes_dir / f"{topic}_notes.jsonl"
        save_notes(notes, output_path)
        print(f"  Saved {len(notes)} notes to {output_path}")
        
        # Reserve some notes for single add_data() testing
        random.shuffle(notes)
        topic_reserved = notes[:RESERVED_PER_TOPIC]
        topic_bulk = notes[RESERVED_PER_TOPIC:]
        
        reserved_notes.extend(topic_reserved)
        bulk_notes.extend(topic_bulk)
        all_notes.extend(notes)
    
    # Save reserved notes (for single add_data() testing)
    reserved_path = notes_dir / "reserved_for_add.jsonl"
    save_notes(reserved_notes, reserved_path)
    print(f"\n{'=' * 60}")
    print(f"Reserved {len(reserved_notes)} notes for add_data() testing")
    print(f"  Saved to: {reserved_path}")
    
    # Save bulk notes (for initial_upload())
    bulk_path = notes_dir / "bulk_upload.jsonl"
    save_notes(bulk_notes, bulk_path)
    print(f"\nPrepared {len(bulk_notes)} notes for initial_upload()")
    print(f"  Saved to: {bulk_path}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total notes generated: {len(all_notes)}")
    print(f"Reserved for add_data(): {len(reserved_notes)} (1 per topic)")
    print(f"For initial_upload():    {len(bulk_notes)}")
    print(f"\nOutput files in {notes_dir}/:")
    for f in sorted(notes_dir.glob("*.jsonl")):
        line_count = sum(1 for _ in open(f))
        print(f"  - {f.name}: {line_count} notes")
    
    # Print sample notes
    print(f"\n{'=' * 60}")
    print("SAMPLE NOTE FORMAT:")
    print(f"{'=' * 60}")
    if reserved_notes:
        print(json.dumps(reserved_notes[0], indent=2))
    
    print(f"\n{'=' * 60}")
    print("USAGE:")
    print(f"{'=' * 60}")
    print("1. Start ArchRAG API server:")
    print("   python -m archrag.api_server")
    print()
    print("2. Start Notes API server:")
    print("   archrag api")
    print()
    print("3. Test single add_data():")
    print("   POST http://localhost:8080/notes/add_data")
    print("   Body:", json.dumps(reserved_notes[0]) if reserved_notes else "{}")
    print()
    print("4. Bulk initial_upload() (70% of notes):")
    print('   POST http://localhost:8080/notes/initial_upload')
    print('   Body: {"source": "bulk_upload", "percentage": 0.7}')


if __name__ == "__main__":
    main()
