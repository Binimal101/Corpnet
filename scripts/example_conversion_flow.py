#!/usr/bin/env python3
"""Example: Trace a complex SQL database entry through the conversion pipeline.

This script demonstrates how a multi-column SQL record flows through:
1. SQL Record → ExternalRecord (via GenericSQLConnector)
2. ExternalRecord → Note Input Dict (via to_note_input)
3. Note Input → MemoryNote (via NoteConstructionService.build_note)
4. MemoryNote → One-sentence context (via LLM in build_note)
5. MemoryNote → Indexed (via embedding and save_note)

Run with: python3 scripts/example_conversion_flow.py
"""

from archrag.domain.models import ExternalRecord, MemoryNote
from archrag.adapters.connectors.sql_connector import GenericSQLConnector
from archrag.services.note_construction import NoteConstructionService

# ── STEP 1: Original SQL Database Entry ──────────────────────────────────

print("=" * 80)
print("STEP 1: Original SQL Database Entry")
print("=" * 80)

sql_record = {
    "id": 1,
    "name": "Alice Chen",
    "email": "alice@company.com",
    "role": "Senior ML Engineer",
    "department_id": 2,
    "bio": "Alice specializes in transformer architectures and has led the development of several production NLP systems. She has published papers on attention mechanisms and efficient inference.",
    "expertise": "natural language processing, transformers, deep learning, attention mechanisms",
    "created_at": "2024-01-15 10:30:00",
    "updated_at": "2024-01-20 14:22:00",
}

print("\nSQL Table: employees")
print("Columns:", list(sql_record.keys()))
print("\nRaw Record:")
for key, value in sql_record.items():
    print(f"  {key:15} = {value}")

# ── STEP 2: SQL Record → ExternalRecord ───────────────────────────────────

print("\n" + "=" * 80)
print("STEP 2: SQL Record → ExternalRecord")
print("Function: GenericSQLConnector.fetch_records()")
print("=" * 80)

# Simulate what GenericSQLConnector.fetch_records() does
text_columns = ["name", "role", "bio", "expertise"]  # Columns selected for text extraction

# Concatenate text columns
text_parts = []
for col in text_columns:
    if col in sql_record and sql_record[col]:
        text_parts.append(str(sql_record[col]))
text_content = " ".join(text_parts)

external_record = ExternalRecord(
    id=str(sql_record["id"]),
    source_table="employees",
    source_database="mock_company",
    content=sql_record,  # All original fields preserved
    text_content=text_content,  # Concatenated text for indexing
    metadata={
        "columns": list(sql_record.keys()),
        "primary_key": "id",
    },
    created_at=str(sql_record.get("created_at")),
    updated_at=str(sql_record.get("updated_at")),
)

print("\nExternalRecord created:")
print(f"  id: {external_record.id}")
print(f"  source_table: {external_record.source_table}")
print(f"  source_database: {external_record.source_database}")
print(f"  text_content: {external_record.text_content[:100]}...")
print(f"  content (original): {len(external_record.content)} fields")
print(f"  metadata: {external_record.metadata}")

# ── STEP 3: ExternalRecord → Note Input Dict ──────────────────────────────

print("\n" + "=" * 80)
print("STEP 3: ExternalRecord → Note Input Dict")
print("Function: ExternalRecord.to_note_input()")
print("=" * 80)

note_input = external_record.to_note_input()

print("\nNote Input Dict (for NoteConstructionService):")
for key, value in note_input.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    elif isinstance(value, list):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value[:100] if isinstance(value, str) and len(value) > 100 else value}")

# ── STEP 4: Note Input → MemoryNote (with LLM-generated one-sentence context) ──

print("\n" + "=" * 80)
print("STEP 4: Note Input → MemoryNote")
print("Function: NoteConstructionService.build_note()")
print("=" * 80)

print("\nThis step performs:")
print("  1. Extract content from input_data")
print("  2. Call LLM with NOTE_CONSTRUCTION_PROMPT to generate:")
print("     - keywords: ['machine learning', 'NLP', 'transformers', ...]")
print("     - context: 'Senior ML engineer specializing in transformer architectures...' (ONE SENTENCE)")
print("     - tags: ['engineering', 'AI/ML', 'personnel']")
print("  3. Compute embedding from: content + context + keywords")
print("  4. Find nearest notes for linking")
print("  5. Generate links to related notes")
print("  6. Return MemoryNote object")

print("\nExample MemoryNote (after LLM processing):")
print("  content: 'Alice Chen Senior ML Engineer Alice specializes in...'")
print("  context: 'Senior ML engineer specializing in transformer architectures and NLP systems'")
print("  keywords: ['machine learning', 'NLP', 'transformers', 'attention mechanisms', 'deep learning']")
print("  tags: ['engineering', 'AI/ML', 'personnel', 'research']")
print("  category: 'employees'")
print("  embedding: [0.123, -0.456, 0.789, ...] (1536-dim vector)")
print("  links: {'note_abc123': 'related expertise', 'note_def456': 'same department'}")

# ── STEP 5: MemoryNote → Indexed (Saved to Store) ────────────────────────

print("\n" + "=" * 80)
print("STEP 5: MemoryNote → Indexed")
print("Function: MemoryNoteStorePort.save_note()")
print("=" * 80)

print("\nThe MemoryNote is saved to the database with:")
print("  - All metadata (keywords, context, tags, links)")
print("  - Embedding vector for semantic search")
print("  - Timestamps and retrieval stats")
print("\nThe ONE-SENTENCE 'context' field is what gets used for:")
print("  - Quick summaries in search results")
print("  - Display in UI")
print("  - Part of the embedding computation (content + context + keywords)")

# ── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY: Complete Conversion Flow")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1. SQL Record (9 columns)                                       │
│    └─> GenericSQLConnector.fetch_records()                     │
│        • Selects text columns: name, role, bio, expertise       │
│        • Concatenates: "Alice Chen Senior ML Engineer Alice..."  │
│                                                                  │
│ 2. ExternalRecord                                                │
│    └─> ExternalRecord.to_note_input()                           │
│        • Extracts text_content as 'content'                     │
│        • Sets category = source_table ('employees')              │
│        • Preserves original fields in metadata                   │
│                                                                  │
│ 3. Note Input Dict                                               │
│    └─> NoteConstructionService.build_note()                    │
│        • Calls LLM with NOTE_CONSTRUCTION_PROMPT                 │
│        • LLM generates:                                          │
│          - keywords: ['NLP', 'transformers', ...]               │
│          - context: 'Senior ML engineer specializing in...' ⭐   │
│          - tags: ['engineering', 'AI/ML']                      │
│        • Computes embedding from: content + context + keywords  │
│        • Finds nearest notes and generates links                │
│                                                                  │
│ 4. MemoryNote (with one-sentence context)                        │
│    └─> MemoryNoteStorePort.save_note()                          │
│        • Saves to SQLite with embedding vector                   │
│        • Now searchable via semantic search                      │
│                                                                  │
│ 5. Indexed & Searchable                                          │
│    • The 'context' field (one sentence) is used for:             │
│      - Display in search results                                │
│      - Quick summaries                                          │
│      - Part of embedding computation                            │
└─────────────────────────────────────────────────────────────────┘

Key Functions:
  • GenericSQLConnector.fetch_records() → ExternalRecord
  • ExternalRecord.to_note_input() → dict
  • NoteConstructionService.build_note() → MemoryNote (with LLM-generated context)
  • MemoryNoteStorePort.save_note() → Persisted & Indexed
""")

print("\nThe ONE-SENTENCE summary is generated by the LLM in step 4,")
print("using the NOTE_CONSTRUCTION_PROMPT which asks for:")
print("  'one sentence summarizing: Main topic/domain, Key arguments/points, Intended audience/purpose'")
