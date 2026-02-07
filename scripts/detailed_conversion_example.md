# Detailed Example: SQL Record → MemoryNote → One-Sentence Index

## Original SQL Database Entry

```sql
-- Table: employees
SELECT * FROM employees WHERE id = 1;

id | name        | email            | role              | department_id | bio                                                                 | expertise                                                      | created_at          | updated_at
---|-------------|------------------|-------------------|---------------|----------------------------------------------------------------------|----------------------------------------------------------------|---------------------|-------------------
1  | Alice Chen  | alice@company.com| Senior ML Engineer| 2             | Alice specializes in transformer architectures and has led the...    | natural language processing, transformers, deep learning      | 2024-01-15 10:30:00 | 2024-01-20 14:22:00
```

**9 columns** with mixed data types (integer, text, timestamp)

---

## Step 1: SQL Record → ExternalRecord

**Function:** `GenericSQLConnector.fetch_records()`  
**File:** `archrag/adapters/connectors/sql_connector.py`

```python
# User specifies which columns to extract as text
text_columns = ["name", "role", "bio", "expertise"]

# Connector concatenates selected columns
text_parts = []
for col in text_columns:
    if col in row_dict and row_dict[col]:
        text_parts.append(str(row_dict[col]))
text_content = " ".join(text_parts)
# Result: "Alice Chen Senior ML Engineer Alice specializes in transformer architectures and has led the development of several production NLP systems. She has published papers on attention mechanisms and efficient inference. natural language processing, transformers, deep learning, attention mechanisms"

# Creates ExternalRecord
external_record = ExternalRecord(
    id="1",
    source_table="employees",
    source_database="mock_company",
    content={...},  # All 9 original fields preserved
    text_content=text_content,  # ← Concatenated text (multi-column → single string)
    metadata={"columns": [...], "primary_key": "id"},
    created_at="2024-01-15 10:30:00",
    updated_at="2024-01-20 14:22:00",
)
```

**Transformation:** 9 separate columns → 1 concatenated text string

---

## Step 2: ExternalRecord → Note Input Dict

**Function:** `ExternalRecord.to_note_input()`  
**File:** `archrag/domain/models.py:282-294`

```python
def to_note_input(self) -> dict[str, Any]:
    """Convert to input format for NoteConstructionService."""
    return {
        "content": self.text_content,  # ← The concatenated text from Step 1
        "category": self.source_table,   # "employees"
        "tags": [self.source_database, self.source_table],  # ["mock_company", "employees"]
        "metadata": {
            "source_id": self.id,
            "source_table": self.source_table,
            "source_database": self.source_database,
            "original_content": self.content,  # All 9 original fields preserved
        },
    }
```

**Result:**
```python
note_input = {
    "content": "Alice Chen Senior ML Engineer Alice specializes in transformer architectures...",
    "category": "employees",
    "tags": ["mock_company", "employees"],
    "metadata": {
        "source_id": "1",
        "source_table": "employees",
        "source_database": "mock_company",
        "original_content": {...}  # All 9 SQL fields
    }
}
```

---

## Step 3: Note Input → MemoryNote (with LLM-generated one-sentence context)

**Function:** `NoteConstructionService.build_note()`  
**File:** `archrag/services/note_construction.py:58-121`

### 3a. Extract Content
```python
content = self._extract_content(input_data)
# Result: "Alice Chen Senior ML Engineer Alice specializes in transformer architectures..."
```

### 3b. Call LLM to Generate Metadata (THE KEY STEP FOR ONE-SENTENCE)

**Function:** `NoteConstructionService._generate_metadata()`  
**Prompt:** `NOTE_CONSTRUCTION_PROMPT` from `archrag/prompts/note_construction.py:14-41`

```python
prompt = NOTE_CONSTRUCTION_PROMPT.format(content=content)
# The prompt asks for:
# - keywords: ["machine learning", "NLP", "transformers", ...]
# - context: "One sentence summarizing: Main topic/domain, Key arguments/points, Intended audience/purpose" ⭐
# - tags: ["engineering", "AI/ML", "personnel"]

result = self._llm.generate_json(prompt, system=NOTE_CONSTRUCTION_SYSTEM)
```

**LLM Response (JSON):**
```json
{
    "keywords": [
        "machine learning",
        "natural language processing",
        "transformer architectures",
        "attention mechanisms",
        "deep learning",
        "NLP systems"
    ],
    "context": "Senior ML engineer specializing in transformer architectures and NLP systems with expertise in attention mechanisms and efficient inference",  ⭐ ONE SENTENCE
    "tags": [
        "engineering",
        "AI/ML",
        "personnel",
        "research",
        "NLP"
    ]
}
```

### 3c. Compute Embedding
```python
embed_text = f"{content} {context} {' '.join(keywords)}"
# Combines: full content + one-sentence context + keywords
embedding = self._embedding.embed(embed_text)
# Result: [0.123, -0.456, 0.789, ...] (1536-dim vector)
```

### 3d. Create MemoryNote
```python
note = MemoryNote(
    content=content,  # Full concatenated text
    keywords=keywords,  # LLM-generated
    context=context,  # ⭐ ONE SENTENCE (LLM-generated)
    tags=tags,  # LLM-generated + user-provided
    category="employees",
    embedding=embedding,  # Vector for semantic search
)
```

**Final MemoryNote:**
```python
MemoryNote(
    id="a1b2c3d4e5f6",
    content="Alice Chen Senior ML Engineer Alice specializes in transformer architectures and has led the development of several production NLP systems. She has published papers on attention mechanisms and efficient inference. natural language processing, transformers, deep learning, attention mechanisms",
    context="Senior ML engineer specializing in transformer architectures and NLP systems with expertise in attention mechanisms and efficient inference",  ⭐ ONE SENTENCE
    keywords=["machine learning", "natural language processing", "transformer architectures", "attention mechanisms", "deep learning", "NLP systems"],
    tags=["engineering", "AI/ML", "personnel", "research", "NLP", "mock_company", "employees"],
    category="employees",
    embedding=[0.123, -0.456, 0.789, ...],  # 1536 dimensions
    links={"note_xyz789": "related expertise", "note_abc123": "same department"},
    timestamp="202401201422",
    retrieval_count=0
)
```

---

## Step 4: MemoryNote → Indexed (Saved)

**Function:** `MemoryNoteStorePort.save_note()`  
**File:** `archrag/adapters/stores/sqlite_memory_note.py`

```python
self._note_store.save_note(note)
# Saves to SQLite table: memory_notes
# Stores:
#   - Full content (for retrieval)
#   - Context (one sentence) ⭐ (for display/summaries)
#   - Keywords, tags, links
#   - Embedding vector (for semantic search)
```

---

## Summary: What Gets Indexed

| Field | Purpose | Used For |
|-------|--------|----------|
| **`content`** | Full concatenated text from all columns | Full-text retrieval, embedding computation |
| **`context`** ⭐ | **One sentence summary (LLM-generated)** | **Quick summaries, search result display, embedding computation** |
| **`keywords`** | LLM-extracted key concepts | Tagging, filtering, embedding computation |
| **`tags`** | Categorical classification | Filtering, organization |
| **`embedding`** | Vector representation | Semantic similarity search |

---

## The One-Sentence Generation

**Key Function:** `NoteConstructionService._generate_metadata()`  
**Prompt:** `NOTE_CONSTRUCTION_PROMPT` (lines 27-31)

The prompt explicitly asks for:
```
"context": 
    // One sentence summarizing:
    // - Main topic/domain
    // - Key arguments/points
    // - Intended audience/purpose
```

**Example transformations:**

| Input (9 columns) | One-Sentence Context (LLM-generated) |
|-------------------|--------------------------------------|
| Employee record with name, role, bio, expertise | "Senior ML engineer specializing in transformer architectures and NLP systems with expertise in attention mechanisms" |
| Research paper with title, abstract, authors, findings | "Neural collaborative filtering approach incorporating side information improves recommendation quality by 18% on cold-start items" |
| Meeting notes with agenda, discussion, action items | "Architecture review meeting discussing hierarchical clustering approach for RAG system with decisions on incremental updates and MCP integration" |

---

## Complete Function Call Chain

```
1. GenericSQLConnector.fetch_records()
   └─> Creates ExternalRecord with concatenated text_content

2. ExternalRecord.to_note_input()
   └─> Returns dict with "content" = text_content

3. DatabaseSyncService._process_record()
   └─> Calls NoteConstructionService.build_note(note_input)

4. NoteConstructionService.build_note()
   ├─> _extract_content() → Gets full text
   ├─> _generate_metadata() → Calls LLM ⭐
   │   └─> LLM generates: keywords, context (ONE SENTENCE), tags
   ├─> Computes embedding from: content + context + keywords
   ├─> Finds nearest notes for linking
   └─> Returns MemoryNote

5. DatabaseSyncService._process_record()
   └─> MemoryNoteStorePort.save_note(note)
       └─> Persists to SQLite with embedding vector
```

---

## Why One Sentence?

The one-sentence `context` field serves multiple purposes:

1. **Display in search results** - Quick summary without showing full content
2. **Embedding computation** - Part of the vector representation (content + context + keywords)
3. **Human readability** - Easy to scan and understand
4. **Storage efficiency** - Compact representation for large knowledge bases

The full `content` is still preserved for detailed retrieval, but the `context` provides the condensed, LLM-generated summary that makes the system efficient and user-friendly.
