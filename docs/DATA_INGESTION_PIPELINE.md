# ArchRAG Data Ingestion Pipeline

> **Purpose**: Comprehensive documentation of the data ingestion pipeline for LLM-based understanding and integration with external systems.

---

## 1. Pipeline Overview

All data ingested into ArchRAG flows through a **unified pipeline** regardless of source format:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNIFIED INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT SOURCES                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│   │  JSONL   │  │   JSON   │  │  SQLite  │  │ Postgres │  │ MongoDB  │     │
│   │  Files   │  │  Files   │  │    DB    │  │    DB    │  │    DB    │     │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│        │             │             │             │             │            │
│        └─────────────┴─────────────┴──────┬──────┴─────────────┘            │
│                                           ▼                                 │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    1. MemoryNote Creation                          │    │
│   │    ┌─────────────────────────────────────────────────────────┐    │    │
│   │    │  • Extract content from input                           │    │    │
│   │    │  • LLM generates keywords & tags                        │    │    │
│   │    │  • Compute structured embedding                         │    │    │
│   │    │  • Store as MemoryNote                                  │    │    │
│   │    └─────────────────────────────────────────────────────────┘    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                           │                                 │
│                                           ▼                                 │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    2. Text Chunking                                │    │
│   │    ┌─────────────────────────────────────────────────────────┐    │    │
│   │    │  • Split content into overlapping chunks                │    │    │
│   │    │  • Attach note metadata to each chunk                   │    │    │
│   │    │  • Store chunks in DocumentStore                        │    │    │
│   │    └─────────────────────────────────────────────────────────┘    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                           │                                 │
│                                           ▼                                 │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    3. Knowledge Graph Extraction                   │    │
│   │    ┌─────────────────────────────────────────────────────────┐    │    │
│   │    │  • LLM extracts entities from each chunk                │    │    │
│   │    │  • LLM extracts relations between entities              │    │    │
│   │    │  • Compute entity embeddings                            │    │    │
│   │    │  • Merge duplicate entities                             │    │    │
│   │    │  • Store in GraphStore                                  │    │    │
│   │    └─────────────────────────────────────────────────────────┘    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                           │                                 │
│                                           ▼                                 │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    4. Community Hierarchy                          │    │
│   │    (Built separately via Leiden clustering + C-HNSW index)        │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Data Models

### 2.1 MemoryNote

The fundamental unit of information storage. Inspired by the A-Mem paper (arXiv 2502.12110).

```python
@dataclass
class MemoryNote:
    # Core content
    content: str                           # Original text content
    id: str                                # UUID, auto-generated
    
    # LLM-generated metadata
    keywords: list[str]                    # Key concepts extracted by LLM
    tags: list[str]                        # Categorical tags from LLM
    
    # User-provided metadata
    category: str                          # Optional classification category
    
    # Tracking
    last_updated: str | None               # Timestamp of last modification
    retrieval_count: int                   # Number of times retrieved
    
    # Vector representation
    embedding: list[float] | None          # Dense vector for similarity search
    embedding_model: str                   # Model used (e.g., "text-embedding-3-small")
```

**Key Methods**:
- `to_document()`: Converts to dict format for KG pipeline
- `increment_retrieval()`: Updates retrieval stats and timestamp

### 2.2 Embedding Format

Embeddings are computed from a **structured string** with clear delimiters:

```
content:{content}|last_updated:{timestamp}|tags:{tag1,tag2}|keywords:{kw1,kw2}|category:{cat}|retrieval_count:{n}
```

**Example**:
```
content:Einstein developed the theory of relativity.|last_updated:202402071200|tags:physics,science,history|keywords:Einstein,relativity,theory|category:scientific_discovery|retrieval_count:0
```

This structured format:
- Preserves semantic meaning of each field
- Enables field-aware similarity matching
- Makes the embedding content auditable

### 2.3 TextChunk

Chunked content for entity extraction:

```python
@dataclass
class TextChunk:
    text: str                              # Chunk text content
    id: str                                # UUID, auto-generated
    source_doc: str                        # Parent document/note ID
    metadata: dict[str, Any]               # Includes note_id, category, tags, keywords
```

### 2.4 ExternalRecord

Records extracted from external databases:

```python
@dataclass
class ExternalRecord:
    id: str                                # Primary key from source
    source_table: str                      # Table/collection name
    source_database: str                   # Database identifier
    content: dict[str, Any]                # All fields as key-value
    text_content: str                      # Concatenated text for indexing
    metadata: dict[str, Any]               # Schema info
    created_at: str | None                 # Source timestamp
    updated_at: str | None                 # Source timestamp
```

**Conversion to MemoryNote Input**:
```python
def to_note_input(self) -> dict[str, Any]:
    return {
        "content": self.text_content,      # Concatenated text
        "category": self.source_table,     # Table name as category
        "tags": [self.source_database, self.source_table],
        "metadata": {
            "source_id": self.id,
            "source_table": self.source_table,
            "source_database": self.source_database,
            "original_content": self.content,  # Preserved
        },
    }
```

---

## 3. Service Architecture

### 3.1 Service Hierarchy

```
ArchRAGOrchestrator
    │
    ├── UnifiedIngestionPipeline       # Main entry point for all ingestion
    │       │
    │       ├── NoteConstructionService    # MemoryNote creation with LLM
    │       │       │
    │       │       ├── LLMPort            # Generates keywords, tags
    │       │       └── EmbeddingPort      # Computes embeddings
    │       │
    │       ├── MemoryNoteStorePort        # Persists MemoryNotes
    │       ├── DocumentStorePort          # Persists TextChunks
    │       └── GraphStorePort             # Persists entities/relations
    │
    └── DatabaseSyncService            # External database integration
            │
            ├── ExternalDatabaseConnectorPort  # SQL/NoSQL adapter
            └── UnifiedIngestionPipeline       # Routes records to pipeline
```

### 3.2 Port/Adapter Pattern

All external dependencies are abstracted behind ports:

| Port | Purpose | Adapters |
|------|---------|----------|
| `LLMPort` | Text generation, JSON extraction | OpenAI, Ollama |
| `EmbeddingPort` | Text → vector conversion | OpenAI, SentenceTransformer, Ollama |
| `MemoryNoteStorePort` | MemoryNote persistence | SQLite |
| `DocumentStorePort` | Chunk persistence, metadata | SQLite, InMemory |
| `GraphStorePort` | Entity/relation storage | SQLite |
| `ExternalDatabaseConnectorPort` | User database access | SQL (SQLAlchemy), NoSQL (MongoDB) |

---

## 4. Ingestion Entry Points

### 4.1 UnifiedIngestionPipeline Methods

```python
class UnifiedIngestionPipeline:
    
    def ingest_single(
        self,
        input_data: dict[str, Any],
        *,
        skip_kg: bool = False,
    ) -> MemoryNote:
        """Ingest one document through the full pipeline."""
    
    def ingest_batch(
        self,
        items: list[dict[str, Any]],
        *,
        skip_kg: bool = False,
    ) -> list[MemoryNote]:
        """Ingest multiple documents."""
    
    def ingest_file(
        self,
        path: str,
    ) -> list[MemoryNote]:
        """Ingest from JSONL or JSON file."""
    
    def ingest_from_external_record(
        self,
        record: ExternalRecord,
    ) -> MemoryNote:
        """Ingest a database record."""
```

### 4.2 Input Format

The pipeline accepts dictionaries with flexible field names:

```python
# Primary content field (required - one of these)
input_data = {
    "content": "...",   # Preferred
    # OR
    "text": "...",      # Alternative
    # OR
    "context": "...",   # Alternative
    
    # Optional metadata (will be merged with LLM-generated)
    "category": "...",
    "tags": ["tag1", "tag2"],
    "keywords": ["kw1", "kw2"],
    "last_updated": "202402071200",
    "retrieval_count": 0,
}
```

---

## 5. Pipeline Steps in Detail

### 5.1 Step 1: MemoryNote Creation

**Service**: `NoteConstructionService`

```python
def build_note(self, input_data: dict[str, Any]) -> MemoryNote:
    # 1. Extract content from input
    content = self._extract_content(input_data)
    
    # 2. LLM generates keywords and tags
    keywords, tags = self._generate_metadata(content)
    
    # 3. Merge with user-provided metadata
    if input_data.get("keywords"):
        keywords = list(set(keywords + input_data["keywords"]))
    if input_data.get("tags"):
        tags = list(set(tags + input_data["tags"]))
    
    # 4. Build structured embedding text
    embed_text = self._build_embedding_text(
        content=content,
        last_updated=last_updated,
        tags=tags,
        keywords=keywords,
        category=category,
        retrieval_count=retrieval_count,
    )
    
    # 5. Compute embedding
    embedding = self._embedding.embed(embed_text)
    embedding_model = self._embedding.model_name()
    
    # 6. Create MemoryNote
    return MemoryNote(
        content=content,
        keywords=keywords,
        tags=tags,
        category=category,
        embedding=embedding,
        embedding_model=embedding_model,
        ...
    )
```

**LLM Prompt for Metadata Extraction**:

```
System: You are an AI memory system that analyzes content and extracts 
structured metadata. Your goal is to create rich, searchable 
representations of information for later retrieval.

Prompt: Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, key concepts)
2. Creating relevant categorical tags

Format the response as a JSON object:
{
    "keywords": [...],
    "tags": [...]
}

Content for analysis:
{content}
```

### 5.2 Step 2: Text Chunking

**Service**: `UnifiedIngestionPipeline._chunk_note()`

```python
def _chunk_note(self, note: MemoryNote) -> list[TextChunk]:
    chunks = []
    content = note.content
    start = 0
    chunk_idx = 0
    
    while start < len(content):
        end = start + self._chunk_size  # Default: 1200 chars
        chunk_text = content[start:end]
        
        chunk = TextChunk(
            text=chunk_text,
            source_doc=note.id,
            metadata={
                "note_id": note.id,
                "note_category": note.category,
                "note_tags": note.tags,
                "note_keywords": note.keywords,
                "chunk_index": chunk_idx,
            },
        )
        chunks.append(chunk)
        
        start += self._chunk_size - self._chunk_overlap  # Default overlap: 100
        chunk_idx += 1
    
    return chunks
```

**Chunking Parameters**:
- `chunk_size`: 1200 characters (default)
- `chunk_overlap`: 100 characters (default)

### 5.3 Step 3: Knowledge Graph Extraction

**Service**: `UnifiedIngestionPipeline._process_note_to_kg()`

For each chunk, LLM extracts entities and relations:

```python
def _extract_entities_relations(self, chunk: TextChunk) -> dict:
    # LLM call to extract structured data
    result = self._llm.generate_json(
        ENTITY_RELATION_EXTRACTION_PROMPT.format(text=chunk.text),
        system=ENTITY_RELATION_EXTRACTION_SYSTEM
    )
    return result  # {"entities": [...], "relations": [...]}
```

**Entity Processing**:
1. Parse entity data from LLM response
2. Merge duplicate entities (matched by lowercase name)
3. Compute embeddings for all entities
4. Persist to GraphStore

**Relation Processing**:
1. Validate source and target entities exist
2. Create Relation objects linking entity IDs
3. Persist to GraphStore

---

## 6. External Database Integration

### 6.1 DatabaseSyncService

Orchestrates syncing from external databases:

```python
class DatabaseSyncService:
    def __init__(
        self,
        connector: ExternalDatabaseConnectorPort,
        ingestion_pipeline: UnifiedIngestionPipeline,
        doc_store: DocumentStorePort,
        *,
        batch_size: int = 100,
    ):
        ...
    
    def full_sync(
        self,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
    ) -> SyncResult:
        """Sync all records from specified tables."""
    
    def incremental_sync(
        self,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
    ) -> SyncResult:
        """Sync only new/changed records since last sync."""
```

### 6.2 Sync Flow

```
ExternalDatabaseConnectorPort
        │
        │ 1. fetch_records(table, text_columns, since_timestamp=...)
        ▼
┌──────────────────────────────────┐
│       ExternalRecord             │
│  ┌────────────────────────────┐  │
│  │ id: "user_123"             │  │
│  │ source_table: "users"      │  │
│  │ text_content: "John Doe,   │  │
│  │   Software Engineer..."    │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
        │
        │ 2. record.to_note_input()
        ▼
┌──────────────────────────────────┐
│       dict[str, Any]             │
│  ┌────────────────────────────┐  │
│  │ content: "John Doe..."     │  │
│  │ category: "users"          │  │
│  │ tags: ["company_db", ...]  │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
        │
        │ 3. UnifiedIngestionPipeline.ingest_single(...)
        ▼
    [Standard Pipeline: MemoryNote → Chunks → KG]
```

### 6.3 ExternalDatabaseConnectorPort Interface

```python
class ExternalDatabaseConnectorPort(ABC):
    # Connection
    def connect(self, connection_config: dict) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    
    # Schema Discovery
    def list_databases(self) -> list[str]: ...
    def list_tables(self, database: str = None) -> list[str]: ...
    def get_table_schema(self, table: str, database: str = None) -> TableSchema: ...
    def discover_relationships(self, database: str = None) -> list[RelationshipInfo]: ...
    
    # Data Extraction
    def fetch_records(
        self,
        table: str,
        text_columns: list[str],
        *,
        limit: int = None,
        offset: int = 0,
        since_timestamp: str = None,
        since_id: str = None,
        timestamp_column: str = None,
        id_column: str = None,
        order_by: str = None,
    ) -> list[ExternalRecord]: ...
    
    def count_records(self, table: str, ...) -> int: ...
    
    # Metadata
    def get_connector_type(self) -> str: ...  # "sql" or "nosql"
    def get_connector_id(self) -> str: ...
    def get_connection_info(self) -> dict: ...  # Sanitized (no passwords)
```

---

## 7. Storage Interfaces

### 7.1 MemoryNoteStorePort

```python
class MemoryNoteStorePort(ABC):
    # CRUD
    def save_note(self, note: MemoryNote) -> None: ...
    def get_note(self, note_id: str) -> MemoryNote | None: ...
    def get_all_notes(self) -> list[MemoryNote]: ...
    def update_note(self, note: MemoryNote) -> None: ...
    def delete_note(self, note_id: str) -> None: ...
    
    # Similarity Search
    def get_nearest_notes(
        self,
        embedding: list[float],
        k: int,
        exclude_ids: list[str] = None,
    ) -> list[MemoryNote]: ...
    
    # Tag/Keyword Search
    def search_by_tags(self, tags: list[str]) -> list[MemoryNote]: ...
    def search_by_keywords(self, keywords: list[str]) -> list[MemoryNote]: ...
    
    # Lifecycle
    def clear(self) -> None: ...
    def count(self) -> int: ...
```

### 7.2 SQLite Schema for MemoryNotes

```sql
CREATE TABLE memory_notes (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    last_updated      TEXT,
    keywords          TEXT NOT NULL DEFAULT '[]',    -- JSON array
    tags              TEXT NOT NULL DEFAULT '[]',    -- JSON array
    category          TEXT NOT NULL DEFAULT '',
    retrieval_count   INTEGER NOT NULL DEFAULT 0,
    embedding         TEXT,                          -- JSON array of floats
    embedding_model   TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_notes_category ON memory_notes(category);
CREATE INDEX idx_notes_last_updated ON memory_notes(last_updated);
```

---

## 8. Integration Points

### 8.1 Adding a New Input Source

To add a new data source (e.g., REST API, message queue):

1. **Create an adapter** implementing `ExternalDatabaseConnectorPort`
2. **Register** in `config.py` factory function
3. **Use** via `DatabaseSyncService` or directly call `UnifiedIngestionPipeline`

Example for a REST API:

```python
class RestAPIConnector(ExternalDatabaseConnectorPort):
    def connect(self, config):
        self._base_url = config["base_url"]
        self._api_key = config["api_key"]
    
    def fetch_records(self, table, text_columns, **kwargs):
        response = requests.get(f"{self._base_url}/{table}", ...)
        return [
            ExternalRecord(
                id=item["id"],
                source_table=table,
                source_database=self._base_url,
                content=item,
                text_content=" ".join(str(item[c]) for c in text_columns),
            )
            for item in response.json()
        ]
```

### 8.2 Customizing the Pipeline

To modify pipeline behavior:

```python
# Skip KG extraction (only create MemoryNotes)
note = pipeline.ingest_single(data, skip_kg=True)

# Custom chunk size
pipeline = UnifiedIngestionPipeline(
    ...,
    chunk_size=2000,
    chunk_overlap=200,
)

# Custom LLM prompts (modify archrag/prompts/note_construction.py)
```

### 8.3 Accessing Ingested Data

```python
# Get all MemoryNotes
notes = memory_note_store.get_all_notes()

# Semantic search for notes
similar_notes = memory_note_store.get_nearest_notes(query_embedding, k=10)

# Get the knowledge graph
kg = unified_pipeline.get_kg_from_notes()
entities = graph_store.get_all_entities()
relations = graph_store.get_all_relations()
```

---

## 9. Configuration

### 9.1 config.yaml Structure

```yaml
embedding:
  adapter: openai
  model: text-embedding-3-small
  dimension: 1536

llm:
  adapter: openai
  model: gpt-4o-mini
  temperature: 0.0

memory_note_store:
  adapter: sqlite
  path: data/archrag.db

document_store:
  adapter: sqlite
  path: data/archrag.db

graph_store:
  adapter: sqlite
  path: data/archrag.db

indexing:
  chunk_size: 1200
  chunk_overlap: 100
```

---

## 10. Summary

| Component | Responsibility |
|-----------|----------------|
| `UnifiedIngestionPipeline` | Routes all input through consistent pipeline |
| `NoteConstructionService` | Creates MemoryNotes with LLM enrichment |
| `DatabaseSyncService` | Syncs external databases |
| `MemoryNote` | Core data model with structured embeddings |
| `ExternalRecord` | Database record representation |
| `TextChunk` | Chunked content for entity extraction |

**Key Guarantees**:
1. All data becomes a MemoryNote before further processing
2. All MemoryNotes get LLM-generated keywords and tags
3. All embeddings use a consistent structured format
4. All data flows through chunking and KG extraction (unless `skip_kg=True`)
5. Provenance is preserved via metadata linking
