# ArchRAG

An implementation of **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation** ([arXiv 2502.09891](https://arxiv.org/abs/2502.09891)) with an **A-Mem inspired MemoryNote system** ([arXiv 2502.12110](https://arxiv.org/abs/2502.12110)).

Built with a **hexagonal / ports & adapters** architecture — every external dependency (LLM, embedding model, database, vector index, clustering algorithm) is behind an abstract port and can be swapped via config.

## Quick Start

### 1. Create the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e ".[all,dev]"
```

### 2. Set your API key

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
```

### 3. Configure

Copy the example config and adjust as needed:

```bash
cp config.example.yaml config.yaml
```

Default config uses OpenAI (`gpt-4o-mini` + `text-embedding-3-small`). See [config.example.yaml](config.example.yaml) for all adapter options (Ollama, SentenceTransformers, etc.).

### 4. Prepare a corpus

JSONL file, one document per line:

```jsonl
{"text": "Albert Einstein developed the theory of special relativity in 1905."}
{"text": "Marie Curie discovered polonium and radium."}
```

Also supports `{"title": "...", "context": "..."}` format from the original paper, and JSON arrays.

### 5. Run

```bash
# Build the full index (KG → hierarchical clustering → C-HNSW)
archrag index corpus.jsonl

# Ask a question
archrag query "What did Einstein win the Nobel Prize for?"
```

## CLI Reference

| Command | Description |
|---|---|
| `archrag index <corpus>` | Build full index from a JSONL / JSON corpus file |
| `archrag query "<question>"` | Answer a question using hierarchical search + adaptive filtering |
| `archrag search "<term>"` | Search entities by name (substring match) |
| `archrag search "<term>" -t chunks` | Search raw text chunks |
| `archrag search "<term>" -t all` | Search both entities and chunks |
| `archrag add <corpus>` | Add new documents to an existing index and re-index |
| `archrag remove "<entity name>"` | Delete an entity and its relations from the KG |
| `archrag info` | Show database stats and current configuration |

Add `-v` for debug logging, `-c path/to/config.yaml` for a custom config:

```bash
archrag -v -c my_config.yaml query "some question"
```

## MCP Server (FastMCP)

ArchRAG exposes its full functionality via an MCP (Model Context Protocol) server for AI agent integration.

### Running the MCP Server

```bash
# Run with stdio transport (for MCP clients)
python -m archrag.mcp_server

# Or use fastmcp CLI
fastmcp run archrag/mcp_server.py:mcp
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ARCHRAG_CONFIG` | `config.yaml` | Path to configuration file |
| `ARCHRAG_FLUSH_INTERVAL` | `180` | Seconds between auto-flush of document queue |

### MCP Tools Reference

#### Core RAG Tools

| Tool | Description |
|---|---|
| `index(corpus_path)` | Build full index from a corpus file (JSONL/JSON) |
| `query(question)` | Answer a question using hierarchical search + adaptive filtering |
| `search(query_str, search_type)` | Search entities/chunks by substring (`entities`, `chunks`, or `all`) |
| `add(documents)` | Enqueue documents for batched indexing (auto-flushes every 3 min) |
| `remove(entity_name)` | Delete an entity and its relations from the knowledge graph |
| `reindex()` | Immediately flush pending documents and rebuild index |
| `info()` | Show database statistics and queue status |

#### MemoryNote Tools (A-Mem inspired)

These tools implement the Zettelkasten-inspired memory system from the A-Mem paper:

| Tool | Description |
|---|---|
| `add_note(content, category, tags, keywords, enable_linking, enable_evolution, add_to_kg)` | Add enriched memory note with LLM-generated metadata |
| `get_note(note_id)` | Retrieve a memory note by ID |
| `get_related_notes(note_id)` | Get notes linked to a given note |
| `search_notes(query_str, k)` | Semantic search for notes by content similarity |
| `delete_note(note_id)` | Delete a memory note |

### MCP Tool Examples

#### Adding a Memory Note

```python
# Via MCP client
result = await client.call_tool("add_note", {
    "content": "The transformer architecture revolutionized NLP by using self-attention.",
    "category": "technology",
    "tags": ["AI", "deep learning"],
    "enable_linking": True,
    "enable_evolution": True
})
# Returns: Created note ID, auto-generated keywords, context, tags, and links
```

#### Querying the Knowledge Graph

```python
result = await client.call_tool("query", {
    "question": "What is the transformer architecture?"
})
# Returns: Answer generated using hierarchical search + adaptive filtering
```

#### Semantic Note Search

```python
result = await client.call_tool("search_notes", {
    "query_str": "machine learning algorithms",
    "k": 5
})
# Returns: Top 5 semantically similar notes
```

## Testing Memory Notes

Sample data and a test script are provided:

```bash
# Run the memory notes test script
python scripts/test_memory_notes.py

# Sample notes are in archrag/dataset/sample_notes.json
```

The test script demonstrates:
1. Adding notes with LLM enrichment
2. Retrieving notes by ID
3. Finding related notes via links
4. Semantic search across notes
5. Deleting notes

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

```
CLI (click) / MCP Server (FastMCP)
 │
 ▼
Orchestrator
 ├── KG Construction Service
 ├── Hierarchical Clustering Service  (Algorithm 1)
 ├── C-HNSW Build Service             (Algorithm 3)
 ├── Hierarchical Search Service      (Algorithm 2)
 ├── Adaptive Filtering Service       (Equations 1 & 2)
 └── Note Construction Service        (A-Mem inspired)
      │
      ▼
   7 Ports (ABCs)
      │
      ▼
   Swappable Adapters
```

### Ports & Adapters

| Port | Adapters |
|---|---|
| **EmbeddingPort** | SentenceTransformers, OpenAI, Ollama |
| **LLMPort** | OpenAI, Ollama |
| **GraphStorePort** | SQLite, In-Memory |
| **DocumentStorePort** | SQLite, In-Memory |
| **VectorIndexPort** | NumPy (brute-force cosine) |
| **ClusteringPort** | Leiden (via igraph + leidenalg) |
| **MemoryNoteStorePort** | SQLite |

## Project Structure

```
archrag/
├── domain/models.py          # Pure dataclasses (Entity, Relation, Community, MemoryNote, etc.)
├── ports/                    # 7 abstract base classes
│   └── memory_note_store.py  # MemoryNote persistence port
├── adapters/
│   ├── embeddings/           # SentenceTransformer, OpenAI, Ollama
│   ├── llms/                 # OpenAI, Ollama
│   ├── stores/               # SQLite & in-memory (graph + document + memory notes)
│   │   └── sqlite_memory_note.py
│   ├── indexes/              # NumPy vector index
│   └── clustering/           # Leiden
├── services/
│   ├── orchestrator.py       # Main entry point
│   ├── note_construction.py  # A-Mem inspired note enrichment
│   └── ...                   # KG, clustering, C-HNSW, search, filtering
├── prompts/
│   ├── note_construction.py  # LLM prompts for note enrichment & linking
│   └── ...                   # Extraction, summarization, filtering
├── mcp_server.py             # FastMCP server with all tools
├── config.py                 # YAML config + adapter factory + dotenv loading
└── cli.py                    # Click CLI entry point
scripts/
└── test_memory_notes.py      # Test script for memory note functionality
tests/                        # Unit tests with mock ports
```

## Configuration

Add these sections to your `config.yaml` for memory note settings:

```yaml
# Memory note store configuration
memory_note_store:
  adapter: sqlite
  path: data/archrag.db

# Memory note construction settings
memory:
  k_nearest: 10           # Number of neighbors for link generation
  enable_evolution: true  # Whether to evolve related notes on insert
```

## Tests

```bash
python -m pytest tests/ -v
```

## Paper References

> **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation**
> [arXiv:2502.09891](https://arxiv.org/abs/2502.09891)

> **A-Mem: Agentic Memory for LLM Agents**
> [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
