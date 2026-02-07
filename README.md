# ArchRAG

An implementation of **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation** ([arXiv 2502.09891](https://arxiv.org/abs/2502.09891)) with an **A-Mem inspired MemoryNote system** ([arXiv 2502.12110](https://arxiv.org/abs/2502.12110)).

Built with a **hexagonal / ports & adapters** architecture — every external dependency (LLM, embedding model, database, vector index, clustering algorithm) is behind an abstract port and can be swapped via config.

## Unified Ingestion Pipeline

**All data sources flow through the same pipeline**, regardless of input format:

```
Input (JSONL, JSON, SQL, MongoDB, API)
  ↓
MemoryNote (LLM enrichment: keywords, context, tags, links)
  ↓
TextChunks (split for entity extraction)
  ↓
Knowledge Graph (entities, relations)
  ↓
Community Hierarchy (Leiden clustering)
  ↓
C-HNSW Index (hierarchical vector search)
```

This ensures consistent treatment of all data and enables full hierarchical traversal for retrieval.

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
| `archrag agent` | **Interactive agent for guided data ingestion** |
| `archrag agent --no-llm` | Form-based guided setup (no LLM required) |
| `archrag serve` | Start MCP server for AI client integration |
| `archrag connections` | List saved database connections |
| `archrag connect <name>` | Connect to a saved database and sync |

Add `-v` for debug logging, `-c path/to/config.yaml` for a custom config:

```bash
archrag -v -c my_config.yaml query "some question"
```

## Interactive Ingestion Agent

ArchRAG includes an interactive agent for guided data ingestion:

```bash
# Full conversational agent (requires LLM API key)
archrag agent

# Form-based guided setup (no LLM required)
archrag agent --no-llm
```

### Agent Features

- **Saved Connections**: Save database connection details (credentials, tables, preferences) with friendly names like "people" or "sales_db"
- **Automatic Reconnection**: Reference saved connections by name without re-entering credentials
- **Persistent Preferences**: Table selections and sync settings are remembered across sessions
- **Session History**: Conversation history is preserved for context
- **Sync Statistics**: Track sync history and record counts per connection

### Example Session

```
$ archrag agent

🤖 ArchRAG Ingestion Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Welcome to ArchRAG! I'm your Ingestion Assistant.
What would you like to do today?

You: I want to index my people database
Assistant: I found your 'people' connection from last week!
Last sync: 2024-02-01, 1,234 records indexed.

Should I sync new records now? (y/n)

You: yes

Syncing... Found 47 new records since last sync.
✅ Sync complete! 47 records added to the index.

You: quit

Goodbye! Your session has been saved.
```

## MCP Server

ArchRAG exposes its functionality via an MCP server for AI agent integration:

```bash
archrag serve
```

Configure Claude Desktop or Cursor to connect:
```json
{
  "mcpServers": {
    "archrag": {
      "command": "archrag",
      "args": ["serve"]
    }
  }
}
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

## Configuration

Add these sections to `config.yaml`:

```yaml
memory_note_store:
  adapter: sqlite
  path: data/archrag.db

memory:
  k_nearest: 10
  enable_evolution: true
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
