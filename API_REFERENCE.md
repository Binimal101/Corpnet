# ArchRAG API Reference

> **Base URL**: `http://localhost:8000`  
> **OpenAPI (Swagger UI)**: `http://localhost:8000/docs`  
> **ReDoc**: `http://localhost:8000/redoc`

### Endpoints at a Glance

| Method   | Path                   | Description                                   |
| -------- | ---------------------- | --------------------------------------------- |
| `GET`    | `/health`              | Liveness check                                |
| `GET`    | `/info`                | Database statistics & queue status            |
| `GET`    | `/visualize`           | Interactive hierarchy visualisation (HTML)     |
| `POST`   | `/index`               | Full rebuild from corpus file                 |
| `POST`   | `/query`               | Answer a question via hierarchical search     |
| `POST`   | `/search`              | Substring search across entities/chunks       |
| `POST`   | `/add`                 | Enqueue documents for batched indexing        |
| `POST`   | `/reindex`             | Flush the pending queue immediately           |
| `DELETE` | `/entities/{name}`     | Remove an entity by name                      |
| `DELETE` | `/clear`               | Wipe the entire database                      |

---

## Starting the Server

```bash
# From the project root
python -m archrag.api_server

# Or with uvicorn directly
uvicorn archrag.api_server:app --host 0.0.0.0 --port 8000

# Environment variables (optional)
ARCHRAG_CONFIG=config.yaml      # path to config YAML
ARCHRAG_HOST=0.0.0.0            # listen host
ARCHRAG_PORT=8000               # listen port
ARCHRAG_FLUSH_INTERVAL=180      # auto-flush interval in seconds
```

---

## Endpoints

### System

#### `GET /health`

Quick liveness check. Returns immediately.

**Response** `200 OK`

```json
{
  "status": "healthy"
}
```

---

#### `GET /info`

Database statistics and queue status.

**Response** `200 OK`

```json
{
  "entities": 109,
  "relations": 94,
  "chunks": 18,
  "hierarchy_levels": 5,
  "pending_in_queue": 0,
  "reindex_status": "idle"
}
```

| Field              | Type   | Description                                           |
| ------------------ | ------ | ----------------------------------------------------- |
| `entities`         | int    | Total entities in the knowledge graph                 |
| `relations`        | int    | Total relations in the knowledge graph                |
| `chunks`           | int    | Total text chunks stored                              |
| `hierarchy_levels` | int    | Number of community hierarchy levels                  |
| `pending_in_queue` | int    | Documents waiting to be indexed                       |
| `reindex_status`   | string | One of: `idle`, `running`, `done: …`, or `error: …`  |

---

### Indexing

#### `POST /index`

Full index rebuild from a corpus file. **Destructive** — wipes all existing data and rebuilds from scratch.

**Request Body**

```json
{
  "corpus_path": "corpus.jsonl"
}
```

| Field         | Type   | Required | Description                                      |
| ------------- | ------ | -------- | ------------------------------------------------ |
| `corpus_path` | string | ✅       | Path to JSONL or JSON array file on the server   |

**Response** `200 OK`

```json
{
  "message": "Indexing complete.",
  "entities": 95,
  "relations": 84,
  "chunks": 15,
  "hierarchy_levels": 5
}
```

**Error** `404 Not Found`

```json
{
  "detail": "Corpus file not found: missing.jsonl"
}
```

---

#### `POST /add`

Enqueue documents for batched indexing. Documents sit in a pending queue and are flushed automatically every 3 minutes, or immediately when `/reindex` is called.

**Request Body**

```json
{
  "documents": [
    {
      "text": "Max Planck originated quantum theory, which won him the Nobel Prize in Physics in 1918."
    },
    {
      "text": "Werner Heisenberg formulated the uncertainty principle in 1927.",
      "source": "wikipedia"
    },
    {
      "text": "Erwin Schrödinger developed his wave equation in 1926.",
      "source": "textbook",
      "id": 42
    }
  ]
}
```

| Field              | Type   | Required | Description                          |
| ------------------ | ------ | -------- | ------------------------------------ |
| `documents`        | array  | ✅       | List of document objects (min 1)     |
| `documents[].text` | string | ✅       | The document text to index           |
| `documents[].source` | string | ❌     | Optional source identifier           |
| `documents[].id`   | any    | ❌       | Optional document ID                 |
| *(extra fields)*   | any    | ❌       | Any additional fields are preserved  |

**Response** `200 OK`

```json
{
  "enqueued": 3,
  "pending": 3,
  "message": "Enqueued 3 document(s). Call /reindex to flush immediately."
}
```

---

#### `POST /reindex`

Immediately flush the pending document queue and trigger a full reindex. Runs in a background thread using blue/green snapshot swap — reads are never blocked.

**Request Body**: None

**Response** `200 OK` — Reindex started

```json
{
  "message": "Reindex started in background for 3 document(s).",
  "pending_before": 3,
  "status": "running"
}
```

**Response** `200 OK` — Queue empty

```json
{
  "message": "Queue is empty — nothing to reindex.",
  "pending_before": 0,
  "status": "done: 3 doc(s) reindexed. 112 entities, 98 relations, 21 chunks, 5 levels."
}
```

**Response** `200 OK` — Already running

```json
{
  "message": "A reindex is already in progress. Use /info to check status.",
  "pending_before": 2,
  "status": "running"
}
```

---

### Retrieval

#### `POST /query`

Answer a natural-language question using hierarchical search across the C-HNSW index followed by LLM-based adaptive filtering.

**Request Body**

```json
{
  "question": "What did Marie Curie discover?"
}
```

| Field      | Type   | Required | Description                          |
| ---------- | ------ | -------- | ------------------------------------ |
| `question` | string | ✅       | Natural-language question to answer  |

**Response** `200 OK`

```json
{
  "question": "What did Marie Curie discover?",
  "answer": "Marie Curie is renowned for her groundbreaking discoveries of two elements: polonium and radium. Her research focused on pitchblende ore..."
}
```

---

#### `POST /search`

Case-insensitive substring search across entities, chunks, or both.

**Request Body**

```json
{
  "query": "Einstein",
  "search_type": "all"
}
```

| Field         | Type   | Required | Default      | Description                                  |
| ------------- | ------ | -------- | ------------ | -------------------------------------------- |
| `query`       | string | ✅       | —            | Substring to search for (case-insensitive)   |
| `search_type` | string | ❌       | `"entities"` | `"entities"`, `"chunks"`, or `"all"`         |

**Response** `200 OK`

```json
{
  "entities": [
    {
      "id": "e442ea7caf37",
      "name": "Albert Einstein",
      "type": "PERSON",
      "description": "A theoretical physicist known for developing the theory of special relativity."
    }
  ],
  "chunks": [
    {
      "id": "f2f3265996c0",
      "text": "Albert Einstein was a theoretical physicist born in Ulm, Germany in 1879...",
      "source": ""
    }
  ]
}
```

When `search_type` is `"entities"`, the `chunks` field is `null` (and vice versa).

---

### Management

#### `DELETE /entities/{entity_name}`

Remove an entity by exact name from the knowledge graph. All relations involving this entity are also cascade-deleted.

**Path Parameter**

| Parameter     | Type   | Description                    |
| ------------- | ------ | ------------------------------ |
| `entity_name` | string | Exact name of entity to delete |

**Response** `200 OK`

```json
{
  "removed": true,
  "entity_name": "Max Planck",
  "message": "Removed entity 'Max Planck' and its relations."
}
```

**Error** `404 Not Found`

```json
{
  "detail": "Entity 'Unknown Person' not found in the knowledge graph."
}
```

---

#### `DELETE /clear`

**Destructive.** Wipe ALL entities, relations, chunks, communities, and vectors from the database. The knowledge graph will be completely empty afterwards. The hierarchy visualisation is regenerated automatically.

**Request Body**: _none_

**Response** `200 OK`

```json
{
  "message": "Database cleared.",
  "entities": 0,
  "relations": 0,
  "chunks": 0,
  "hierarchy_levels": 0
}
```

| Field              | Type   | Description                                 |
| ------------------ | ------ | ------------------------------------------- |
| `message`          | string | Status message                              |
| `entities`         | int    | Entity count after clear (should be 0)      |
| `relations`        | int    | Relation count after clear (should be 0)    |
| `chunks`           | int    | Chunk count after clear (should be 0)       |
| `hierarchy_levels` | int    | Hierarchy levels after clear (should be 0)  |

---

#### `GET /visualize`

Returns an interactive Plotly HTML page with three views of the community hierarchy:

- **🌳 DAG View** — Directed acyclic graph showing community containment
- **📦 Treemap** — Encapsulatory tiles you can click to drill into
- **🎯 Sunburst** — Concentric rings representing hierarchy levels

The visualisation is **auto-regenerated** after every mutation (`/index`, `/reindex`, `/entities/{name}`, `/clear`). If no hierarchy data exists, an empty-state page is returned.

**Response** `200 OK` — `text/html`

Open the response in a browser for the full interactive experience, or use `/docs` → "Try it out" → view in a new tab.

---

## Corpus File Format

The corpus file (used by `/index`) is either **JSONL** (one JSON object per line) or a **JSON array**:

### JSONL format

```jsonl
{"text": "Albert Einstein was a theoretical physicist born in Ulm, Germany in 1879."}
{"text": "Marie Curie discovered radium and polonium through her research on radioactivity."}
{"text": "Nikola Tesla invented the AC induction motor and transformer."}
```

### JSON array format

```json
[
  {"text": "Albert Einstein was a theoretical physicist born in Ulm, Germany in 1879."},
  {"text": "Marie Curie discovered radium and polonium."}
]
```

Each document must have at minimum a `text` field. Additional fields (`source`, `id`, `title`, etc.) are preserved as metadata.

---

## Pipeline Overview

When documents are indexed, they go through:

1. **Chunking** — Split text into overlapping segments (default 1200 chars, 100 overlap)
2. **KG Construction** — LLM extracts entities and relations from each chunk
3. **Embedding** — Entities are embedded using the configured model
4. **Hierarchical Clustering** — Leiden algorithm + LLM-generated community summaries
5. **C-HNSW Build** — Community-based Hierarchical Navigable Small World index

Queries go through:

1. **Hierarchical Search** — Multi-layer traversal of the C-HNSW index
2. **Adaptive Filtering** — LLM scores relevance and generates a grounded answer

---

## Error Handling

All errors return standard HTTP status codes with a JSON body:

```json
{
  "detail": "Human-readable error message"
}
```

| Status | Meaning                                |
| ------ | -------------------------------------- |
| 200    | Success                                |
| 404    | Resource not found (entity, file)      |
| 422    | Validation error (bad request body)    |
| 500    | Internal server error                  |

---

## Configuration

The server reads `config.yaml` (or the path in `ARCHRAG_CONFIG`). See `config.example.yaml` for all options.
