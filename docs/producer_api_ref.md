# Producer API Reference

The **Producer** is the write-side FastAPI server that owns the ArchRAG
database and exposes the full pipeline over HTTP.

## Configuration

Settings live in `producer/config.yaml` under the `server` key:

```yaml
server:
  host: 0.0.0.0       # bind address
  port: 8000           # listen port
  base_url: /          # path prefix for all endpoints (e.g. /api/v1)
```

All endpoints below are relative to `base_url`.

## Starting the server

```bash
# From project root
python -m producer.api            # uses host/port/base_url from config.yaml
uvicorn producer.api:app --port 8000   # manual override
```

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI).

---

## Endpoints

### `POST /index`

Wipe all existing data and rebuild the full index from a corpus file.

**Request body:**

```json
{
  "corpus_path": "corpus.jsonl"
}
```

| Field         | Type   | Required | Description                              |
|---------------|--------|----------|------------------------------------------|
| `corpus_path` | string | yes      | Path to a JSONL or JSON-array file on disk. |

**Response (200):**

```json
{
  "status": "ok",
  "message": "Indexing complete. 42 entities, 38 relations, 15 chunks, 3 hierarchy levels.",
  "entities": 42,
  "relations": 38,
  "chunks": 15,
  "hierarchy_levels": 3
}
```

---

### `POST /add`

Enqueue documents for batched indexing. Documents sit in a pending queue
and are flushed every 3 minutes, or when `POST /reindex` is called.

**Request body:**

```json
{
  "documents": [
    {
      "content": "Ada Lovelace was a mathematician...",
      "category": "biography",
      "tags": "math,computing",
      "keywords": "Ada Lovelace,Babbage"
    }
  ]
}
```

| Field       | Type          | Required | Description                                           |
|-------------|---------------|----------|-------------------------------------------------------|
| `documents` | array[object] | yes      | List of document dicts, each with at least a `content` key. |

**Response (200):**

```json
{
  "status": "ok",
  "enqueued": 1,
  "pending": 1,
  "message": "Enqueued 1 doc(s). 1 total pending. Call POST /reindex or wait for auto-flush."
}
```

---

### `DELETE /remove`

Remove an entity (and its relations) from the knowledge graph.

**Request body:**

```json
{
  "entity_name": "Ada Lovelace"
}
```

| Field         | Type   | Required | Description                |
|---------------|--------|----------|----------------------------|
| `entity_name` | string | yes      | Exact entity name to delete. |

**Response (200) — found:**

```json
{
  "status": "ok",
  "message": "Removed entity 'Ada Lovelace'."
}
```

**Response (200) — not found:**

```json
{
  "status": "not_found",
  "message": "Entity 'Ada Lovelace' not found."
}
```

---

### `POST /reindex`

Immediately flush the pending document queue in the background.
Returns instantly — poll `GET /stats` to check progress.

**Request body:** _none_

**Response (200) — started:**

```json
{
  "status": "started",
  "pending": 3,
  "message": "Reindex launched for 3 doc(s). Poll GET /stats for progress."
}
```

**Response (200) — queue empty:**

```json
{
  "status": "empty",
  "message": "Queue empty — nothing to reindex. Last: idle"
}
```

**Response (200) — already running:**

```json
{
  "status": "already_running",
  "message": "A reindex is already in progress."
}
```

---

### `GET /stats`

Return current database statistics and queue status.

**Request body:** _none_

**Response (200):**

```json
{
  "entities": 42,
  "relations": 38,
  "chunks": 15,
  "hierarchy_levels": 3,
  "pending": 0,
  "reindex_status": "idle"
}
```

| Field              | Type   | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| `entities`         | int    | Total entity count in the knowledge graph.               |
| `relations`        | int    | Total relation count.                                    |
| `chunks`           | int    | Total text chunk count.                                  |
| `hierarchy_levels` | int    | Depth of the community hierarchy.                        |
| `pending`          | int    | Documents waiting in the ingestion queue.                |
| `reindex_status`   | string | One of `"idle"`, `"running"`, `"done: …"`, `"error: …"`. |

---

### `POST /query`

Answer a natural-language question using hierarchical search + adaptive
filtering.

**Request body:**

```json
{
  "question": "What contributions did Ada Lovelace make to computing?"
}
```

| Field      | Type   | Required | Description                       |
|------------|--------|----------|-----------------------------------|
| `question` | string | yes      | Natural-language question to answer. |

**Response (200):**

```json
{
  "answer": "Ada Lovelace is widely regarded as the first computer programmer..."
}
```

---

### `POST /search`

Search the knowledge graph by substring.

**Request body:**

```json
{
  "query_str": "lovelace",
  "search_type": "all"
}
```

| Field         | Type   | Required | Default      | Description                                        |
|---------------|--------|----------|--------------|----------------------------------------------------|
| `query_str`   | string | yes      | —            | Case-insensitive substring to search for.          |
| `search_type` | string | no       | `"entities"` | What to search: `"entities"`, `"chunks"`, or `"all"`. |

**Response (200) — search_type `"all"`:**

```json
{
  "entities": [
    {
      "id": "a1b2c3d4e5f6",
      "name": "Ada Lovelace",
      "type": "PERSON",
      "description": "English mathematician and writer..."
    }
  ],
  "chunks": [
    {
      "id": "f6e5d4c3b2a1",
      "content": "Ada Lovelace was a mathematician who...",
      "category": "biography",
      "tags": "math,computing",
      "keywords": "Ada Lovelace,Babbage"
    }
  ]
}
```

Only the requested key(s) appear in the response — `"entities"` when
`search_type` is `"entities"`, `"chunks"` when `"chunks"`, both when `"all"`.
