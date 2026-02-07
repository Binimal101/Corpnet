# Consumer API Reference

The **Consumer** is a thin read-only FastAPI gateway that forwards
every request to the Producer over HTTP. It never touches the
database directly.

## Configuration

Settings live in `consumer/config.yaml`:

```yaml
# Where the producer is running (include base_url if the producer uses one)
producer_url: http://localhost:8000

server:
  host: 0.0.0.0       # bind address
  port: 8001           # listen port
  base_url: /          # path prefix for all endpoints
```

`producer_url` must include the producer's `base_url` if one is set.
For example, if the producer uses `base_url: /api/v1`:

```yaml
producer_url: http://localhost:8000/api/v1
```

## Starting the server

```bash
# From project root
python -m consumer.mcp_server          # uses host/port from config.yaml
uvicorn consumer.mcp_server:app --port 8001   # manual override
```

Interactive docs are available at `http://localhost:8001/docs` (Swagger UI).

> **Prerequisite:** The producer must be running for `/query`, `/search`,
> and `/info` to return data. `/health` always responds even if the
> producer is down.

---

## Endpoints

### `GET /health`

Local health check with producer reachability probe.

**Request body:** _none_

**Response (200) — producer reachable:**

```json
{
  "status": "healthy",
  "producer": "reachable",
  "producer_url": "http://localhost:8000"
}
```

**Response (200) — producer unreachable:**

```json
{
  "status": "healthy",
  "producer": "unreachable",
  "producer_url": "http://localhost:8000"
}
```

The consumer itself is always `"healthy"` — only the `producer` field
indicates whether the upstream is reachable.

---

### `POST /query`

Forward a natural-language question to the producer's `POST /query`.

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

**Response (502) — producer unreachable:**

```json
{
  "detail": "Cannot reach producer at http://localhost:8000"
}
```

---

### `POST /search`

Forward a substring search to the producer's `POST /search`.

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

**Response (200):**

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

**Response (502) — producer unreachable:**

```json
{
  "detail": "Cannot reach producer at http://localhost:8000"
}
```

---

### `GET /info`

Forward to the producer's `GET /stats` and return the result.

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
| `pending`          | int    | Documents waiting in the producer's ingestion queue.     |
| `reindex_status`   | string | One of `"idle"`, `"running"`, `"done: …"`, `"error: …"`. |

**Response (502) — producer unreachable:**

```json
{
  "detail": "Cannot reach producer at http://localhost:8000"
}
```
