# Corpnet

**Represent organisational data flows as a monolithic hierarchical knowledge network for enhanced recall.**

Corpnet ingests an organisation's unstructured documents, extracts a knowledge graph, clusters it hierarchically, and exposes the result through a **producer / consumer** HTTP architecture so every team and tool across the organisation can query the same living knowledge base.

Under the hood the pipeline is an implementation of **ArchRAG — Attributed Community-based Hierarchical Retrieval-Augmented Generation** ([arXiv 2502.09891](https://arxiv.org/abs/2502.09891)), with a **hexagonal / ports-and-adapters** core: every external dependency (LLM, embedding model, database, vector index, clustering algorithm) sits behind an abstract port and can be swapped via YAML config.

---

## System Model

```
                ┌──────────────────────┐
                │      Producer        │   one per organisation
                │  (FastAPI :8000)     │
                │                      │
                │  corpus ──► KG ──►   │
                │  clusters ──► C-HNSW │
                │  SQLite WAL DB       │
                └──────┬───────────────┘
                       │  HTTP
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │Consumer A│ │Consumer B│ │Consumer…│   one or more
     │ :8001    │ │ :8002    │ │         │
     └─────────┘ └─────────┘ └─────────┘
```

* **Producer** — the single authoritative write node.  Owns the SQLite database, runs the full ArchRAG pipeline (KG construction → hierarchical Leiden clustering → C-HNSW index build), and serves both write and read endpoints.
* **Consumer(s)** — lightweight read-only HTTP proxies.  Each consumer forwards `/query`, `/search`, and `/info` requests to the producer and returns the response.  Consumers are stateless and horizontally scalable.

---

## Quick Start

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all,dev]"
```

### 2. Set your API key

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
```

### 3. Configure

```powershell
copy config.example.yaml config.yaml
```

Default config uses OpenAI (`gpt-4o-mini` + `text-embedding-3-small`).  
See [config.example.yaml](config.example.yaml) for all adapter options (Ollama, SentenceTransformers, etc.).

Producer-specific and consumer-specific settings live in their own config files:

| File | Purpose |
|---|---|
| `producer/config.yaml` | Host, port, base URL, data dir, all adapter choices |
| `consumer/config.yaml` | Host, port, base URL, upstream `producer_url` |

### 4. Prepare a corpus

JSONL file, one document per line:

```jsonl
{"text": "Albert Einstein developed the theory of special relativity in 1905."}
{"text": "Marie Curie discovered polonium and radium."}
```

Also supports `{"title": "...", "context": "..."}` format from the original paper, and JSON arrays.

### 5. Run the Producer

```powershell
python -m producer.api          # reads host/port from producer/config.yaml
# or
uvicorn producer.api:app --host 0.0.0.0 --port 8000
```

### 6. Run a Consumer

```powershell
python -m consumer.mcp_server   # reads host/port from consumer/config.yaml
# or
uvicorn consumer.mcp_server:app --host 0.0.0.0 --port 8001
```

---

## API Reference

Full endpoint documentation lives in [`docs/api/`](docs/api/):

| Doc | Endpoints |
|---|---|
| [Producer API Reference](docs/api/producer_api_ref.md) | `POST /index`, `POST /add`, `DELETE /remove`, `POST /reindex`, `GET /stats`, `POST /query`, `POST /search` |
| [Consumer API Reference](docs/api/consumer_api_ref.md) | `GET /health`, `POST /query`, `POST /search`, `GET /info` |

Matching **Postman collections** are available in the shared Postman workspace for interactive testing.

---

## CLI Reference

The `archrag` CLI can also drive the pipeline directly (no HTTP server required):

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

```powershell
archrag -v -c my_config.yaml query "some question"
```

---

## Architecture

```
CLI (click) / FastAPI
 │
 ▼
Orchestrator
 ├── KG Construction Service
 ├── Hierarchical Clustering Service  (Algorithm 1)
 ├── C-HNSW Build Service             (Algorithm 3)
 ├── Hierarchical Search Service       (Algorithm 2)
 ├── Adaptive Filtering Service        (Equations 1 & 2)
 └── Ingestion Queue                   (batched add → auto-flush)
      │
      ▼
   6 Ports (ABCs)
      │
      ▼
   Swappable Adapters
```

### Ports & Adapters

| Port | Adapters |
|---|---|
| **EmbeddingPort** | SentenceTransformers, OpenAI, Ollama |
| **LLMPort** | OpenAI, Ollama |
| **GraphStorePort** | SQLite, In-Memory, Stub |
| **DocumentStorePort** | SQLite, In-Memory, Stub |
| **VectorIndexPort** | NumPy (brute-force cosine), Stub |
| **ClusteringPort** | Leiden (via igraph + leidenalg) |

---

## Project Structure

```
Corpnet/
├── producer/
│   ├── api.py              # FastAPI write+read server (single per org)
│   ├── config.yaml         # Producer-specific settings
│   └── data/               # SQLite DB + CHNSW vectors (owned by producer)
├── consumer/
│   ├── mcp_server.py       # FastAPI read-only proxy (one or more per org)
│   ├── config.yaml         # Consumer-specific settings (incl. producer_url)
│   └── adapters/           # Stub adapters for test isolation
├── archrag/
│   ├── domain/models.py    # Pure dataclasses (Entity, Relation, Community, …)
│   ├── ports/              # 6 abstract base classes
│   ├── adapters/
│   │   ├── embeddings/     # SentenceTransformer, OpenAI, Ollama
│   │   ├── llms/           # OpenAI, Ollama
│   │   ├── stores/         # SQLite & in-memory (graph + document)
│   │   ├── indexes/        # NumPy vector index
│   │   └── clustering/     # Leiden
│   ├── services/           # Business logic (KG, clustering, C-HNSW, search, filtering, ingestion queue)
│   ├── prompts/            # LLM prompt templates
│   ├── config.py           # YAML config → adapter factory → orchestrator
│   └── cli.py              # Click CLI entry point
├── docs/api/               # Endpoint reference (producer + consumer)
├── tests/                  # Unit & integration tests
├── pyproject.toml
├── config.example.yaml
└── README.md
```

---

## Tests

```powershell
python -m pytest tests/ -v
```

---

## Paper Reference

> **ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation**  
> [arXiv:2502.09891](https://arxiv.org/abs/2502.09891)
