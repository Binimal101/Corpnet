# ArchRAG — Hexagonal Architecture Plan

## Paper Summary

**ArchRAG** (Attributed Community-based Hierarchical RAG) is a graph-based
Retrieval-Augmented Generation system with two phases:

### Offline Indexing
1. **KG Construction** — Chunk corpus → LLM extracts entities & relations → merge into a Knowledge Graph.
2. **LLM-based Hierarchical Clustering** — Augment KG (KNN edges by attribute similarity) → weighted community detection (Leiden) → LLM summarises each community → build higher-level graph of communities → repeat → produces a hierarchical tree Δ of Attributed Communities (ACs).
3. **C-HNSW Index** — Map entities (layer 0) and ACs (layers 1…L) to embeddings → build a Community-based HNSW index with intra-layer links (M nearest neighbours) and inter-layer links (nearest neighbour in adjacent layer).

### Online Retrieval
1. **Hierarchical Search** — Embed query → start from top layer, greedy traverse intra-layer links to find k nearest neighbours per layer, follow inter-layer links downward → collect results R₀…R_L.
2. **Adaptive Filtering-based Generation** — For each Rᵢ, LLM extracts an analysis report with relevance scores → sort and merge reports → LLM produces final answer.

---

## Hexagonal (Ports & Adapters) Design

The core insight: **anything that touches an external model or a persistent layer
goes behind a port**. The domain logic depends only on abstract interfaces.
Adapters are swapped via configuration.

```
                    ┌─────────────────────────────────────┐
                    │          DOMAIN / SERVICES           │
                    │                                      │
                    │  KGConstructionService                │
                    │  HierarchicalClusteringService        │
                    │  CHNSWBuildService                    │
                    │  HierarchicalSearchService            │
                    │  AdaptiveFilteringService             │
                    │  ArchRAGOrchestrator                  │
                    │                                      │
                    │  Domain Models (Entity, Relation,     │
                    │   KnowledgeGraph, Community,          │
                    │   CommunityHierarchy, CHNSWIndex)     │
                    └──┬───┬───┬───┬───┬───┬───────────────┘
                       │   │   │   │   │   │
            ┌──────────┘   │   │   │   │   └──────────┐
            ▼              ▼   ▼   ▼   ▼              ▼
     ┌──────────┐  ┌──────┐ ┌─┐ ┌─┐ ┌──────┐  ┌──────────┐
     │EmbeddingP│  │LLM P │ │G│ │V│ │DocStr│  │Clustering│
     │   ort    │  │ ort  │ │r│ │e│ │Port  │  │   Port   │
     └────┬─────┘  └──┬───┘ │a│ │c│ └──┬───┘  └────┬─────┘
          │            │     │p│ │t│    │            │
          ▼            ▼     │h│ │o│    ▼            ▼
   ┌────────────┐ ┌────────┐│S│ │r│┌────────┐ ┌──────────┐
   │Nomic       │ │OpenAI  ││t│ │I││JSON    │ │Leiden    │
   │SentenceTfm │ │Ollama  ││o│ │n││SQLite  │ │Spectral  │
   │OpenAI Embed│ │Llama   ││r│ │d││        │ │SCAN      │
   └────────────┘ └────────┘│e│ │e│└────────┘ └──────────┘
                            │P│ │x│
                            │o│ │P│
                            │r│ │o│
                            │t│ │r│
                            └┬┘ │t│
                             │  └┬┘
                             ▼   ▼
                        ┌──────┐┌───────┐
                        │SQLite││Numpy  │
                        │Neo4j ││FAISS  │
                        └──────┘└───────┘
```

---

## Port Interfaces

| Port | Responsibility | Key Methods |
|------|---------------|-------------|
| **EmbeddingPort** | Text → vector | `embed(text) → list[float]`, `embed_batch(texts) → list[list[float]]` |
| **LLMPort** | Prompt → completion | `generate(prompt, system?) → str`, `generate_json(prompt, system?) → dict` |
| **GraphStorePort** | Persist KG (entities + relations) | `save_entities()`, `save_relations()`, `get_entity()`, `get_neighbours()`, `get_subgraph()` |
| **VectorIndexPort** | ANN index for C-HNSW | `add_vectors()`, `search()`, `save()`, `load()` |
| **DocumentStorePort** | Persist corpus chunks, community summaries, hierarchy metadata | `save_document()`, `get_document()`, `save_hierarchy()`, `load_hierarchy()` |
| **ClusteringPort** | Weighted graph → communities | `cluster(nodes, edges, weights) → list[set[str]]` |

---

## Domain Models (pure Python dataclasses)

| Model | Fields |
|-------|--------|
| **TextChunk** | id, text, metadata, source_doc |
| **Entity** | id, name, description, embedding? |
| **Relation** | id, source_id, target_id, description, weight? |
| **KnowledgeGraph** | entities: dict, relations: list |
| **Community** | id, level, member_entity_ids, summary, embedding? |
| **CommunityHierarchy** | levels: list[list[Community]], parent_map |
| **CHNSWLayer** | level, node_ids, intra_links, inter_links_down |
| **CHNSWIndex** | layers: list[CHNSWLayer], embeddings: dict |
| **SearchResult** | node_id, level, distance, text |
| **AnalysisReport** | points: list[{description, score}] |

---

## Services (depend ONLY on ports)

### 1. KGConstructionService
- Input: list of TextChunks
- Uses: LLMPort (extract entities/relations), EmbeddingPort (entity embeddings), GraphStorePort (persist)
- Output: KnowledgeGraph persisted

### 2. HierarchicalClusteringService
- Input: KnowledgeGraph
- Uses: EmbeddingPort (similarity), ClusteringPort (detect communities), LLMPort (summarise), DocumentStorePort (persist hierarchy)
- Algorithm: iterative — augment graph, compute weights, cluster, summarise, build upper graph, repeat
- Output: CommunityHierarchy persisted

### 3. CHNSWBuildService
- Input: CommunityHierarchy + entity embeddings
- Uses: EmbeddingPort (community summary embeddings), VectorIndexPort (store & link)
- Algorithm: top-down insertion with intra/inter-layer links
- Output: CHNSWIndex persisted

### 4. HierarchicalSearchService
- Input: query string
- Uses: EmbeddingPort (query vector), VectorIndexPort (traverse C-HNSW)
- Algorithm: start at top layer, SearchLayer per layer, follow inter-layer links
- Output: list of SearchResult per layer

### 5. AdaptiveFilteringService
- Input: search results per layer
- Uses: LLMPort (filter prompt → analysis reports, merge prompt → final answer)
- Output: final answer string

### 6. ArchRAGOrchestrator
- Wires everything together
- `index(corpus_path)` → runs KG construction + clustering + C-HNSW build
- `query(question)` → runs hierarchical search + adaptive filtering

---

## Adapter Implementations (Phase 1)

| Port | Default Adapter | Swap-in Options |
|------|----------------|-----------------|
| EmbeddingPort | `SentenceTransformerAdapter` (nomic-embed-text) | `OpenAIEmbeddingAdapter`, `OllamaEmbeddingAdapter` |
| LLMPort | `OllamaAdapter` (llama3.1) | `OpenAIAdapter`, `AnthropicAdapter` |
| GraphStorePort | `SQLiteGraphStore` | `InMemoryGraphStore` (tests), future: Neo4j |
| VectorIndexPort | `NumpyVectorIndex` (pure-python C-HNSW) | `FAISSVectorIndex` |
| DocumentStorePort | `SQLiteDocumentStore` | `JSONDocumentStore`, `InMemoryDocStore` (tests) |
| ClusteringPort | `LeidenAdapter` | `SpectralClusteringAdapter`, `SCANAdapter` |

---

## File Layout

```
archrag/
├── __init__.py
├── domain/
│   ├── __init__.py
│   └── models.py              # Pure dataclasses
├── ports/
│   ├── __init__.py
│   ├── embedding.py           # EmbeddingPort ABC
│   ├── llm.py                 # LLMPort ABC
│   ├── graph_store.py         # GraphStorePort ABC
│   ├── vector_index.py        # VectorIndexPort ABC
│   ├── document_store.py      # DocumentStorePort ABC
│   └── clustering.py          # ClusteringPort ABC
├── adapters/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── sentence_transformer.py
│   │   ├── openai_embedding.py
│   │   └── ollama_embedding.py
│   ├── llms/
│   │   ├── __init__.py
│   │   ├── ollama_llm.py
│   │   └── openai_llm.py
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── sqlite_graph.py
│   │   ├── sqlite_document.py
│   │   ├── in_memory_graph.py
│   │   └── in_memory_document.py
│   ├── indexes/
│   │   ├── __init__.py
│   │   └── numpy_vector.py
│   └── clustering/
│       ├── __init__.py
│       └── leiden.py
├── services/
│   ├── __init__.py
│   ├── kg_construction.py
│   ├── hierarchical_clustering.py
│   ├── chnsw_build.py
│   ├── hierarchical_search.py
│   ├── adaptive_filtering.py
│   └── orchestrator.py
├── prompts/
│   ├── __init__.py
│   ├── extraction.py          # Entity/relation extraction prompts
│   ├── summarization.py       # Community summary prompts
│   └── filtering.py           # Adaptive filtering & merge prompts
├── config.py                  # YAML config loading + adapter factory
└── cli.py                     # Click-based CLI
tests/
├── __init__.py
├── conftest.py                # Fixtures: mock ports, sample data
├── test_models.py
├── test_kg_construction.py
├── test_clustering.py
├── test_chnsw.py
├── test_search.py
├── test_filtering.py
└── test_orchestrator.py
config.example.yaml
pyproject.toml
```

---

## Configuration (config.example.yaml)

```yaml
embedding:
  adapter: sentence_transformer  # | openai | ollama
  model: nomic-embed-text-v1.5
  dimension: 768

llm:
  adapter: ollama                # | openai
  model: llama3.1:8b
  base_url: http://localhost:11434
  temperature: 0.0

graph_store:
  adapter: sqlite                # | in_memory
  path: data/archrag.db

document_store:
  adapter: sqlite                # | in_memory | json
  path: data/archrag.db

vector_index:
  adapter: numpy                 # | faiss
  distance_metric: cosine

clustering:
  adapter: leiden                # | spectral | scan
  resolution: 1.0

indexing:
  chunk_size: 1200
  chunk_overlap: 100
  max_hierarchy_levels: 5
  knn_k: auto                   # auto = avg node degree
  similarity_threshold: 0.7

retrieval:
  k_per_layer: 5
  ef_search: 100

chnsw:
  M: 32                         # max connections per node
  ef_construction: 100
```

---

## Key Design Decisions

1. **Pure domain** — models.py has zero imports from adapters or external libs.
2. **Ports are ABCs** — every service constructor takes ports as arguments (dependency injection).
3. **Adapters are leaf nodes** — they import external libraries (sentence-transformers, openai, leidenalg, etc.) but nothing imports them except the config factory.
4. **Config-driven wiring** — `config.py` reads YAML → instantiates the right adapter for each port → passes them to services.
5. **C-HNSW in pure Python/NumPy** — avoids the custom FAISS fork from the paper; later swappable to FAISS via VectorIndexPort.
6. **Testability** — every service can be tested with InMemory* adapters and a mock LLM port.
