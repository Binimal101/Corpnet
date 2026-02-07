# DAC-HRAG: Project Setup & Architecture Guide

> **For**: Cursor with Claude Opus 4.5
> **Purpose**: Set up the full file directory, define module boundaries, interfaces, and implementation order for the Distributed Attributed Community-Hierarchical RAG system.
> **Read this file first. Do not create files outside this structure without explicit instruction.**

---

## Project Overview

DAC-HRAG is a distributed, access-controlled retrieval-augmented generation system. It combines:
- **ArchRAG-style** hierarchical community clustering and C-HNSW indexing
- **DRAG-style** peer-to-peer knowledge retrieval with topic-aware routing
- **Filtered-DiskANN** label-based access control at the vector search layer

The system has 5 major subsystems: **Ingestion**, **Clustering**, **Indexing**, **Routing/Networking**, and **Query/Generation**. Each subsystem is an independent package with well-defined interfaces.

---

## Directory Structure

```
dac-hrag/
│
├── README.md                          # Project overview, quickstart, architecture diagram
├── LICENSE
├── .env.example                       # Template for env vars (API keys, DB URLs, ports)
├── .gitignore
├── docker-compose.yml                 # Full local dev stack (postgres, peers, coordinator)
├── Makefile                           # Common commands: make dev, make test, make peer, etc.
├── pyproject.toml                     # Root Python project config (monorepo with workspaces)
├── requirements.txt                   # Pinned top-level deps
│
├── docs/
│   ├── architecture.md                # High-level architecture doc (from our synthesis)
│   ├── PROTOCOL.md                    # Wire protocol spec for peer-to-peer messages
│   ├── ACCESS_CONTROL.md              # Permission model, label taxonomy, IAM mapping
│   ├── CLUSTERING.md                  # Hierarchical clustering algorithm details
│   ├── LATENCY_BUDGET.md              # Per-phase latency targets and benchmarks
│   └── diagrams/
│       ├── system-overview.mermaid    # Full system diagram
│       ├── query-flow.mermaid         # Query lifecycle sequence diagram
│       ├── hierarchy-layers.mermaid   # Community hierarchy visualization
│       └── peer-topology.mermaid      # P2P overlay network diagram
│
│
│   ════════════════════════════════════════════════════════════════
│   CORE LIBRARIES (shared types, configs, utilities)
│   ════════════════════════════════════════════════════════════════
│
├── core/
│   ├── __init__.py
│   ├── config.py                      # Global config loader (reads .env, CLI args, defaults)
│   ├── types.py                       # Shared dataclasses/TypedDicts used across all modules
│   │                                  #   - Vector, Chunk, Entity, Relation
│   │                                  #   - Community, CommunityHierarchy
│   │                                  #   - PermissionLabel, PermissionSet
│   │                                  #   - QueryRequest, QueryResult, PeerInfo
│   │                                  #   - Message (wire protocol base type)
│   ├── embedding.py                   # Embedding interface (abstracts OpenAI / local models)
│   │                                  #   - embed_text(text) -> Vector
│   │                                  #   - embed_batch(texts) -> list[Vector]
│   ├── llm.py                         # LLM interface (abstracts Claude / GPT / local Llama)
│   │                                  #   - generate(prompt, context) -> str
│   │                                  #   - extract_topics(query) -> list[str]
│   │                                  #   - extract_entities(text) -> list[Entity]
│   │                                  #   - summarize_community(entities, relations) -> str
│   ├── auth.py                        # Permission resolution
│   │                                  #   - resolve_permissions(user_token) -> PermissionSet
│   │                                  #   - check_access(PermissionSet, label_set) -> bool
│   │                                  #   - create_user(name, groups) -> UserToken
│   ├── errors.py                      # Custom exception hierarchy
│   └── logging.py                     # Structured logging setup (JSON, correlation IDs)
│
│
│   ════════════════════════════════════════════════════════════════
│   SUBSYSTEM 1: INGESTION
│   Converts raw data sources into chunked, embedded, labeled vectors
│   ════════════════════════════════════════════════════════════════
│
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py                    # Main ingestion orchestrator
│   │                                  #   - ingest(source, permission_labels) -> list[Chunk]
│   │                                  #   Steps: source → extract → chunk → embed → tag → store
│   │
│   ├── extractors/                    # Raw data → plain text extraction
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base: Extractor.extract(source) -> list[Document]
│   │   ├── file_extractor.py          # Local files: PDF, DOCX, MD, TXT
│   │   ├── github_extractor.py        # GitHub: issues, PRs, READMEs, commit messages
│   │   ├── slack_extractor.py         # Slack: channel messages, threads
│   │   └── email_extractor.py         # Email: IMAP/SMTP ingestion
│   │
│   ├── chunking/                      # Text → chunks with metadata
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base: Chunker.chunk(doc) -> list[Chunk]
│   │   ├── recursive_chunker.py       # Recursive character splitting with overlap
│   │   ├── semantic_chunker.py        # Embedding-based boundary detection
│   │   └── code_chunker.py            # AST-aware splitting for source code
│   │
│   ├── labeling/                      # Permission label assignment
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base: Labeler.label(chunk, source_meta) -> PermissionSet
│   │   ├── source_labeler.py          # Inherit labels from data source (Slack channel → team label)
│   │   └── policy_labeler.py          # Apply org-defined labeling rules (regex, keyword, LLM-based)
│   │
│   └── knowledge_graph/               # Entity/relation extraction for ArchRAG KG construction
│       ├── __init__.py
│       ├── extractor.py               # LLM-based entity + relation extraction from chunks
│       │                              #   - extract_subgraph(chunk) -> (list[Entity], list[Relation])
│       ├── merger.py                   # Merge subgraphs, deduplicate entities across chunks
│       │                              #   - merge(subgraphs) -> KnowledgeGraph
│       └── graph.py                   # KnowledgeGraph data structure
│                                      #   - nodes: dict[str, Entity]
│                                      #   - edges: list[Relation]
│                                      #   - to_adjacency(), add_node(), add_edge()
│
│
│   ════════════════════════════════════════════════════════════════
│   SUBSYSTEM 2: CLUSTERING
│   Builds the hierarchical community structure (ArchRAG adaptation)
│   ════════════════════════════════════════════════════════════════
│
├── clustering/
│   ├── __init__.py
│   ├── pipeline.py                    # Full clustering orchestrator
│   │                                  #   - build_hierarchy(kg: KnowledgeGraph) -> CommunityHierarchy
│   │                                  #   Runs the iterative loop: augment → cluster → summarize → recurse
│   │
│   ├── augmentation/                  # Graph augmentation (ArchRAG step 1)
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract: Augmenter.augment(graph, embeddings) -> AugmentedGraph
│   │   ├── knn_augmenter.py           # Add edges between nodes with cosine sim > threshold
│   │   │                              #   Uses FAISS for fast k-NN computation
│   │   └── codicil_augmenter.py       # Add edges based on shared community from prev iteration
│   │
│   ├── detection/                     # Community detection algorithms
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract: Detector.detect(graph) -> list[Community]
│   │   ├── leiden_detector.py         # Weighted Leiden via leidenalg / igraph
│   │   └── spectral_detector.py       # Spectral clustering fallback
│   │
│   ├── summarization/                 # Community → summary embedding
│   │   ├── __init__.py
│   │   └── summarizer.py             # LLM-generated community summaries
│   │                                  #   - summarize(community) -> (text_summary, summary_vector)
│   │                                  #   Uses core.llm.summarize_community()
│   │
│   ├── hierarchy.py                   # CommunityHierarchy data structure
│   │                                  #   - levels: list[list[Community]]
│   │                                  #   - parent_map: dict[community_id, parent_community_id]
│   │                                  #   - get_level(n), get_children(community_id)
│   │                                  #   - get_path_to_root(community_id) -> list[community_id]
│   │                                  #   - label_set(community_id) -> PermissionSet (union of children)
│   │
│   └── federated/                     # Distributed clustering (cross-peer hierarchy building)
│       ├── __init__.py
│       ├── local_cluster.py           # Phase 1: per-peer local clustering
│       │                              #   - cluster_local(local_kg) -> list[Community] + summaries
│       ├── coordinator.py             # Phase 2: collect summaries, run cross-peer Leiden
│       │                              #   - merge_communities(peer_summaries) -> CommunityHierarchy
│       └── publisher.py               # Phase 3: distribute hierarchy to super-peers
│                                      #   - publish(hierarchy, super_peers) -> None
│
│
│   ════════════════════════════════════════════════════════════════
│   SUBSYSTEM 3: INDEXING
│   Vector storage + filtered search (C-HNSW + Filtered-DiskANN)
│   ════════════════════════════════════════════════════════════════
│
├── indexing/
│   ├── __init__.py
│   │
│   ├── vector_store/                  # Low-level vector storage backends
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract VectorStore interface:
│   │   │                              #   - insert(vector, metadata, labels) -> id
│   │   │                              #   - search(query_vec, k, filter_labels) -> list[Result]
│   │   │                              #   - delete(id) -> None
│   │   │                              #   - bulk_insert(vectors, metadatas, labels) -> list[id]
│   │   ├── pgvector_store.py          # PostgreSQL + pgvectorscale (StreamingDiskANN)
│   │   │                              #   Implements filtered label search natively
│   │   │                              #   Primary production backend
│   │   └── faiss_store.py             # In-memory FAISS (for testing / small deployments)
│   │                                  #   Filtered search via post-filter (less efficient)
│   │
│   ├── chnsw/                         # C-HNSW index (ArchRAG's hierarchical index)
│   │   ├── __init__.py
│   │   ├── index.py                   # C-HNSW index structure
│   │   │                              #   - layers: list[VectorStore]  (one per hierarchy level)
│   │   │                              #   - inter_layer_links: dict[community_id, list[child_ids]]
│   │   │                              #   - build(hierarchy: CommunityHierarchy) -> None
│   │   │                              #   - search(query_vec, labels, k_per_layer) -> HierarchicalResult
│   │   └── builder.py                 # C-HNSW construction from CommunityHierarchy
│   │                                  #   - Top-down insertion: insert summaries layer by layer
│   │                                  #   - Intra-layer: standard HNSW neighbor linking
│   │                                  #   - Inter-layer: parent → child containment links
│   │
│   └── freshness/                     # Real-time index maintenance
│       ├── __init__.py
│       └── updater.py                 # Handles streaming inserts/deletes
│                                      #   - on_new_chunk(chunk) -> update local index + re-evaluate community
│                                      #   - on_delete(chunk_id) -> tombstone + periodic compaction
│                                      #   Wraps FreshDiskANN / pgvectorscale streaming behavior
│
│
│   ════════════════════════════════════════════════════════════════
│   SUBSYSTEM 4: NETWORKING / ROUTING
│   P2P overlay, peer discovery, hierarchical query routing
│   ════════════════════════════════════════════════════════════════
│
├── network/
│   ├── __init__.py
│   │
│   ├── peer/                          # Individual peer node logic
│   │   ├── __init__.py
│   │   ├── node.py                    # Peer node lifecycle
│   │   │                              #   - start(config) -> running peer
│   │   │                              #   - join_network(bootstrap_peers) -> None
│   │   │                              #   - leave_network() -> None
│   │   │                              #   - handle_message(msg: Message) -> Response
│   │   │                              #   Owns: local VectorStore, local KG, local LLM
│   │   ├── registry.py                # Peer self-registration with the hierarchy
│   │   │                              #   - register_communities(local_communities) -> None
│   │   │                              #   - heartbeat() -> None (periodic liveness signal)
│   │   └── state.py                   # Local peer state
│   │                                  #   - peer_id, address, port
│   │                                  #   - owned_communities: list[community_id]
│   │                                  #   - neighbor_table: dict[peer_id, PeerInfo]
│   │                                  #   - expertise_cache: dict[topic, list[peer_id]]
│   │
│   ├── routing/                       # Query routing through the hierarchy
│   │   ├── __init__.py
│   │   ├── router.py                  # Main query router (the DAC-HRAG algorithm)
│   │   │                              #   - route(query: QueryRequest) -> list[PeerInfo]
│   │   │                              #   Implements the hierarchical descent:
│   │   │                              #     Layer N super-peers → Layer N-1 → ... → Layer 0 leaf peers
│   │   │                              #   Uses expertise cache for shortcutting known topics
│   │   ├── hierarchy_resolver.py      # Resolves community_id → responsible peers at each layer
│   │   │                              #   - resolve(community_id, layer) -> list[PeerInfo]
│   │   │                              #   Backed by DHT or routing table
│   │   └── cache.py                   # DRAG-style expertise cache (upgraded to hierarchy-aware)
│   │                                  #   - record_hit(topic, peer_path: list[PeerInfo]) -> None
│   │                                  #   - lookup(topic) -> Optional[list[PeerInfo]]
│   │                                  #   - evict(max_age, max_size) -> None
│   │
│   ├── transport/                     # Wire protocol / message passing
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract Transport interface:
│   │   │                              #   - send(peer_id, message) -> Response
│   │   │                              #   - broadcast(peer_ids, message) -> list[Response]
│   │   ├── grpc_transport.py          # gRPC-based transport (production)
│   │   └── local_transport.py         # In-process transport (testing / single-machine simulation)
│   │
│   ├── discovery/                     # Peer discovery and overlay management
│   │   ├── __init__.py
│   │   ├── bootstrap.py               # Initial peer discovery (hardcoded seeds / mDNS / config)
│   │   ├── dht.py                     # Kademlia-style DHT for community → peer mapping
│   │   │                              #   - put(key, value) -> None
│   │   │                              #   - get(key) -> value
│   │   │                              #   - find_peers(community_id) -> list[PeerInfo]
│   │   └── super_peer.py              # Super-peer election and management
│   │                                  #   - elect(candidates) -> list[PeerInfo]
│   │                                  #   - promote(peer_id) -> None (peer becomes super-peer for a layer)
│   │                                  #   - demote(peer_id) -> None
│   │
│   └── messages/                      # Message type definitions (wire protocol)
│       ├── __init__.py
│       ├── query.py                   # QueryMessage, QueryResponse
│       ├── routing.py                 # RouteRequest, RouteResponse (hierarchy traversal)
│       ├── registration.py            # PeerRegister, CommunityPublish, Heartbeat
│       └── sync.py                    # HierarchySync, CommunityUpdate (hierarchy propagation)
│
│
│   ════════════════════════════════════════════════════════════════
│   SUBSYSTEM 5: QUERY ENGINE / GENERATION
│   End-to-end query processing: embed → route → retrieve → generate
│   ════════════════════════════════════════════════════════════════
│
├── query/
│   ├── __init__.py
│   ├── engine.py                      # Top-level query orchestrator
│   │                                  #   - query(text, user_token) -> Answer
│   │                                  #   Full pipeline:
│   │                                  #     1. Embed query (core.embedding)
│   │                                  #     2. Extract topics (core.llm)
│   │                                  #     3. Resolve permissions (core.auth)
│   │                                  #     4. Route through hierarchy (network.routing.router)
│   │                                  #     5. Parallel filtered search at leaf peers
│   │                                  #     6. Aggregate + adaptive filter
│   │                                  #     7. Generate answer (core.llm)
│   │
│   ├── aggregation.py                 # Merge results from multiple peers
│   │                                  #   - merge(results: list[list[Result]]) -> list[Result]
│   │                                  #   De-duplicate, re-rank by score, enforce global top-k
│   │
│   ├── filtering.py                   # ArchRAG adaptive filtering
│   │                                  #   - adaptive_filter(results, query_vec, threshold) -> list[Result]
│   │                                  #   Score each result for relevance, discard below threshold
│   │                                  #   Prevents "lost in the middle" in LLM context
│   │
│   ├── privacy.py                     # DRAG-style privacy filtering (runs on peer before returning)
│   │                                  #   - filter_pii(text) -> text
│   │                                  #   - redact_sensitive(text, rules) -> text
│   │
│   └── generation.py                  # LLM answer generation with retrieved context
│                                      #   - generate_answer(query, context_chunks) -> Answer
│                                      #   Formats context, manages token budget, calls LLM
│
│
│   ════════════════════════════════════════════════════════════════
│   API / ENTRYPOINTS
│   ════════════════════════════════════════════════════════════════
│
├── api/
│   ├── __init__.py
│   ├── server.py                      # FastAPI application factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── query.py                   # POST /query  — user-facing query endpoint
│   │   ├── ingest.py                  # POST /ingest — trigger ingestion for a data source
│   │   ├── auth.py                    # POST /auth/login, /auth/groups — user + group management
│   │   ├── admin.py                   # GET /admin/peers, /admin/hierarchy — introspection
│   │   └── health.py                  # GET /health, /ready — liveness + readiness probes
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth_middleware.py         # JWT validation, permission label injection into request
│   │   └── logging_middleware.py      # Request/response logging with correlation IDs
│   └── schemas.py                     # Pydantic request/response models for all endpoints
│
│
│   ════════════════════════════════════════════════════════════════
│   CLI / PEER ENTRYPOINT
│   ════════════════════════════════════════════════════════════════
│
├── cli/
│   ├── __init__.py
│   ├── main.py                        # CLI entrypoint (click or typer)
│   │                                  #   dac-hrag peer start --port 8001 --bootstrap peer1:8000
│   │                                  #   dac-hrag ingest ./docs --labels eng-team
│   │                                  #   dac-hrag query "why is my build failing" --user alice
│   │                                  #   dac-hrag cluster build --source local
│   │                                  #   dac-hrag admin hierarchy --show
│   └── commands/
│       ├── __init__.py
│       ├── peer.py                    # Peer lifecycle commands (start, stop, status)
│       ├── ingest.py                  # Ingestion commands
│       ├── query_cmd.py               # Direct query from CLI
│       ├── cluster.py                 # Clustering commands (build, inspect, export)
│       └── admin.py                   # Admin/debug commands (dump hierarchy, list peers)
│
│
│   ════════════════════════════════════════════════════════════════
│   TESTING
│   ════════════════════════════════════════════════════════════════
│
├── tests/
│   ├── conftest.py                    # Shared fixtures: test embeddings, mock LLM, sample KG
│   ├── fixtures/
│   │   ├── sample_docs/               # Small set of test documents across 3-4 topics
│   │   ├── sample_kg.json             # Pre-built knowledge graph for testing
│   │   ├── sample_hierarchy.json      # Pre-built community hierarchy
│   │   └── users.json                 # Test users with different permission sets
│   │
│   ├── unit/                          # Fast, isolated, no external deps
│   │   ├── test_chunking.py
│   │   ├── test_labeling.py
│   │   ├── test_auth.py
│   │   ├── test_clustering.py
│   │   ├── test_hierarchy.py
│   │   ├── test_chnsw.py
│   │   ├── test_routing.py
│   │   ├── test_aggregation.py
│   │   ├── test_filtering.py
│   │   └── test_cache.py
│   │
│   ├── integration/                   # Requires DB / network but single-process
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_vector_store.py
│   │   ├── test_filtered_search.py    # CRITICAL: verify access control correctness
│   │   ├── test_query_engine.py
│   │   └── test_hierarchy_build.py
│   │
│   └── distributed/                   # Multi-peer simulation tests
│       ├── test_peer_join_leave.py
│       ├── test_distributed_query.py
│       ├── test_hierarchy_propagation.py
│       ├── test_access_control_e2e.py # Same query, different users → different results
│       └── test_latency.py            # Measure end-to-end latency at N peers
│
│
│   ════════════════════════════════════════════════════════════════
│   INFRASTRUCTURE
│   ════════════════════════════════════════════════════════════════
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.peer            # Single peer node image
│   │   ├── Dockerfile.coordinator     # Coordinator node image (runs cross-peer clustering)
│   │   └── Dockerfile.api             # API gateway image
│   ├── docker-compose.yml             # Local dev: 3 peers + 1 coordinator + postgres + API
│   ├── scripts/
│   │   ├── init_db.sql                # PostgreSQL schema: vectors table, DiskANN index, labels
│   │   ├── seed_data.py               # Load sample data into a running cluster for demos
│   │   └── simulate_network.py        # Spin up N in-process peers for load testing
│   └── proto/
│       └── dac_hrag.proto             # gRPC service + message definitions (if using gRPC transport)
│
│
│   ════════════════════════════════════════════════════════════════
│   BENCHMARKS
│   ════════════════════════════════════════════════════════════════
│
└── benchmarks/
    ├── recall_benchmark.py            # Compare DAC-HRAG recall vs centralized baseline
    ├── latency_benchmark.py           # Measure per-phase latency at varying peer counts
    ├── message_benchmark.py           # Count messages per query vs DRAG-TARW and DRAG-FL
    └── access_control_benchmark.py    # Verify zero unauthorized results under load
```

---

## Module Dependency Graph

Dependencies flow **downward only**. No circular imports. `core` depends on nothing internal. Every other module imports from `core` but never from a sibling subsystem's internals — only through the sibling's public interface (its `__init__.py` exports).

```
                            ┌──────────┐
                            │   cli/   │
                            │   api/   │
                            └────┬─────┘
                                 │ uses
                                 ▼
                          ┌─────────────┐
                          │   query/    │
                          │   engine    │
                          └──┬───┬───┬──┘
                             │   │   │
                uses         │   │   │         uses
           ┌─────────────────┘   │   └─────────────────┐
           ▼                     ▼                     ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
   │  indexing/    │    │  network/    │    │   clustering/    │
   │  (C-HNSW,    │    │  (routing,   │    │   (Leiden,       │
   │  VectorStore) │    │  transport,  │    │   hierarchy,     │
   │              │    │  discovery)  │    │   federated)     │
   └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘
          │                   │                     │
          │         uses      │         uses        │         uses
          └─────────┬─────────┴─────────────────────┘
                    ▼
             ┌─────────────┐          ┌──────────────┐
             │    core/    │◄─────────│  ingestion/  │
             │  (types,    │  uses    │  (extractors,│
             │  embedding, │          │  chunking,   │
             │  llm, auth) │          │  labeling,   │
             └─────────────┘          │  KG)         │
                                      └──────────────┘
```

**Import rules:**
- `core/` imports only stdlib and external packages (openai, faiss, etc.)
- `ingestion/` imports from `core/` only
- `clustering/` imports from `core/` and `ingestion.knowledge_graph.graph` (the KG data structure)
- `indexing/` imports from `core/` and `clustering.hierarchy` (the hierarchy data structure)
- `network/` imports from `core/`, `indexing.vector_store.base`, and `clustering.hierarchy`
- `query/` imports from all subsystems' public interfaces
- `api/` and `cli/` import from `query/` and `core/`

---

## Key Interfaces (Contracts Between Modules)

These are the critical interfaces that define module boundaries. Implement these first as abstract base classes, then build concrete implementations.

### `core/types.py` — Shared Types

```python
@dataclass
class Vector:
    data: list[float]
    dim: int

@dataclass
class Chunk:
    id: str
    text: str
    vector: Vector
    source: str                    # origin (file path, URL, channel, etc.)
    labels: set[str]               # permission labels
    metadata: dict[str, Any]

@dataclass
class Entity:
    id: str
    name: str
    description: str
    vector: Vector
    source_chunks: list[str]       # chunk IDs this entity was extracted from

@dataclass
class Relation:
    source_entity: str             # entity ID
    target_entity: str             # entity ID
    relation_type: str
    description: str

@dataclass
class Community:
    id: str
    level: int                     # hierarchy level (0 = leaf entities)
    summary: str
    vector: Vector                 # summary embedding
    labels: set[str]               # union of member labels
    member_ids: list[str]          # child entity or community IDs
    peer_owners: list[str]         # peer IDs that hold data for this community

@dataclass
class QueryRequest:
    text: str
    vector: Vector
    topics: list[str]
    user_labels: set[str]          # resolved permission labels
    k: int = 10

@dataclass
class QueryResult:
    chunk_id: str
    text: str
    score: float
    source: str
    peer_id: str
```

### `indexing/vector_store/base.py` — Vector Store Interface

```python
class VectorStore(ABC):
    @abstractmethod
    def insert(self, chunk: Chunk) -> str: ...

    @abstractmethod
    def search(
        self,
        query: Vector,
        k: int,
        filter_labels: set[str] | None = None
    ) -> list[QueryResult]: ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...
```

### `network/transport/base.py` — Transport Interface

```python
class Transport(ABC):
    @abstractmethod
    async def send(self, peer_id: str, message: Message) -> Response: ...

    @abstractmethod
    async def broadcast(
        self, peer_ids: list[str], message: Message
    ) -> list[Response]: ...

    @abstractmethod
    async def listen(self, port: int, handler: Callable) -> None: ...
```

### `network/routing/router.py` — Router Interface

```python
class Router(ABC):
    @abstractmethod
    async def route(self, request: QueryRequest) -> list[QueryResult]:
        """
        Full hierarchical descent:
        1. Start at super-peers (top layer)
        2. Greedy search community summaries at each layer
        3. Descend to child communities
        4. At leaf layer, run filtered vector search on data peers
        5. Aggregate and return
        """
        ...
```

---

## Implementation Order

Build bottom-up. Each phase produces a working, testable increment.

### Phase 1: Core + Types + Single-Node RAG (days 1-2)
1. `core/types.py` — all shared dataclasses
2. `core/config.py` — env var loading
3. `core/embedding.py` — OpenAI or local embedding wrapper
4. `core/llm.py` — LLM wrapper (start with OpenAI, swap later)
5. `core/auth.py` — simple in-memory group → label mapping
6. `ingestion/chunking/recursive_chunker.py` — basic chunking
7. `ingestion/labeling/source_labeler.py` — inherit labels from source metadata
8. `indexing/vector_store/pgvector_store.py` — pgvectorscale with DiskANN + label filter
9. `query/engine.py` — minimal: embed → search local store → generate
10. **TEST**: `test_filtered_search.py` — same query, user A sees docs X, user B sees docs Y

### Phase 2: Knowledge Graph + Clustering (days 3-4)
1. `ingestion/knowledge_graph/extractor.py` — LLM entity/relation extraction
2. `ingestion/knowledge_graph/merger.py` — subgraph merging
3. `clustering/augmentation/knn_augmenter.py` — FAISS k-NN graph augmentation
4. `clustering/detection/leiden_detector.py` — weighted Leiden via leidenalg
5. `clustering/summarization/summarizer.py` — LLM community summaries
6. `clustering/pipeline.py` — iterative loop: augment → cluster → summarize → recurse
7. `clustering/hierarchy.py` — CommunityHierarchy data structure
8. **TEST**: `test_hierarchy_build.py` — verify multi-level hierarchy from sample corpus

### Phase 3: C-HNSW Index (days 5-6)
1. `indexing/chnsw/builder.py` — build C-HNSW from CommunityHierarchy
2. `indexing/chnsw/index.py` — hierarchical search (top-down greedy descent)
3. Integrate with `query/engine.py` — queries now route through C-HNSW before leaf search
4. `query/aggregation.py` — merge results from multiple layers
5. `query/filtering.py` — ArchRAG adaptive filtering
6. **TEST**: `test_chnsw.py` — verify hierarchical search finds correct communities

### Phase 4: Networking + Distribution (days 7-10)
1. `network/transport/local_transport.py` — in-process message passing (for testing)
2. `network/peer/node.py` — peer lifecycle (start, join, handle messages)
3. `network/peer/state.py` — local peer state management
4. `network/messages/` — all message types
5. `network/routing/router.py` — hierarchical descent across peers
6. `network/routing/cache.py` — expertise cache
7. `network/discovery/bootstrap.py` — simple seed-based discovery
8. `clustering/federated/local_cluster.py` — per-peer local clustering
9. `clustering/federated/coordinator.py` — cross-peer hierarchy merging
10. **TEST**: `test_distributed_query.py` — 3+ peers, query routes correctly

### Phase 5: API + CLI + Demo (days 11-12)
1. `api/server.py` + all routes
2. `cli/main.py` + all commands
3. `infra/docker-compose.yml` — full local stack
4. `infra/scripts/seed_data.py` — load demo data
5. **DEMO**: Two users query the same cluster, get different results based on permissions

---

## Database Schema (PostgreSQL + pgvectorscale)

```sql
-- infra/scripts/init_db.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;

CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text        TEXT NOT NULL,
    source      TEXT NOT NULL,
    embedding   VECTOR(1536) NOT NULL,     -- adjust dim to your embedding model
    labels      TEXT[] NOT NULL DEFAULT '{}', -- permission labels
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- StreamingDiskANN index with label-based filtering
CREATE INDEX chunks_embedding_idx
    ON chunks
    USING diskann (embedding vector_cosine_ops);

-- Label index for filtered queries
CREATE INDEX chunks_labels_idx ON chunks USING GIN (labels);

-- Community summaries table (hierarchy layers 1+)
CREATE TABLE communities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level           INT NOT NULL,
    summary         TEXT NOT NULL,
    embedding       VECTOR(1536) NOT NULL,
    labels          TEXT[] NOT NULL DEFAULT '{}',
    member_ids      UUID[] NOT NULL DEFAULT '{}',   -- child chunk or community IDs
    parent_id       UUID REFERENCES communities(id),
    peer_owner_ids  TEXT[] NOT NULL DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX communities_embedding_idx
    ON communities
    USING diskann (embedding vector_cosine_ops);

CREATE INDEX communities_level_idx ON communities (level);
CREATE INDEX communities_labels_idx ON communities USING GIN (labels);

-- Users and groups for access control
CREATE TABLE users (
    id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name    TEXT UNIQUE NOT NULL,
    groups  TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE groups (
    name    TEXT PRIMARY KEY,
    labels  TEXT[] NOT NULL DEFAULT '{}'    -- permission labels this group grants
);
```

---

## Environment Variables

```bash
# .env.example

# === Embedding ===
EMBEDDING_PROVIDER=openai              # openai | local
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# === LLM ===
LLM_PROVIDER=anthropic                 # anthropic | openai | local
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...

# === Database ===
DATABASE_URL=postgresql://dachrag:password@localhost:5432/dachrag

# === Peer Networking ===
PEER_ID=peer-001
PEER_HOST=0.0.0.0
PEER_PORT=8001
BOOTSTRAP_PEERS=localhost:8000         # comma-separated seed peers

# === Clustering ===
LEIDEN_RESOLUTION=1.0                  # higher = more/smaller communities
KNN_K=10                               # k for graph augmentation
KNN_THRESHOLD=0.7                      # cosine similarity threshold for k-NN edges
HIERARCHY_MAX_LEVELS=5

# === Search ===
SEARCH_K_PER_LAYER=5                   # top-k communities to descend at each layer
SEARCH_K_FINAL=10                      # final top-k results to return
RELEVANCE_THRESHOLD=0.6                # adaptive filtering cutoff

# === Auth ===
JWT_SECRET=change-me-in-production
```

---

## Makefile

```makefile
.PHONY: dev test peer coordinator seed lint

# Start full local dev stack
dev:
	docker compose -f infra/docker-compose.yml up --build

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run only unit tests (fast, no deps)
unit:
	pytest tests/unit/ -v --tb=short

# Start a single peer node
peer:
	python -m cli.main peer start --port $(PORT) --bootstrap $(BOOTSTRAP)

# Start the coordinator
coordinator:
	python -m cli.main peer start --port 8000 --role coordinator

# Seed demo data into a running cluster
seed:
	python infra/scripts/seed_data.py

# Run the API server
api:
	uvicorn api.server:app --host 0.0.0.0 --port 8080 --reload

# Lint
lint:
	ruff check . && mypy .
```

---

## Notes for the Agent

1. **Always start with `core/types.py`**. Every module depends on it. Get the dataclasses right first.
2. **Use abstract base classes for all interfaces** (`VectorStore`, `Transport`, `Router`, `Extractor`, etc.). Implement concrete classes second.
3. **`__init__.py` files are the public API** of each package. Only export what other packages need. Keep internal helpers private.
4. **No cross-subsystem internal imports.** If `network/` needs something from `clustering/`, it imports from `clustering.hierarchy` (the public module), not from `clustering.detection.leiden_detector` (internal implementation).
5. **Test filtered search correctness obsessively.** The access control guarantee is the project's primary differentiator. Write a `test_access_control_e2e.py` that runs 1000 queries with random user/label combinations and asserts zero unauthorized results.
6. **Start with `local_transport.py`** for networking tests. Get the distributed query algorithm working in-process before adding real network I/O.
7. **The `pgvector_store.py` is the critical path.** Get the DiskANN index + label filtering working in PostgreSQL before building the C-HNSW layer on top.
8. **Community hierarchy is eventually consistent.** Don't block queries on hierarchy updates. Stale hierarchy = slightly suboptimal routing, not incorrect results. Correctness comes from the leaf-level filtered search.
