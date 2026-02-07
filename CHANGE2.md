# DAC-HRAG Architecture Changes v2 — Hackathon Document Alignment

> **Source**: DAC-HRAG-Hackathon-Project.docx
> **Purpose**: Align Cursor's implementation with decisions made in the hackathon build plan.
> **Apply these changes on top of the previous architecture changes file.**

---

## Change 1: Flattened Project Structure

The hackathon doc uses a **flattened `src/` layout** rather than the deep nested structure from the original project setup. All modules live directly under `src/` with single files instead of subdirectories with `base.py` + implementations.

### New directory structure (replaces the original)

```
dac-hrag/
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
├── config/
│   ├── default.yaml
│   └── test.yaml
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── config.py
│   │   ├── embeddings.py          # was core/embedding.py
│   │   └── llm.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py             # single file, not chunking/ subdirectory
│   │   ├── labeler.py             # single file, not labeling/ subdirectory
│   │   ├── entity_extractor.py    # was ingestion/knowledge_graph/extractor.py
│   │   └── pipeline.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py     # was ingestion/knowledge_graph/graph.py + merger.py
│   │   ├── knn_augmentation.py    # was clustering/augmentation/knn_augmenter.py
│   │   ├── leiden.py              # was clustering/detection/leiden_detector.py
│   │   ├── summarizer.py          # was clustering/summarization/summarizer.py
│   │   └── hierarchy.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # single file, not vector_store/ subdirectory
│   │   ├── chnsw.py               # was indexing/chnsw/index.py + builder.py
│   │   └── recall_canary.py       # was indexing/freshness/updater.py split
│   ├── networking/
│   │   ├── __init__.py
│   │   ├── messages.py            # was network/messages/ subdirectory (all types in one file)
│   │   ├── peer.py                # was network/peer/node.py + state.py
│   │   ├── router.py              # was network/routing/router.py + hierarchy_resolver.py + cache.py
│   │   ├── local_transport.py
│   │   ├── grpc_transport.py
│   │   ├── coordinator.py         # was clustering/federated/coordinator.py + publisher.py
│   │   └── dht.py                 # was network/discovery/dht.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── reranker.py            # was query/aggregation.py + filtering.py combined
│   │   └── generator.py           # was query/generation.py
│   └── api/
│       ├── __init__.py
│       ├── server.py
│       ├── routes.py              # single file, not routes/ subdirectory
│       └── cli.py                 # was cli/main.py + commands/ subdirectory
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_ingestion.py
│   ├── test_clustering.py
│   ├── test_indexing.py
│   ├── test_networking.py
│   ├── test_query.py
│   ├── test_api.py
│   └── test_access_control.py
├── scripts/
│   ├── seed_data.py
│   ├── load_test.py
│   └── migrate_db.py
└── proto/
    └── dachrag.proto
```

### Key merges

| Original (deep structure) | New (flat structure) | Rationale |
|---|---|---|
| `ingestion/knowledge_graph/extractor.py` + `merger.py` + `graph.py` | `ingestion/entity_extractor.py` + `clustering/knowledge_graph.py` | Entity extraction is ingestion; graph construction is clustering |
| `clustering/augmentation/knn_augmenter.py` | `clustering/knn_augmentation.py` | No need for abstract base + implementation pattern for a single algorithm |
| `clustering/detection/leiden_detector.py` | `clustering/leiden.py` | Same reasoning |
| `network/messages/*.py` (4 files) | `networking/messages.py` | All message types in one file is clearer for a hackathon |
| `network/peer/node.py` + `state.py` + `registry.py` | `networking/peer.py` | Peer state and registry are small enough to colocate |
| `network/routing/router.py` + `cache.py` + `hierarchy_resolver.py` | `networking/router.py` | Router includes cache and resolution logic |
| `query/aggregation.py` + `filtering.py` | `query/reranker.py` | Merge + dedup + adaptive filter are one logical step |
| `api/routes/*.py` (5 files) + `middleware/*.py` | `api/routes.py` | Single routes file for hackathon scope |
| `cli/main.py` + `commands/*.py` | `api/cli.py` | CLI lives alongside the API |

### Files to NOT create

These files from the original structure are **not needed**:
- Any `base.py` abstract base class files (implement concretely)
- `core/auth.py` (deferred)
- `core/errors.py` (use built-in exceptions for now)
- `core/logging.py` (use stdlib logging)
- `indexing/freshness/` directory (recall_canary.py covers monitoring; pgvectorscale handles freshness)
- `network/discovery/bootstrap.py` + `super_peer.py` (coordinator handles discovery)
- `query/privacy.py` (deferred)
- `infra/` directory (docker-compose lives at root)
- `benchmarks/` directory (scripts/load_test.py covers this)
- `docs/` directory (README.md is sufficient for hackathon)

---

## Change 2: Configuration System — YAML + Env Overlay

The hackathon doc specifies a **YAML-based config with env var overrides**, replacing the pure `.env` approach.

### Create `config/default.yaml`

```yaml
database:
  host: localhost
  port: 5432
  name: dachrag
  user: dachrag
  password: password
  pool_size: 10

embeddings:
  provider: sentence-transformers     # sentence-transformers | openai
  model: nomic-ai/nomic-embed-text-v1.5
  dimension: 768
  batch_size: 64

llm:
  provider: anthropic                 # anthropic | openai | local
  model: claude-sonnet-4-20250514
  temperature: 0.1
  max_tokens: 2048

clustering:
  leiden_resolution: [2.0, 1.0, 0.5]  # resolution per level
  min_community_size: 3
  knn_k: 10
  knn_threshold: 0.5
  silhouette_threshold: 0.3
  max_hierarchy_levels: 5

indexing:
  ef_search: 64
  ef_construction: 128
  num_neighbors: 32

routing:
  similarity_threshold: 0.35
  min_communities_per_layer: 1
  max_communities_per_layer: 20
  top_k_results: 10

networking:
  grpc_port: 50051
  max_peers: 100
  heartbeat_interval: 30
  coordinator_address: localhost:50050

maintenance:
  centroid_update_threshold: 1000
  canary_interval_minutes: 30
  recluster_cooldown_seconds: 3600
```

### Modify `src/core/config.py`

```python
"""
YAML config loader with environment variable overlay.
Env vars take precedence over YAML values.
Env var naming: DACHRAG__{section}__{key} (double underscore separator)
e.g., DACHRAG__DATABASE__HOST overrides database.host
"""
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "dachrag"
    user: str = "dachrag"
    password: str = "password"
    pool_size: int = 10

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class EmbeddingsConfig:
    provider: str = "sentence-transformers"
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimension: int = 768
    batch_size: int = 64

@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_tokens: int = 2048

@dataclass
class ClusteringConfig:
    leiden_resolution: list = field(default_factory=lambda: [2.0, 1.0, 0.5])
    min_community_size: int = 3
    knn_k: int = 10
    knn_threshold: float = 0.5
    silhouette_threshold: float = 0.3
    max_hierarchy_levels: int = 5

@dataclass
class RoutingConfig:
    similarity_threshold: float = 0.35
    min_communities_per_layer: int = 1
    max_communities_per_layer: int = 20
    top_k_results: int = 10

@dataclass
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    # ... networking, maintenance, indexing configs follow same pattern

_settings: Settings | None = None

def get_settings(config_path: str = "config/default.yaml") -> Settings:
    global _settings
    if _settings is None:
        _settings = _load_settings(config_path)
    return _settings

def _load_settings(config_path: str) -> Settings:
    # Load YAML, overlay env vars, return Settings
    ...
```

---

## Change 3: Type System Refinements

The hackathon doc defines more detailed types than our original. Key differences:

### `DocumentChunk` replaces `Chunk`

```python
@dataclass
class DocumentChunk:
    chunk_id: str           # UUID
    doc_id: str             # Parent document ID
    text: str
    embedding: list[float] | None   # None before embedding step
    labels: set[str]        # Permission labels (empty set for MVP)
    metadata: dict[str, Any]        # source_file, page, position, etc.
    entities: list[str]             # Extracted entity names
    relations: list[tuple[str, str, str]]  # (source, relation_type, target)
```

Key additions vs our `Chunk`:
- `doc_id` field (chunks reference their parent document)
- `entities` and `relations` fields (extracted KG data lives ON the chunk, not separate)
- `embedding` is `Optional` (None before the embedding step runs)

### `QueryResponse` with observability

```python
@dataclass
class QueryResponse:
    query_id: str
    answer: str
    results: list[SearchResult]
    routing_path: list[str]             # Community IDs traversed
    latency_ms: dict[str, float]        # Per-stage timing: {"embed": 12, "route": 85, ...}
    total_messages: int                 # Inter-peer messages used
```

### `Community` with bidirectional links

```python
@dataclass
class Community:
    community_id: str
    level: int
    summary: str
    summary_embedding: list[float]
    labels: set[str]            # Union of member labels (empty for MVP)
    member_ids: list[str]       # Child entity or community IDs
    parent_id: str | None       # Upward link
    children_ids: list[str]     # Downward links (explicit, not just member_ids)
    peer_id: str                # Which peer owns this community
```

Key additions:
- `children_ids` explicit field (parent AND children, not just members)
- `peer_id` (single owner, not a list — simpler for hackathon)

### `PeerInfo` with super-peer flag

```python
@dataclass
class PeerInfo:
    peer_id: str
    address: str
    port: int
    communities: list[str]      # Community IDs this peer owns
    is_super_peer: bool
    last_heartbeat: float
```

### Serialization requirement

All types must implement `to_dict()` and `from_dict()` for wire transport (msgpack serialization). Add these as methods on each dataclass:

```python
@dataclass
class DocumentChunk:
    # ... fields ...
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "embedding": self.embedding,
            "labels": list(self.labels),
            "metadata": self.metadata,
            "entities": self.entities,
            "relations": [list(r) for r in self.relations],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DocumentChunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            text=d["text"],
            embedding=d.get("embedding"),
            labels=set(d.get("labels", [])),
            metadata=d.get("metadata", {}),
            entities=d.get("entities", []),
            relations=[tuple(r) for r in d.get("relations", [])],
        )
```

---

## Change 4: Query Lifecycle — 6 Stages, Not 7

The hackathon doc defines a **6-stage pipeline** with modified stage 3 (routing), stage 4 (leaf search), and stage 5 (aggregation) semantics:

| Stage | Hackathon Doc | Change from Previous |
|---|---|---|
| 1. Embed | Same | — |
| 2. Extract Topics | Same | — |
| 3. Hierarchical Routing | Similarity search on community **embeddings**. Threshold-based selection, pruned to safety cap. Labels intersection check included (post-MVP). | Clarifies: embeddings not text. Confirms threshold + safety cap. |
| **4. Leaf Search** | **Super peers forward query to leaf peers. Leaf peers run search against their local vector store via an abstract `search()` function.** | **Changed**: No more parallel Filtered-DiskANN description. The leaf peer just runs its own search — the super peer doesn't reach into the leaf peer's store. The query flows DOWN the hierarchy and each leaf peer executes locally. |
| **5. Aggregate & Filter** | **Results bubble BACK UP to the initiating super-peer. Super-peer merges and re-ranks by similarity, then returns to the querying party.** | **Changed**: Results flow back up the same routing path. The super-peer is the aggregation point, not the query engine. |
| 6. Generate Answer | Top-k results formatted with source attribution, passed to LLM | Same |

### Key routing flow change

The hackathon doc makes the **super-peer the orchestrator of the entire query**, not the query engine on the client side:

```
Client → Super-peer (Layer N)
    Super-peer searches its community embeddings
    Selects matching sub-communities
    Forwards query to Layer N-1 peers
        Layer N-1 peers search their community embeddings
        Select matching sub-communities
        Forward to Layer N-2 peers
            ... continues until leaf peers ...
            Leaf peers search their local vector store
            Return results UP to their parent peer
        Layer N-2 peers aggregate results, return UP
    Layer N-1 peers aggregate results, return UP
Super-peer aggregates all results, re-ranks
Super-peer returns to client
```

This is a **recursive descent with recursive aggregation** pattern, not a flat dispatch-and-collect pattern.

### Modify `networking/router.py`

```python
class HierarchicalRouter:
    """
    Recursive descent routing. The query enters at a super-peer and
    cascades DOWN through the hierarchy. Results bubble back UP.
    
    Each peer at each layer:
    1. Receives the query
    2. Searches its own community embeddings (threshold-based)
    3. Forwards to child peers for matching communities
    4. Collects results from children
    5. Returns merged results to its parent
    """
    
    async def handle_query(self, query: QueryRequest) -> list[SearchResult]:
        """Called on a peer when it receives a query."""
        
        # Am I a leaf peer?
        if self.peer.is_leaf:
            # Search local vector store directly
            results = await self.peer.vector_store.search(
                query.embedding,
                top_k=query.top_k,
            )
            return results
        
        # I'm a routing peer — find matching child communities
        matching_communities = await self.search_community_embeddings(
            query_embedding=query.embedding,
            threshold=query.similarity_threshold,
        )
        
        # Forward query to peers owning those communities
        all_results = []
        child_peer_ids = set()
        for community in matching_communities:
            child_peer_ids.add(community.peer_id)
        
        # Parallel dispatch to child peers
        responses = await self.transport.broadcast(
            list(child_peer_ids),
            QueryMessage.from_request(query),
        )
        
        # Aggregate results from children
        for response in responses:
            all_results.extend(response.results)
        
        # Re-rank and return up to parent
        merged = self.reranker.merge_and_rerank(all_results, query.top_k)
        return merged
```

### Modify `query/engine.py`

The query engine is now simpler — it's the **client-side entrypoint**, not the full orchestrator:

```python
class QueryEngine:
    """
    Client-side query entrypoint. Sends query to a super-peer
    and receives the aggregated results.
    """
    
    async def query(self, text: str, top_k: int = 10) -> QueryResponse:
        # 1. Embed
        embedding = await self.embedder.embed_text(text)
        
        # 2. Extract topics
        topics = await self.llm.extract_topics(text)
        
        # 3. Build query request
        request = QueryRequest(
            query_id=str(uuid4()),
            text=text,
            embedding=embedding,
            top_k=top_k,
            topic_hints=topics,
        )
        
        # 4. Send to super-peer (it handles routing + leaf search + aggregation)
        super_peer = self.get_super_peer()
        response = await self.transport.send(super_peer.peer_id, QueryMessage.from_request(request))
        
        # 5. Generate answer from returned results
        answer = await self.generator.generate(text, response.results)
        
        return QueryResponse(
            query_id=request.query_id,
            answer=answer,
            results=response.results,
            routing_path=response.routing_path,
            latency_ms=response.latency_ms,
            total_messages=response.total_messages,
        )
```

---

## Change 5: KG Entities Live on the Chunk, Not Separate

The hackathon doc puts `entities` and `relations` directly on `DocumentChunk`, not in a separate KnowledgeGraph-only data structure. This simplifies ingestion:

```
Ingestion pipeline:
1. Chunk text (chunker.py)
2. Extract entities + relations per chunk (entity_extractor.py)
3. Store entities/relations ON the DocumentChunk
4. Embed the chunk
5. Store in vector store

Then separately for clustering:
6. Build KnowledgeGraph from all DocumentChunks (knowledge_graph.py reads chunks)
```

### Modify `ingestion/entity_extractor.py`

```python
class EntityExtractor:
    """
    Extracts entities and relations from a DocumentChunk using LLM.
    Results are stored DIRECTLY on the chunk object.
    """
    
    async def extract(self, chunk: DocumentChunk) -> DocumentChunk:
        """Mutates chunk in-place: populates entities and relations fields."""
        prompt = f"Extract entities and relationships from:\n{chunk.text}"
        response = await self.llm.generate(prompt)
        chunk.entities = self._parse_entities(response)
        chunk.relations = self._parse_relations(response)
        return chunk
    
    async def extract_batch(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Batch extraction with concurrency limit."""
        return await asyncio.gather(*[self.extract(c) for c in chunks])
```

### Modify `clustering/knowledge_graph.py`

The KG is built FROM chunks, not during ingestion:

```python
class KnowledgeGraph:
    """
    In-memory knowledge graph built from DocumentChunks.
    Nodes are entities, edges are relations.
    Each node stores which chunks it appears in and inherits their labels.
    """
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk's entities and relations to the graph."""
        for entity_name in chunk.entities:
            self._get_or_create_node(entity_name).add_chunk(chunk)
        for source, rel_type, target in chunk.relations:
            self._add_edge(source, target, rel_type, weight=1)
    
    def to_igraph(self) -> igraph.Graph:
        """Convert to igraph format for Leiden clustering."""
        ...
```

---

## Change 6: Serialization Uses msgpack, Not JSON

The hackathon doc specifies **msgpack for wire protocol** (compact binary), not JSON:

```python
# In networking/messages.py

import msgpack

@dataclass
class QueryMessage:
    message_id: str
    sender_peer_id: str
    timestamp: float
    query_id: str
    query_embedding: list[float]
    topic_hints: list[str]
    hop_count: int
    ttl: int
    
    def to_bytes(self) -> bytes:
        return msgpack.packb(self.to_dict())
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "QueryMessage":
        return cls.from_dict(msgpack.unpackb(data))
```

Add `msgpack` to `pyproject.toml` dependencies.

---

## Change 7: Database Schema — `vectors` Table, Not `chunks`

The hackathon doc uses `vectors` as the table name with `doc_id` column:

```sql
-- scripts/migrate_db.py generates this schema

CREATE EXTENSION IF NOT EXISTS vectorscale;

CREATE TABLE vectors (
    chunk_id UUID PRIMARY KEY,
    doc_id TEXT NOT NULL,
    embedding vector(768),          -- 768 for nomic, 1536 for OpenAI
    text_content TEXT,
    labels TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON vectors USING diskann (embedding);
CREATE INDEX ON vectors USING gin (labels);

-- Communities table stays the same
CREATE TABLE communities (
    community_id UUID PRIMARY KEY,
    level INT NOT NULL,
    summary TEXT NOT NULL,
    summary_embedding vector(768) NOT NULL,
    labels TEXT[] NOT NULL DEFAULT '{}',
    member_ids UUID[] NOT NULL DEFAULT '{}',
    parent_id UUID REFERENCES communities(community_id),
    children_ids UUID[] NOT NULL DEFAULT '{}',
    peer_id TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON communities USING diskann (summary_embedding);
CREATE INDEX ON communities (level);
```

Note: `labels` columns are kept in the schema even for MVP (just populated with empty arrays). This avoids a schema migration when access control is added later.

---

## Change 8: Embedding Default — nomic-embed-text at 768 dims

The hackathon doc defaults to **nomic-ai/nomic-embed-text-v1.5** via sentence-transformers, not OpenAI. This is significant because:
- Dimension is **768**, not 1536
- No API key required for local inference
- All vector columns and index declarations must use 768

Update all `vector(1536)` references to `vector(768)` in the schema and any hardcoded dimension references. The dimension should ultimately come from `config.embeddings.dimension`.

---

## Change 9: Leiden Resolution Schedule is Per-Level, Not Global

The hackathon doc specifies resolution as a **list** `[2.0, 1.0, 0.5]` for 3 levels:
- Level 0 (fine): resolution 2.0 (more, smaller communities)
- Level 1 (mid): resolution 1.0
- Level 2 (broad): resolution 0.5 (fewer, larger communities)

The `hierarchical_cluster()` method iterates through this list, using each resolution for one level of the hierarchy.

---

## Change 10: Routing in the Hackathon Doc Still References top-k + labels

Section 3, Stage 3 of the hackathon doc says:

> "Communities that are found within the similarity threshold (pruned to top-k) are followed to the next layer down."

And still references `user_labels` filtering. Per our previous changes:
- **Threshold-based selection is primary** (confirmed by hackathon doc)
- **top-k is a safety cap**, not the primary selection mechanism (confirmed: "pruned to top-k")
- **Labels filtering is post-MVP** but the infrastructure (empty label sets, schema columns) stays

No code change needed — the previous architecture changes file already specifies this correctly. The hackathon doc confirms the threshold + safety cap approach.

---

## Summary of All Changes

| # | Change | Impact |
|---|---|---|
| 1 | Flattened `src/` layout — single files instead of subdirectories | All import paths change |
| 2 | YAML config with env overlay | New `config/` directory, new config.py approach |
| 3 | Richer type system (`DocumentChunk`, `QueryResponse` with observability) | `core/types.py` rewrite |
| 4 | Super-peer orchestrates query (recursive descent/aggregation) | `networking/router.py` + `query/engine.py` redesign |
| 5 | Entities/relations live on DocumentChunk | `ingestion/entity_extractor.py` + `clustering/knowledge_graph.py` |
| 6 | msgpack wire serialization | Add dependency, `to_bytes()`/`from_bytes()` on all message types |
| 7 | `vectors` table name, `doc_id` column | `scripts/migrate_db.py` schema |
| 8 | nomic-embed-text default at 768 dims | Schema, config, all vector dimension references |
| 9 | Leiden resolution is per-level list | `clustering/leiden.py`, config |
| 10 | Confirms threshold + top-k safety cap for routing | No change needed (previous changes file covers this) |
