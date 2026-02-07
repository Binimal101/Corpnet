# DAC-HRAG Architecture Changes — Cursor Instructions

> **Priority**: Apply these changes across all relevant files before beginning implementation.
> **Scope**: These changes affect `core/`, `ingestion/`, `network/routing/`, `query/`, `api/`, and the Phase 1 implementation order.

---

## Change 1 + 5: Defer Access Control Past MVP

Auth and access control are **not part of the MVP**. Remove them from Phase 1 entirely. The permission label system, JWT auth, filtered search by labels, and all access control enforcement will be implemented in a later phase.

### What to remove/skip for MVP

**Do not create these files yet:**
- `core/auth.py`
- `api/middleware/auth_middleware.py`
- `api/routes/auth.py`
- `ingestion/labeling/` (entire directory)
- `tests/unit/test_auth.py`
- `tests/integration/test_filtered_search.py` (the access-control-specific one)
- `tests/distributed/test_access_control_e2e.py`

**Modify `core/types.py`:**

```python
# BEFORE
@dataclass
class Chunk:
    id: str
    text: str
    vector: Vector
    source: str
    labels: set[str]           # ← REMOVE for MVP
    metadata: dict[str, Any]

@dataclass
class Community:
    id: str
    level: int
    summary: str
    vector: Vector
    labels: set[str]           # ← REMOVE for MVP
    member_ids: list[str]
    peer_owners: list[str]

@dataclass
class QueryRequest:
    text: str
    vector: Vector
    topics: list[str]
    user_labels: set[str]      # ← REMOVE for MVP
    k: int = 10

# AFTER
@dataclass
class Chunk:
    id: str
    text: str
    vector: Vector
    source: str
    metadata: dict[str, Any]

@dataclass
class Community:
    id: str
    level: int
    summary: str
    vector: Vector
    member_ids: list[str]
    peer_owners: list[str]

@dataclass
class QueryRequest:
    text: str
    vector: Vector
    topics: list[str]
    k: int = 10
```

**Modify `indexing/vector_store/base.py`:**

```python
# BEFORE
class VectorStore(ABC):
    @abstractmethod
    def search(
        self,
        query: Vector,
        k: int,
        filter_labels: set[str] | None = None    # ← REMOVE for MVP
    ) -> list[QueryResult]: ...

# AFTER
class VectorStore(ABC):
    @abstractmethod
    def search(
        self,
        query: Vector,
        k: int,
    ) -> list[QueryResult]: ...
```

**Modify `indexing/vector_store/pgvector_store.py`:**
- Build the DiskANN index without label filtering for MVP
- Skip the GIN index on labels
- Remove `filter_labels` parameter from search queries

**Modify DB schema (`infra/scripts/init_db.sql`):**

```sql
-- MVP schema: no labels columns, no GIN indexes, no users/groups tables

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;

CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text        TEXT NOT NULL,
    source      TEXT NOT NULL,
    embedding   VECTOR(1536) NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX chunks_embedding_idx
    ON chunks
    USING diskann (embedding vector_cosine_ops);

CREATE TABLE communities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level           INT NOT NULL,
    summary         TEXT NOT NULL,
    embedding       VECTOR(1536) NOT NULL,
    member_ids      UUID[] NOT NULL DEFAULT '{}',
    parent_id       UUID REFERENCES communities(id),
    peer_owner_ids  TEXT[] NOT NULL DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX communities_embedding_idx
    ON communities
    USING diskann (embedding vector_cosine_ops);

CREATE INDEX communities_level_idx ON communities (level);
```

**Modify `api/server.py`:**
- Remove auth middleware from the middleware stack
- Remove `/auth/*` routes
- All endpoints are unauthenticated for MVP

**Modify `query/engine.py`:**
- Remove permission resolution step (Step 3 in query lifecycle)
- Remove `filter_labels` from all search calls
- The query lifecycle becomes: embed → extract topics → route → leaf search → aggregate → generate (6 steps, not 7)

**Modify Phase 1 exit criteria:**
- ~~"Same query, two users with different permissions, provably different result sets"~~ 
- New: "Query returns correct results from ingested data. Recall@10 > 85% vs brute-force baseline."

### Stub for later implementation

Leave a single file `core/auth.py` as a placeholder:

```python
"""
Access control module — NOT IMPLEMENTED IN MVP.

Post-MVP, this module will provide:
- resolve_permissions(user_token) -> PermissionSet
- check_access(PermissionSet, label_set) -> bool
- JWT validation and group claim extraction
- Permission label propagation through query routing

See docs/ACCESS_CONTROL.md for the full design.
"""

# TODO: Implement post-MVP
```

---

## Change 2: Peer Data Upload Format

The ingestion API accepts structured objects where each record is a key-value metadata object paired with its pre-computed embedding. Peers upload pre-embedded data — the system does **not** compute embeddings on behalf of the uploader in the standard ingestion path.

### New data model

The upload format is a list of records. Each record has two parts:
1. **`metadata`**: An object of arbitrary key-value pairs describing the entity (e.g., `{"name": "John Doe", "role": "engineer", "team": "backend"}`)
2. **`embedding`**: The pre-computed embedding vector for this record

```python
# In core/types.py — ADD this type

@dataclass
class IngestRecord:
    """
    A single record uploaded by a peer node.
    
    The peer provides pre-computed embeddings alongside structured metadata.
    The metadata keys are arbitrary and domain-specific — the system treats
    them as opaque key-value pairs stored in the vector store's metadata column.
    
    Example input (JSON):
    [
        {
            "metadata": {"name": "John Doe", "role": "engineer", "department": "backend"},
            "embedding": [0.0123, -0.0456, 0.0789, ...]
        },
        {
            "metadata": {"title": "CI Pipeline Architecture", "type": "document", "author": "Jane Smith"},
            "embedding": [0.0321, -0.0654, 0.0987, ...]
        }
    ]
    """
    metadata: dict[str, Any]
    embedding: list[float]
```

### Modify `api/schemas.py`

```python
from pydantic import BaseModel, Field

class IngestRecordSchema(BaseModel):
    metadata: dict[str, Any] = Field(
        ...,
        description="Arbitrary key-value pairs describing the entity",
        examples=[{"name": "John Doe", "role": "engineer"}]
    )
    embedding: list[float] = Field(
        ...,
        description="Pre-computed embedding vector"
    )

class IngestRequest(BaseModel):
    records: list[IngestRecordSchema] = Field(
        ...,
        description="List of records to ingest, each with metadata and pre-computed embedding"
    )
    source: str = Field(
        ...,
        description="Source identifier (e.g., 'team-backend', 'docs-v2')"
    )

class IngestResponse(BaseModel):
    ingested: int
    chunk_ids: list[str]
```

### Modify `api/routes/ingest.py`

```python
from api.schemas import IngestRequest, IngestResponse

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest pre-embedded records into the local peer's vector store.
    
    Each record is a metadata dict + embedding pair. The system stores
    the metadata as-is and indexes the embedding for similarity search.
    """
    chunk_ids = []
    for record in request.records:
        chunk = Chunk(
            id=generate_id(),
            text=serialize_metadata(record.metadata),  # Flatten for text search fallback
            vector=Vector(data=record.embedding, dim=len(record.embedding)),
            source=request.source,
            metadata=record.metadata,
        )
        chunk_id = await vector_store.insert(chunk)
        chunk_ids.append(chunk_id)
    
    return IngestResponse(ingested=len(chunk_ids), chunk_ids=chunk_ids)
```

### Helper: `serialize_metadata`

```python
def serialize_metadata(metadata: dict[str, Any]) -> str:
    """
    Convert metadata dict to a flat text representation for storage
    in the text column. This serves as a fallback for text-based search
    and as the content returned to the LLM for generation.
    
    {"name": "John Doe", "role": "engineer"} → "name: John Doe | role: engineer"
    """
    return " | ".join(f"{k}: {v}" for k, v in metadata.items())
```

### Modify `ingestion/pipeline.py`

The pipeline now has two paths:

1. **Pre-embedded path** (primary for peer uploads): Records arrive with embeddings already computed. Pipeline does: validate → store → register with local community tracker. No chunking, no embedding computation.

2. **Raw text path** (for file/connector ingestion): Raw documents still go through extract → chunk → embed → store. This path calls `core/embedding.py` to compute embeddings.

```python
class IngestionPipeline:
    
    async def ingest_records(self, records: list[IngestRecord], source: str) -> list[str]:
        """
        Pre-embedded path: records arrive with embeddings.
        No chunking or embedding computation needed.
        """
        chunk_ids = []
        for record in records:
            chunk = Chunk(
                id=generate_id(),
                text=serialize_metadata(record.metadata),
                vector=Vector(data=record.embedding, dim=len(record.embedding)),
                source=source,
                metadata=record.metadata,
            )
            chunk_id = await self.vector_store.insert(chunk)
            chunk_ids.append(chunk_id)
            self.community_tracker.on_new_chunk(chunk)  # Update community centroid
        
        return chunk_ids
    
    async def ingest_documents(self, documents: list[Document], source: str) -> list[str]:
        """
        Raw text path: documents need extraction, chunking, and embedding.
        """
        chunk_ids = []
        for doc in documents:
            text_chunks = self.chunker.chunk(doc)
            embeddings = await self.embedder.embed_batch([c.text for c in text_chunks])
            for chunk_text, embedding in zip(text_chunks, embeddings):
                chunk = Chunk(
                    id=generate_id(),
                    text=chunk_text.text,
                    vector=embedding,
                    source=source,
                    metadata=chunk_text.metadata,
                )
                chunk_id = await self.vector_store.insert(chunk)
                chunk_ids.append(chunk_id)
                self.community_tracker.on_new_chunk(chunk)
        
        return chunk_ids
```

### Modify DB schema

The `chunks` table already supports this via the `metadata` JSONB column. The arbitrary key-value pairs from the upload are stored directly in this column and are queryable via PostgreSQL's JSONB operators.

```sql
-- Example queries against metadata:
SELECT * FROM chunks WHERE metadata->>'name' = 'John Doe';
SELECT * FROM chunks WHERE metadata->>'department' = 'backend';
```

---

## Change 3: Routing Uses Embedding Similarity, Not Text Summaries

Clarify that at **every hierarchy layer**, routing decisions are made by **cosine similarity between the query embedding and community summary embeddings** — never by text comparison or LLM interpretation of text summaries.

The text summary is stored for human readability and for LLM context generation at the final answer step. It is **never used in the routing path**.

### Modify `network/routing/router.py`

```python
class HierarchicalRouter(Router):
    """
    Routes queries through the community hierarchy using ONLY embedding similarity.
    
    At each layer:
    1. Take the query embedding
    2. Compute cosine similarity against all community summary EMBEDDINGS at this layer
    3. Select communities above the similarity threshold (see Change 4)
    4. Follow inter-layer links to descend into those communities' children
    5. Repeat until leaf layer
    
    Text summaries are NOT used in routing. They exist only for:
    - Human inspection of community contents
    - LLM context at the final answer generation step
    - Debugging and observability
    """
    
    async def route(self, request: QueryRequest) -> list[QueryResult]:
        query_vec = request.vector
        
        # Start at top layer
        current_communities = self.hierarchy.get_level(self.hierarchy.top_level)
        
        for level in range(self.hierarchy.top_level, 0, -1):
            # Embedding-only similarity search at this layer
            # This searches the C-HNSW layer's vector store, NOT text
            matching = await self.chnsw.search_layer(
                level=level,
                query_vec=query_vec,
                similarity_threshold=request.similarity_threshold,
                candidates=current_communities,
            )
            
            # Descend: get children of matching communities
            current_communities = []
            for community in matching:
                children = self.hierarchy.get_children(community.id)
                current_communities.extend(children)
        
        # At leaf layer: parallel vector search on data-owning peers
        results = await self.search_leaf_peers(current_communities, query_vec, request.k)
        return results
```

### Modify `indexing/chnsw/index.py`

```python
class CHNSWIndex:
    """
    C-HNSW index: one VectorStore per hierarchy level.
    
    Each layer stores community summary EMBEDDINGS (not text).
    Search at each layer is pure vector similarity — cosine distance
    between query embedding and stored community embeddings.
    """
    
    def __init__(self):
        self.layers: dict[int, VectorStore] = {}  # level -> vector store of community embeddings
        self.inter_layer_links: dict[str, list[str]] = {}  # community_id -> child_ids
    
    async def search_layer(
        self,
        level: int,
        query_vec: Vector,
        similarity_threshold: float,
        candidates: list[Community] | None = None,
    ) -> list[Community]:
        """
        Search a single hierarchy layer by embedding similarity.
        
        Returns all communities whose summary embedding has cosine
        similarity >= similarity_threshold with the query embedding.
        
        No text processing, no LLM calls, no string matching.
        Pure vector math.
        """
        layer_store = self.layers[level]
        
        # Search returns results sorted by similarity
        results = await layer_store.search_by_threshold(
            query=query_vec,
            threshold=similarity_threshold,
            candidate_ids=[c.id for c in candidates] if candidates else None,
        )
        
        return results
```

### Modify `clustering/summarization/summarizer.py`

Make the dual purpose of summaries explicit:

```python
class CommunitySummarizer:
    """
    Generates community summaries that serve TWO distinct purposes:
    
    1. EMBEDDING (for routing): The summary text is embedded into a vector.
       This vector is what the router uses for similarity search.
       The text itself is never seen by the router.
    
    2. TEXT (for generation): The summary text is included in LLM context
       at the final answer generation step, so the LLM understands what
       each community represents.
    
    The quality of the summary text directly impacts both:
    - Routing accuracy (via the embedding it produces)
    - Answer quality (via the context it provides to the LLM)
    """
    
    async def summarize(self, community: Community) -> tuple[str, Vector]:
        """
        Returns (text_summary, summary_embedding).
        
        The text_summary is stored for human readability and LLM context.
        The summary_embedding is what gets indexed in the C-HNSW layer
        and used for all routing decisions.
        """
        text_summary = await self.llm.summarize_community(
            entities=[...],
            relations=[...],
        )
        summary_embedding = await self.embedder.embed_text(text_summary)
        return text_summary, summary_embedding
```

---

## Change 4: Similarity Threshold Instead of Top-K for Community Descent

Replace fixed top-k community selection with a **cosine similarity threshold**. At each hierarchy layer, the router descends into **all communities whose summary embedding exceeds the threshold** — not a fixed number.

### Why this matters

- **Focused queries** (e.g., "What is the Rust borrow checker?") match strongly to one or two communities. With top-k=5, you'd waste 3-4 hops searching irrelevant communities. With threshold, you search only the 1-2 that matter.
- **Broad queries** (e.g., "How do our microservices communicate?") match moderately to many communities. With top-k=5, you might miss relevant ones. With threshold, you search all that are above the relevance floor.
- **The system self-tunes**: query specificity determines fan-out automatically.

### Modify `core/types.py`

```python
@dataclass
class QueryRequest:
    text: str
    vector: Vector
    topics: list[str]
    similarity_threshold: float = 0.35   # Cosine similarity floor for community descent
    k: int = 10                          # Final top-k results to return after leaf search

    # The threshold controls how many communities are explored at each layer.
    # Lower threshold = more communities searched = higher recall, higher latency.
    # Higher threshold = fewer communities = lower recall, lower latency.
    # Default 0.35 is calibrated for a good recall/latency tradeoff.
```

### Modify `core/config.py`

```python
# Routing
DEFAULT_SIMILARITY_THRESHOLD = 0.35      # Cosine similarity floor for community descent
MIN_COMMUNITIES_PER_LAYER = 1           # Always descend into at least 1 community
MAX_COMMUNITIES_PER_LAYER = 20          # Safety cap to prevent fan-out explosion

# The threshold is the PRIMARY routing control.
# MIN/MAX are safety bounds only — the threshold should handle normal cases.
```

### Modify `indexing/vector_store/base.py`

Add a threshold-based search method alongside the existing top-k method:

```python
class VectorStore(ABC):
    
    @abstractmethod
    def search(self, query: Vector, k: int) -> list[QueryResult]:
        """Top-k search. Used at leaf layer for final result retrieval."""
        ...
    
    @abstractmethod
    def search_by_threshold(
        self,
        query: Vector,
        threshold: float,
        candidate_ids: list[str] | None = None,
        max_results: int = 20,
    ) -> list[QueryResult]:
        """
        Threshold-based search. Returns ALL results with cosine similarity >= threshold.
        Used at hierarchy layers for community routing.
        
        Args:
            query: Query embedding vector
            threshold: Minimum cosine similarity (0.0 to 1.0)
            candidate_ids: Optional filter to only search among specific IDs
            max_results: Safety cap to prevent unbounded result sets
        
        Returns:
            Results with score >= threshold, sorted by descending similarity.
            Always returns at least 1 result (the best match) even if below threshold.
        """
        ...
```

### Modify `indexing/vector_store/pgvector_store.py`

```python
class PgVectorStore(VectorStore):
    
    async def search_by_threshold(
        self,
        query: Vector,
        threshold: float,
        candidate_ids: list[str] | None = None,
        max_results: int = 20,
    ) -> list[QueryResult]:
        """
        PostgreSQL implementation of threshold-based similarity search.
        
        Uses cosine distance (1 - cosine_similarity) with pgvector's <=> operator.
        Filters results where cosine_similarity >= threshold.
        """
        # pgvector uses cosine distance: 0 = identical, 2 = opposite
        # cosine_similarity = 1 - cosine_distance
        # threshold on similarity = (1 - threshold) on distance
        max_distance = 1.0 - threshold
        
        if candidate_ids:
            query_sql = """
                SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS similarity
                FROM communities
                WHERE id = ANY($2)
                AND (embedding <=> $1::vector) <= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            """
            rows = await self.conn.fetch(query_sql, query.data, candidate_ids, max_distance, max_results)
        else:
            query_sql = """
                SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS similarity
                FROM communities
                WHERE (embedding <=> $1::vector) <= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            rows = await self.conn.fetch(query_sql, query.data, max_results, max_distance)
        
        results = [self._row_to_result(row) for row in rows]
        
        # Guarantee: always return at least 1 result (best match),
        # even if nothing exceeds threshold. This prevents dead-end routing.
        if not results:
            fallback_sql = """
                SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS similarity
                FROM communities
                ORDER BY embedding <=> $1::vector
                LIMIT 1
            """
            rows = await self.conn.fetch(fallback_sql, query.data)
            results = [self._row_to_result(row) for row in rows]
        
        return results
```

### Modify `network/routing/router.py`

```python
class HierarchicalRouter(Router):
    
    async def route(self, request: QueryRequest) -> list[QueryResult]:
        query_vec = request.vector
        threshold = request.similarity_threshold
        
        current_communities = self.hierarchy.get_level(self.hierarchy.top_level)
        
        for level in range(self.hierarchy.top_level, 0, -1):
            # Threshold-based search: descend into ALL communities above threshold
            matching = await self.chnsw.search_layer(
                level=level,
                query_vec=query_vec,
                similarity_threshold=threshold,
                candidates=current_communities,
            )
            
            # Safety bounds
            if len(matching) < MIN_COMMUNITIES_PER_LAYER:
                # Threshold too aggressive — fallback to best match
                matching = matching[:1]  # search_by_threshold guarantees >= 1
            elif len(matching) > MAX_COMMUNITIES_PER_LAYER:
                # Too many matches — take only the top MAX to cap latency
                matching = matching[:MAX_COMMUNITIES_PER_LAYER]
            
            # Descend to children of matching communities
            current_communities = []
            for community in matching:
                children = self.hierarchy.get_children(community.id)
                current_communities.extend(children)
        
        # Leaf layer: standard top-k search on data peers
        results = await self.search_leaf_peers(current_communities, query_vec, request.k)
        return results
```

### Modify query engine to expose threshold

```python
# In query/engine.py

class QueryEngine:
    async def query(
        self,
        text: str,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        k: int = 10,
    ) -> Answer:
        vector = await self.embedder.embed_text(text)
        topics = await self.llm.extract_topics(text)
        
        request = QueryRequest(
            text=text,
            vector=vector,
            topics=topics,
            similarity_threshold=similarity_threshold,
            k=k,
        )
        
        results = await self.router.route(request)
        filtered = self.adaptive_filter.filter(results, vector)
        answer = await self.generator.generate_answer(text, filtered)
        return answer
```

### Modify `api/routes/query.py`

```python
class QueryRequestSchema(BaseModel):
    text: str
    similarity_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Cosine similarity floor for community routing. "
                    "Lower = broader search (higher recall, higher latency). "
                    "Higher = focused search (lower recall, lower latency)."
    )
    k: int = Field(default=10, ge=1, le=100, description="Number of final results to return")
```

---

## Summary of Changes

| # | Change | Files Affected |
|---|--------|---------------|
| 1+5 | Defer auth/access control past MVP | `core/types.py`, `core/auth.py` (stub only), `indexing/vector_store/base.py`, `indexing/vector_store/pgvector_store.py`, `infra/scripts/init_db.sql`, `api/server.py`, `api/routes/auth.py` (skip), `api/middleware/auth_middleware.py` (skip), `query/engine.py`, `ingestion/labeling/` (skip) |
| 2 | Peer upload format: metadata + embedding pairs | `core/types.py` (add IngestRecord), `api/schemas.py`, `api/routes/ingest.py`, `ingestion/pipeline.py` |
| 3 | Routing uses embedding similarity only, not text | `network/routing/router.py`, `indexing/chnsw/index.py`, `clustering/summarization/summarizer.py` (docstrings) |
| 4 | Similarity threshold instead of top-k for descent | `core/types.py` (QueryRequest), `core/config.py`, `indexing/vector_store/base.py`, `indexing/vector_store/pgvector_store.py`, `network/routing/router.py`, `query/engine.py`, `api/routes/query.py`, `api/schemas.py` |

### Updated Phase 1 Scope

Phase 1 now delivers:

1. `core/types.py` — shared dataclasses (Chunk, Community, QueryRequest, IngestRecord — NO labels/permissions)
2. `core/config.py` — env var loading including similarity threshold defaults
3. `core/embedding.py` — embedding abstraction (for raw text path only)
4. `core/llm.py` — LLM wrapper
5. `ingestion/pipeline.py` — dual-path: pre-embedded records (primary) + raw text (secondary)
6. `indexing/vector_store/pgvector_store.py` — DiskANN index, top-k search, threshold-based search (NO label filtering)
7. `query/engine.py` — embed → route (threshold-based) → search → generate (NO auth step)
8. `api/routes/ingest.py` — POST /ingest accepting `[{metadata, embedding}]`
9. `api/routes/query.py` — POST /query with configurable similarity_threshold

**Phase 1 exit criteria:**
- Ingest 1,000+ records via the metadata+embedding API
- Query returns correct results with threshold-based routing
- Recall@10 > 85% vs brute-force baseline
- Threshold parameter demonstrably controls recall/latency tradeoff
