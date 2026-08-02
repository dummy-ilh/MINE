# RAG Interview Master Notes — Module 3: Indexing & Vector Databases

> **How to use these notes:** This module is where RAG becomes a systems engineering problem. Interviewers test whether you can reason about scale tradeoffs quantitatively — not just name algorithms. The napkin math in section 7 and the Q&A drill are high-signal differentiators.

---

## Quick Summary

Indexing is the infrastructure layer that makes retrieval fast at scale. Exact nearest-neighbour search is provably correct but O(N·d) — completely infeasible for millions of vectors at interactive latency. Approximate Nearest Neighbour (ANN) algorithms trade a small amount of recall for orders-of-magnitude speedup, using three core strategies: graph-based navigation (HNSW), cluster-based pruning (IVF), and vector compression (Product Quantization). The right choice depends on your scale, update frequency, memory budget, and whether you need metadata filtering — and the ability to do back-of-envelope memory math live in an interview is a strong differentiator.

---

## 1. The Core Problem: Why Exact Search Doesn't Scale

### What Exact kNN Actually Does

Given a query vector q and a corpus of N document vectors, exact k-nearest-neighbour search:

1. Computes distance from q to **every** vector in the corpus
2. Sorts all N distances
3. Returns the k smallest

**Time complexity:** O(N · d) per query, where N = number of vectors, d = embedding dimension.

### The Numbers That Make This Unacceptable

> **Think of it like this:** Exact kNN is like finding the nearest coffee shop by walking to every coffee shop in the city, measuring the distance, writing it down, then sorting your list. Approximate search is like using a city map — you might miss the café hidden in an alley, but you find a great option in seconds.

**Concrete example:**

```
N = 10,000,000 documents
d = 768 dimensions (typical BERT-class embedding)
Distance computation per pair ≈ 768 multiply-accumulate operations

Total operations per query = 10M × 768 = 7.68 billion operations
At 10 GFLOPS (single CPU core) ≈ 0.77 seconds per query

Target interactive latency: < 50ms
```

You're off by ~15x on a single query, before any overhead. And this scales linearly with N — doubling the corpus doubles query time.

**This is why every vector database and index in this module exists:** they all approximate kNN faster than brute force, with different tradeoffs on speed, memory, and recall.

### The ANN Tradeoff Framing

ANN search introduces a tunable tradeoff:

```
Recall@k = (relevant vectors in ANN top-k) / (relevant vectors in exact top-k)
```

Recall@10 of 0.95 means: for every 10 results you return, you're missing ~0.5 that exact search would have found. In practice, 0.90–0.97 recall is acceptable for most RAG applications — the downstream LLM can compensate for occasional missed results.

---

## 2. ANN Algorithm Families

### 2.1 HNSW — Hierarchical Navigable Small World

> **Think of it like this:** HNSW is like a transit system with express routes (top layers) and local stops (bottom layer). You take the express train to the right neighbourhood, then walk the last block. You'd never start on the local line — you'd be walking forever.

#### How It Works

HNSW builds a multi-layer graph over your vector corpus:

```
Layer 2 (sparsest — "highway"):    A -------- E -------- I
                                        \                /
Layer 1 (medium density):          A -- C -- E -- G -- I
                                      \       \
Layer 0 (dense — "local streets"): A-B-C-D-E-F-G-H-I-J-K
```

**Search algorithm:**
1. Enter at a random node in the top layer
2. Greedily navigate toward the query vector (move to the nearest neighbour of the current node)
3. When no neighbour is closer than the current node → drop down one layer
4. Repeat until you reach Layer 0, where you perform fine-grained search

**Why this works:** The sparse upper layers let you jump across the full vector space in a few hops; the dense bottom layer finds the precise nearest neighbours once you're in the right region.

#### Key Tunable Parameters

| Parameter | What it controls | Effect of increasing |
|-----------|-----------------|----------------------|
| `M` | Max connections per node (graph degree) | Better recall, more memory (O(M) per node) |
| `efConstruction` | Search depth during index building | Better graph quality, slower build time |
| `efSearch` | Search depth during query time | Better recall, slower queries — **main serving-time knob** |

#### HNSW Tradeoffs

| Strength | Weakness |
|----------|----------|
| Excellent recall/speed tradeoff | Graph must be held in memory (expensive at billion-scale) |
| Online/incremental insertions — no retraining needed | Deletion is nontrivial (see section 6) |
| Widely implemented (FAISS, Qdrant, Weaviate, pgvector) | Build time slower than IVF |

---

### 2.2 IVF — Inverted File Index

> **Think of it like this:** IVF is like a library organised by topic. When you search for "machine learning papers," you don't search every shelf — you go to the computer science section and search there. You might miss a ML paper misfiled under engineering, but you save enormous time.

#### How It Works

**Build time:**
1. Run k-means clustering on all vectors → `nlist` cluster centroids
2. Assign each vector to its nearest centroid
3. Store vectors grouped by cluster

**Query time:**
1. Find the `nprobe` closest cluster centroids to the query vector
2. Search only within those `nprobe` clusters (skip the rest)
3. Return top-k results from the searched clusters

```
Total vectors searched = (N / nlist) × nprobe

Example: N=10M, nlist=1000, nprobe=10
Vectors searched = (10M / 1000) × 10 = 100,000
Speedup ≈ 10M / 100K = 100×  (versus brute force)
```

#### The `nprobe` Dial

`nprobe` is the single most important serving-time knob for IVF:

| `nprobe` | Behaviour |
|---------|-----------|
| 1 | Search only the closest cluster — fastest, lowest recall |
| `nlist` | Search all clusters — equivalent to brute force, perfect recall |
| 10–50 (typical) | Balanced tradeoff; tune empirically against your eval set |

#### The Cluster Staleness Problem

> **Interview gotcha worth flagging proactively:** "IVF cluster centroids go stale."

IVF cluster centroids are trained on a **snapshot** of the data at index-build time. As new documents are inserted and the data distribution shifts, the original centroids no longer accurately represent the actual data layout. Result: vectors get assigned to suboptimal clusters, and the query's `nprobe` nearest centroids miss increasingly many relevant vectors. This degradation is **silent** — you won't see an error, just slowly worsening recall.

**Fix:** Monitor recall on a held-out eval set. Rebuild/retrain the index periodically (hourly, daily, weekly — depends on update rate). For continuous high-frequency updates, prefer HNSW which doesn't have this staleness problem.

---

### 2.3 IVF-PQ — IVF + Product Quantization

#### What Product Quantization (PQ) Adds

PQ solves a different problem from IVF: **memory**. IVF speeds up search by limiting which vectors you compare against. PQ reduces the storage cost of each vector.

**How PQ compresses vectors:**

```
Original vector: 768 float32 values × 4 bytes = 3,072 bytes per vector

PQ compression:
1. Split the 768-dim vector into M sub-vectors of d/M dimensions each
   (e.g. M=96 sub-vectors of 8 dims each)
2. For each sub-dimension space, train a small codebook of K* centroids
   (e.g. K*=256 centroids — fits in 1 byte per sub-vector)
3. Replace each sub-vector with the index of its nearest codebook centroid

Compressed: 96 sub-vectors × 1 byte = 96 bytes per vector
Compression ratio: 3,072 / 96 = 32×
```

At 100M vectors: 300GB (raw) → ~9.4GB (PQ-compressed) — now fits in a single machine's RAM.

#### The Accuracy Cost

PQ distances are **approximate** — you're comparing a query sub-vector against a codebook centroid, not the actual stored sub-vector. This introduces quantisation error on top of the IVF recall penalty.

**Standard mitigation:** Two-stage retrieval.

```
Stage 1: IVF-PQ search over full index
  → Fast, cheap, approximate
  → Returns top-k' candidates (e.g. k'=100)

Stage 2: Exact re-score the k' candidates with full-precision vectors
  → Slow but tiny (only 100 vectors, not 10M)
  → Re-ranks to get final top-k (e.g. k=10)
```

This is the same two-stage pattern as bi-encoder + cross-encoder reranking from Module 1 — just applied at the index level. Good to connect these dots explicitly in interviews.

---

### 2.4 ScaNN — Google's Anisotropic Quantisation

Standard PQ minimises overall quantisation error uniformly across all dimensions. ScaNN's insight: **not all quantisation errors are equally bad for ranking**.

For inner-product search, an error in the direction that affects the dot-product ranking (the "parallel" component relative to the query) matters far more than an error orthogonal to the query direction (which barely changes the ranking).

ScaNN penalises parallel-direction errors more during codebook training, producing quantised vectors that preserve ranking order better than standard PQ at the same compression ratio.

**Result:** ScaNN consistently near the top of [ann-benchmarks.com](https://ann-benchmarks.com) leaderboards on recall vs queries-per-second. Implemented in Google Vertex AI Matching Engine. Know it exists and the *why* — you don't need implementation depth for most interviews.

---

## 3. FAISS: The Building Blocks Library

FAISS (Facebook/Meta AI Similarity Search) is a C++ library with Python bindings — you self-host and choose your index type explicitly. It is not a database; it provides the ANN algorithm implementations that many vector databases build on top of.

### Index Type Reference Table

| Index | Mechanism | Memory | Speed | Recall | Use when |
|-------|-----------|--------|-------|--------|----------|
| `IndexFlatL2` / `IndexFlatIP` | Brute-force exact search | High (full-precision) | Slow — O(N·d) | Perfect (100%) | ≤100K vectors, or as ground-truth baseline to measure ANN recall loss |
| `IndexIVFFlat` | IVF clustering + exact search within clusters | Medium | Fast | High | Medium corpora, memory not constrained, need better recall than PQ |
| `IndexIVFPQ` | IVF + product quantisation | Very low | Very fast | Lower | Large corpora (100M+) where memory is the binding constraint |
| `IndexHNSWFlat` | HNSW graph, full-precision storage | High | Very fast | Very high | Best recall/speed tradeoff when you can afford full-precision RAM |

### Library vs System: The Key Interview Distinction

FAISS gives you the ANN algorithms. You own:
- **Persistence** (FAISS indexes are in-memory; you must serialise/load them)
- **Metadata storage** (FAISS stores only vectors and integer IDs — metadata lives separately)
- **Filtering** (no native metadata filtering — you implement it on top)
- **Sharding** (FAISS is single-node — you implement distribution yourself)
- **Updates** (FAISS has limited deletion support — you manage tombstoning)

This is why managed vector databases exist: they wrap FAISS (or equivalent) in a production-ready system that handles all of the above.

---

## 4. Managed Vector Databases

### When to Use Which: The Decision Framework

The real decision driver is **operational context**, not feature checklists:

```
Do you already run Postgres in production?
  → Yes → pgvector (avoid a new system dependency)
  → No  ↓

Do you need zero ops / fully managed / fastest time to production?
  → Yes → Pinecone

Do you need self-hosting for compliance / data residency?
  → Yes → Weaviate, Qdrant, or Milvus (open-source, self-hostable)
       → Need GraphQL / rich schema? → Weaviate
       → Need strongest filter performance? → Qdrant
       → Need billion-scale + deep customisation? → Milvus
```

### Feature Comparison (Know the Differences, Not Just the Names)

| | Pinecone | Weaviate | Qdrant | Milvus | pgvector |
|---|---|---|---|---|---|
| **Ops model** | Fully managed SaaS | Open-source + managed option | Open-source + managed option | Open-source + managed option | Postgres extension |
| **Hybrid search** | Native (sparse-dense) | Native (BM25 + vector fusion) | Native | Supported | Manual (combine with Postgres full-text) |
| **Metadata filtering** | Strong | Strong (GraphQL) | Strong, filter-aware traversal | Strong | SQL `WHERE` clause |
| **Horizontal scale** | Automatic (proprietary) | Distributed mode | Distributed mode | Cloud-native distributed | Limited (Postgres sharding) |
| **Best for** | Zero-ops teams | Rich schema + graph features | Filter-heavy workloads | Very large scale | Teams already on Postgres |

### The pgvector Case — Why It Wins More Often Than Expected

pgvector lets you add a `vector` column to any Postgres table:

```sql
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT,
    department  TEXT,
    updated_at  TIMESTAMPTZ,
    embedding   VECTOR(768)    -- new pgvector column
);

CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Combined vector + metadata query in one SQL statement:
SELECT content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE department = 'legal'
  AND updated_at > NOW() - INTERVAL '1 year'
ORDER BY embedding <=> $1
LIMIT 10;
```

**Why this matters:** Your metadata and vectors are in the same transaction scope — no sync problem between a separate metadata store and a separate vector index. At startup scale (≤ a few million vectors), this operational simplicity often outweighs the raw performance advantage of a dedicated vector DB.

---

## 5. Metadata Filtering: Pre-Filter vs Post-Filter vs Hybrid

### The Setup

Suppose a query needs: "find vectors similar to q, but **only** from `department=legal`."

This is the norm in production RAG — almost every real deployment has some filtering requirement (by user, by document type, by date, by access permission).

### Three Approaches, Each With a Failure Mode

#### Post-Filtering (Filter After Search)

```
Run ANN search on full index → top-100 by similarity
Apply metadata filter: keep only department=legal
Return up to k results from filtered set
```

**Failure mode — highly selective filters:**

If only 2% of the corpus is `department=legal`, the top-100 ANN results probably contain ~2 legal documents on average. You asked for k=10 but get 2. Worse, there might be 50 highly relevant legal documents at similarity rank 150–200 that were never considered.

#### Pre-Filtering (Filter Before Search)

```
Apply metadata filter: restrict candidate set to department=legal vectors only
Run ANN search on filtered subset
Return top-k
```

**Failure mode — high-cardinality filters:**

> **Think of it like this:** Your IVF index was trained on 10 million documents. Now you filter to 500 documents belonging to one specific user. ANN structures built on 10M vectors don't gracefully degrade to 500-vector search — you've essentially disabled the index.

If `department=legal` contains only 500 documents, the ANN structures built on the full 10M-vector corpus are useless for this tiny subset. You fall back to brute-force over 500 vectors (fast, but now you've gained nothing from the index). Worse, if you're filtering by `user_id` with thousands of users, you'd need a separate index per user — completely unscalable.

#### Hybrid / Filter-Aware Traversal (What Production Systems Actually Do)

Push filter awareness **into the graph traversal itself**:

During HNSW graph traversal, when visiting a candidate node, check its metadata. If it fails the filter, skip it (don't count it toward your efSearch budget), but continue exploring its neighbours (which might satisfy the filter).

```
Query: similarity to q, filter: department=legal

HNSW traversal:
  Visit node A (department=finance) → filtered out, but explore A's neighbours
  Visit node B (department=legal)   → keep, add to result set
  Visit node C (department=legal)   → keep, add to result set
  Visit node D (department=HR)      → filtered out, explore neighbours
  ...
```

This avoids both failure modes — you never reduce the candidate pool before search, and you don't waste result slots on non-matching post-filter candidates. Requires native support in the index implementation (Qdrant and Weaviate both implement this).

### Decision Summary

| Approach | Use when | Breaks when |
|----------|----------|-------------|
| Post-filter | Filters are loosely selective (most docs match) | Filter is highly selective (very few matches in top-k) |
| Pre-filter | Filter cardinality is low (few distinct values, each with many docs) | Filter cardinality is high (many distinct values, each with few docs) |
| Hybrid traversal | General purpose — prefer this | Not supported by your index implementation |

---

## 6. Index Update Strategies

### Real-Time Upsert (HNSW-based systems)

HNSW supports incremental insertion natively — adding a new node means connecting it into the existing graph layers. No retraining required.

**Good for:** Live support ticket systems, real-time document ingestion, any corpus with continuous small-volume updates.

**Caveat:** Each insertion updates the graph in-place. Under very high concurrent insert load, graph quality can degrade slightly — most implementations use locking or lock-free structures to mitigate this.

### Batch Rebuild (IVF-based systems)

IVF cluster centroids are trained on a snapshot. They don't update automatically as new data arrives.

**The silent decay pattern:**
```
T=0:  Train centroids on 1M docs → good cluster boundaries
T=3mo: Insert 200K new docs → assigned to old centroids
T=6mo: Recall@10 has silently dropped from 0.95 to 0.87
T=?:  Users notice degraded answer quality — hard to diagnose
```

**Fix:** Instrument recall on a held-out eval set. Set alerts. Rebuild on a schedule or when recall drops below threshold.

**Mandatory rebuild trigger:** Any embedding model migration. See Q&A drill.

### Handling Deletes — The Tombstone Pattern

Most ANN structures don't support true fast deletion (removing a node from an HNSW graph requires repairing the graph connections — expensive). Standard pattern:

```
Delete operation:
  1. Mark vector as tombstone in metadata (soft delete)
  2. Filter tombstoned vectors out at query time (check metadata before returning results)
  3. Periodic compaction: rebuild index, skipping tombstoned vectors (reclaims storage, removes them from graph)
```

**Why not just delete immediately?** Graph repair after node deletion is O(M·log(N)) per deletion and can't be done cheaply under high-throughput write loads. Tombstoning costs O(1) and defers the structural cleanup to scheduled compaction.

---

## 7. Scaling: Memory Math and Sharding

### The Memory Calculation Every Interviewer Loves

> **Differentiator:** Being able to do this napkin math live — casually and correctly — signals that you've actually built these systems.

**Base formula:**

```
Memory (raw vectors) = N × d × bytes_per_float

For float32: bytes_per_float = 4
For float16: bytes_per_float = 2
```

**Worked examples:**

| Scenario | N | d | Format | Raw memory | Notes |
|----------|---|---|--------|------------|-------|
| Small startup | 1M | 384 | float32 | 1.5 GB | Fits on one machine easily |
| Medium product | 10M | 768 | float32 | 30 GB | Fits on beefy single machine |
| Large enterprise | 100M | 768 | float32 | 300 GB | Exceeds single machine RAM |
| Web-scale | 1B | 1536 | float32 | 6 TB | Requires PQ + sharding |

Plus HNSW graph overhead: approximately `4 × M × N × 4 bytes` for the adjacency lists, where M≈16 typically → roughly 0.5–1× the raw vector size in additional memory.

**The PQ Compression Calculation:**

```
Original: 768-dim float32 = 3,072 bytes/vector

PQ with M=96 sub-vectors, K*=256 codes:
  Storage per vector = M × log₂(K*) bits = 96 × 8 bits = 96 bytes/vector
  Compression ratio = 3,072 / 96 = 32×

At 100M vectors:
  Raw:          300 GB
  PQ-compressed:  9.4 GB  ← fits in RAM on a single machine
```

### Sharding Strategies

#### Random / Hash Sharding

```
For each vector, assign to shard = hash(vector_id) % num_shards
```

**Pros:** Simple, perfectly balanced shards.  
**Cons:** Every query must fan out to **all** shards (scatter-gather pattern), because you have no idea which shard holds relevant vectors. Query latency = max(shard_latency) — tail latency of the slowest shard on every query.

#### Semantic / Cluster-Based Sharding

```
Train coarse clusters → assign each cluster to a shard
Route query to shards whose cluster centroids are close to the query
```

**Pros:** Queries only fan out to relevant shards — reduces fan-out by (num_shards / num_relevant_clusters).  
**Cons:** Routing complexity; unbalanced shard sizes if cluster sizes are uneven; requires routing layer aware of shard topology.

**When to use which:**

| | Random sharding | Semantic sharding |
|---|---|---|
| **Simplicity** | Much simpler | Complex routing layer |
| **Load balance** | Perfect | Can be skewed |
| **Fan-out** | Always full fan-out | Partial fan-out for selective queries |
| **Use when** | Small number of shards (≤10) | Many shards, query latency is critical |

---

## Interview Q&A Drill

---

**Q: Your team wants to add real-time document updates to a RAG system currently using FAISS IndexIVFPQ. What's the issue and what would you recommend?**

A: Two issues. First, IVF cluster centroids were trained on a snapshot of the data — new vectors are inserted into the existing clusters, but as more new data arrives, the original centroids stop representing the actual data distribution. Recall degrades silently over time without any error signal. Second, FAISS IndexIVFPQ has limited production infrastructure around it — no native metadata filtering, no built-in persistence, no deletion support beyond rebuilding.

For real-time updates I'd recommend switching to an HNSW-based index, either FAISS `IndexHNSWFlat` if staying in FAISS, or a managed system like Qdrant or Weaviate that provides HNSW with production infrastructure. HNSW supports incremental insertion natively with no retraining required. If we must keep IVF-PQ for its memory efficiency at scale, I'd instrument recall on a held-out eval set, set alerts for recall degradation, and schedule periodic full index rebuilds — and design the ingestion pipeline to buffer new documents for batch-indexed updates rather than per-document real-time upserts.

---

**Q: Walk me through pre-filtering vs post-filtering and when each breaks.**

A: Post-filtering runs ANN search first on the full index, then discards results that fail the metadata filter. This breaks when the filter is highly selective — if only 1% of documents are `department=legal`, you search top-100 by similarity and end up with ~1 legal document on average, even though there might be 50 highly relevant legal documents sitting just outside your top-100 window.

Pre-filtering restricts the candidate set to matching vectors before running ANN search. This breaks with high-cardinality filters — filtering to a specific `user_id` with only 200 associated documents means you're running ANN search on a 200-vector subset, where the index structures trained on the full corpus provide no benefit. At extreme cardinality you'd need a separate index per user, which doesn't scale.

Production systems solve this with filter-aware traversal integrated into the ANN algorithm itself — during HNSW graph traversal, non-matching nodes are skipped but their neighbours are still explored, so the search effectively self-routes toward the relevant region of the graph without pre-restricting the candidate set. This is what Qdrant and Weaviate implement natively.

---

**Q: You need to migrate from embedding model A to embedding model B. Walk through what breaks if you do this naively.**

A: If you embed new documents with model B while leaving existing documents embedded with model A, you have vectors from two geometrically incompatible spaces in the same index. Cosine similarity between a model-B query vector and a model-A document vector is meaningless — they don't share a coordinate system. The retriever will silently return garbage for any query that should match old documents, with no error signal. Monitoring won't flag it unless you specifically track recall on a held-out eval set that includes old-document queries.

The correct approach: full re-embed of the entire corpus with model B, full index rebuild, then a blue-green swap — build the new index offline, validate recall against the eval set, then cut over traffic atomically. This is why embedding model migrations are expensive and should be planned carefully rather than done incrementally.

---

**Q: When would pgvector be the right choice over Pinecone, even though it's less specialised?**

A: When you already run Postgres for the application's primary data. The alternative — Postgres for metadata, Pinecone for vectors — creates a sync problem: every document insert/update/delete must be applied transactionally to both systems. If one fails and the other succeeds, your metadata and vector index are inconsistent. With pgvector, both live in the same Postgres transaction, eliminating this class of bug.

pgvector is the right tradeoff at moderate scale (typically ≤ a few million vectors) where you don't yet need Pinecone-grade horizontal scaling, and where operational simplicity, transactional consistency, and avoiding a new system dependency outweigh raw ANN performance. Once you're scaling to tens of millions of vectors with strict latency SLAs, the specialised systems pull ahead.

---

**Q: How much RAM does a 50-million-vector HNSW index need for 1536-dim OpenAI embeddings stored in float32?**

A: Start with raw vector storage:

```
50M vectors × 1536 dims × 4 bytes (float32) = 307 GB
```

Add HNSW graph overhead — the adjacency list storage for M=16 connections per node:

```
Graph overhead ≈ 50M nodes × 16 connections × 4 bytes per ID × 2 layers average
              ≈ 50M × 128 bytes ≈ 6.4 GB
```

Total ≈ 313 GB. That exceeds a standard cloud machine's RAM (typically 128–256 GB for memory-optimised instances), so you have two options: use float16 to halve vector storage to ~154 GB (within range of a large instance), apply Product Quantisation to cut storage 16–32×, or shard across multiple machines. For 50M vectors at this dimensionality, I'd evaluate float16 first — it halves memory with minimal precision loss for similarity search, and only move to PQ if float16 still exceeds budget.

---

## Key Gotchas Summary

| Gotcha | Correct understanding |
|--------|----------------------|
| "Just use HNSW for everything" | HNSW requires full-precision vectors in memory — at billion-scale this is prohibitive; IVF-PQ may be necessary |
| "IVF degrades gracefully with updates" | IVF cluster centroids go stale silently as data distribution shifts; monitor recall and rebuild periodically |
| "Increasing nprobe always helps recall" | True up to nlist (full search), but at the cost of query time — there's a knee-of-curve tradeoff to find empirically |
| "Pre-filtering is obviously better than post-filtering" | Breaks with high-cardinality filters; filter-aware traversal is the production answer |
| "You can incrementally migrate to a new embedding model" | Impossible — vectors from different models are geometrically incompatible; always full re-embed + rebuild |
| "FAISS is a vector database" | FAISS is a library — no persistence, no filtering, no sharding, no deletion. Vector databases wrap it with production infrastructure |
| "Deleting a vector from HNSW is cheap" | Node deletion requires graph repair — expensive. Use tombstone + periodic compaction instead |

---

*Next: Module 4 — Retrieval Strategies (Dense, Sparse, Hybrid, Query Transformation)*
