# RAG Interview Prep — Day 4
## Vector Databases & Indexing (Extended Deep Dive)

---

## 🚀 Quick Summary

A vector database's entire reason for existing is to answer "which of my millions (or billions) of vectors are closest to this query vector?" fast enough for a production system — and doing that fast requires trading a small amount of accuracy for a massive amount of speed via **Approximate Nearest Neighbor (ANN)** algorithms. Today goes deep on the full indexing landscape (not just HNSW and IVF, but also Product Quantization, LSH, and IVF-PQ hybrids), the actual recall/latency/memory math behind each, how real vector databases differ as products, and how to reason about scaling a vector index the way an Apple systems interview would expect — sharding, replication, filtering-at-scale, and back-of-envelope capacity planning.

**Think of it like organizing a warehouse of a billion boxes.** Walking every aisle to find the exact box you want (brute-force exact search) works perfectly but takes forever past a certain warehouse size. Every ANN algorithm is a different *filing system* — some build a hierarchy of hub-and-spoke shortcuts (HNSW), some pre-sort boxes into labeled zones and only search the nearest zones (IVF), some shrink every box down to a compressed summary so more fit in memory at once (Product Quantization) — and picking the right filing system depends on how many boxes you have, how often new boxes arrive, how much shelf space (memory) you have, and how fast you need to find things.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Exact NN (brute-force / Flat index)** | Compare the query against every single vector — perfectly accurate, doesn't scale |
| **ANN (Approximate Nearest Neighbor)** | Algorithms that trade a small, tunable amount of accuracy for large speed/memory gains at scale |
| **HNSW** | Hierarchical Navigable Small World — multi-layer graph-based ANN index |
| **IVF** | Inverted File Index — clustering-based ANN index (search only nearby clusters) |
| **Product Quantization (PQ)** | Compresses each vector into a compact code by quantizing sub-vector segments independently, drastically reducing memory |
| **IVF-PQ** | A hybrid combining IVF's cluster-based candidate narrowing with PQ's memory-efficient compressed storage |
| **LSH (Locality-Sensitive Hashing)** | Hashes similar vectors into the same "buckets" with high probability, so search only checks the query's bucket |
| **Recall@k (in the ANN context)** | Here specifically means: what fraction of the *true* top-k nearest neighbors did the ANN search actually return, vs. exact search |
| **QPS** | Queries per second — the throughput a vector index/database can sustain |
| **Sharding** | Splitting an index across multiple machines, typically by data volume |
| **Replication** | Duplicating an index (or shard) across multiple machines for read throughput and fault tolerance |

---

# PHASE 1 — Intuition & The Full Indexing Landscape

## The full menu, not just HNSW vs. IVF

```
                         HOW SHOULD I SEARCH MY VECTORS?
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                              ▼
  EXACT SEARCH                 GRAPH-BASED ANN              PARTITION-BASED ANN
  (Flat / brute-force)              (HNSW)                  (IVF, LSH)
        │                            │                              │
  perfect accuracy,           hub-and-spoke graph,          cluster/hash first,
  O(N) per query,             coarse-to-fine search,        search only relevant
  fine for <100K vectors      great for frequent updates    partition(s)
                                                                     │
                                                              ┌──────┴──────┐
                                                              ▼             ▼
                                                            IVF           LSH
                                                      (k-means         (hash
                                                       clusters)       buckets)

                              COMPRESSION LAYER (orthogonal — can combine with any of the above)
                                     │
                              PRODUCT QUANTIZATION (PQ)
                              shrinks memory footprint, often paired
                              with IVF as "IVF-PQ" in production systems
```

**Key framing for the interview:** these aren't all competing for the same slot. HNSW and IVF are both answers to "how do I avoid comparing against every vector," while Product Quantization is an orthogonal answer to "how do I avoid storing every vector at full precision." Production systems very often **combine** them — e.g., FAISS's popular `IVF-PQ` index type uses IVF to narrow the search to a few clusters, then PQ-compressed vectors within those clusters for a fast, memory-efficient distance comparison.

---

# PHASE 2 — Math & Mechanics, Algorithm by Algorithm

## 1. Flat / Brute-Force (Exact Search)

**Mechanism:** Compare the query vector against every single indexed vector, compute exact similarity/distance for each, sort, return top-k.

**Complexity:** `O(N × d)` per query, where `N` = number of vectors, `d` = dimensionality — this is linear in corpus size, which is the entire problem.

**Worked example — when brute-force stops being viable:** Say a single similarity comparison (dot product over `d=768` dimensions) takes roughly 1 microsecond on typical hardware.
```
N = 10,000 vectors:      10,000 × 1μs = 10 ms per query   → totally fine
N = 1,000,000 vectors:   1,000,000 × 1μs = 1,000 ms = 1s   → too slow for interactive use
N = 100,000,000 vectors: 100,000,000 × 1μs = 100s          → completely unusable
```
This is the concrete justification for why ANN exists at all — the linear scaling is fine at small scale and catastrophic past roughly a million vectors for latency-sensitive applications.

**When to actually use it:** Small datasets (under ~100K–1M vectors depending on latency budget), or as a **ground-truth baseline** to measure ANN recall against during evaluation (you can't know your HNSW index is achieving 95% recall without something to compare it to).

---

## 2. HNSW (Hierarchical Navigable Small World) — Deeper Mechanics

**Structure:** A multi-layer graph. The top layer has few nodes with long-range connections (like highways between distant cities); each layer down has progressively more nodes with shorter-range connections (like local streets), down to the bottom layer, which contains *every* vector.

**Search process:**
1. Start at a fixed entry point in the top (sparsest) layer.
2. Greedily walk toward the query by hopping to whichever neighbor is closer, until no neighbor improves — this quickly gets you to the right general neighborhood using very few long-range hops.
3. Drop down one layer, repeat the greedy walk (now with finer-grained connections), continuing down to the bottom layer.
4. At the bottom layer, do a final local search among the closest candidates found, returning the top-k.

**Key hyperparameters (know these cold):**

| Hyperparameter | What it controls | Effect of increasing it |
|---|---|---|
| `M` | Max connections per node in the graph | Higher recall/accuracy, more memory, slower index build |
| `ef_construction` | Search effort used *while building* the graph | Better-quality graph (more accurate neighbor connections), much slower index build |
| `ef_search` | Search effort used *at query time* | Higher recall, higher query latency — this is the main tunable knob *after* the index already exists, no rebuild needed |

**Worked numerical example — recall/latency trade-off via ef_search:**
This is illustrative (real numbers vary by dataset/implementation), but the *shape* of the trade-off is what interviewers want you to reason through:
```
ef_search = 10:   recall@10 ≈ 0.85,  latency ≈ 1.2 ms/query
ef_search = 50:   recall@10 ≈ 0.95,  latency ≈ 3.5 ms/query
ef_search = 200:  recall@10 ≈ 0.99,  latency ≈ 9.0 ms/query
```
Notice the **diminishing returns**: going from ef_search 10→50 buys +10 points of recall for +2.3ms; going from 50→200 buys only +4 points of recall for +5.5ms. This is the classic ANN trade-off curve — early increases in search effort are cheap wins, later increases cost much more for much less.

**Memory footprint:** HNSW's graph edges are real memory overhead on top of the raw vectors themselves — roughly, memory scales with `N × M × (bytes per edge reference)`, in addition to `N × d × 4 bytes` for the raw float32 vectors. This is *why* HNSW is generally more memory-hungry than IVF for the same corpus — you're storing a graph structure, not just the vectors.

**Update behavior:** New vectors can be inserted into an existing HNSW graph incrementally (find its place via the same greedy-search mechanism used for queries, then wire it into the graph), without a full rebuild — this is HNSW's single biggest practical advantage for RAG corpora that change continuously.

---

## 3. IVF (Inverted File Index) — Deeper Mechanics

**Structure:** Run k-means (or similar clustering) over the corpus to produce `nlist` cluster centroids. Every vector is assigned to its nearest centroid. At query time, the query is compared against the `nlist` centroids first (cheap — there are far fewer centroids than vectors), then a full search is only performed **within** the `nprobe` closest clusters.

**Key hyperparameters:**

| Hyperparameter | What it controls | Effect of increasing it |
|---|---|---|
| `nlist` | Number of clusters | More clusters = finer partitioning, faster per-cluster search, but risk of the true nearest neighbor sitting in a cluster that wasn't searched if clustering boundaries are imperfect |
| `nprobe` | Number of clusters searched per query | Higher = better recall, slower query (directly analogous to HNSW's `ef_search`) |

**Worked numerical example — why nprobe matters:**
Say a corpus of 10 million vectors is split into `nlist = 1000` clusters (~10,000 vectors/cluster on average).
```
nprobe = 1:    search ~10,000 vectors (1 cluster)   → very fast, but if the true nearest
                                                        neighbor is in a different cluster
                                                        near the boundary, you miss it → lower recall
nprobe = 10:   search ~100,000 vectors (10 clusters) → 10x more comparisons, catches boundary
                                                        cases in nearby clusters, higher recall
nprobe = 100:  search ~1,000,000 vectors (100 clusters) → 100x comparisons, approaching
                                                          brute-force-level recall, much slower
```
This is the same speed/recall dial as `ef_search` in HNSW, just implemented via a different mechanism (searching more partitions vs. deeper graph traversal).

**Why boundary cases cause recall loss:** If a vector sits geometrically near the edge between two clusters, it's assigned to only one of them (whichever centroid it's nearest to) — but a query that lands *just* on the other side of that boundary won't find it unless `nprobe` is large enough to also search the neighboring cluster. This is the core accuracy limitation of pure IVF, and it's why `nprobe > 1` is almost always used in practice.

**Update behavior:** New vectors can be assigned to an existing cluster (fast — just find the nearest centroid), but the cluster *centroids themselves* were computed from the original data distribution. As enough new data arrives that the distribution shifts, centroids become stale and recall degrades gradually — eventually requiring a re-clustering (re-running k-means on the full updated corpus), which is a heavier operation than HNSW's incremental insertion.

---

## 4. Product Quantization (PQ) — Compression, Not Just Search Speed

**The problem PQ solves:** Storing raw float32 vectors at scale is memory-expensive (Day 2's quantization discussion touched on this). PQ is a more aggressive, structured form of compression than simple int8 quantization.

**Mechanism, step by step:**
1. Split each `d`-dimensional vector into `m` smaller sub-vectors (e.g., a 768-dim vector split into 8 sub-vectors of 96 dimensions each).
2. For each of the `m` sub-vector "slots," run k-means clustering *independently* across all vectors in the corpus, producing a small codebook of (e.g.) 256 representative centroids for that slot.
3. Each vector is now represented not by its raw sub-vector values, but by **which centroid ID (0–255) it's closest to, in each of the 8 slots** — i.e., 8 small integers instead of 768 floats.

**Worked numerical example — the compression math:**
```
Raw vector: 768 dimensions × 4 bytes (float32) = 3072 bytes per vector

PQ-compressed: split into m=8 sub-vectors, each sub-vector represented
               by a single centroid ID from a 256-entry codebook
               → 256 possible values fits in 1 byte (log2(256) = 8 bits)
               → 8 sub-vectors × 1 byte = 8 bytes per vector

Compression ratio = 3072 / 8 = 384x smaller
```
That's a **384x memory reduction** — dramatically more aggressive than simple int8 quantization's ~4x, because PQ exploits *structure* in the data (via clustering) rather than just truncating numerical precision uniformly.

**The accuracy cost:** Distance calculations between a query and a PQ-compressed vector become *approximate* — you're computing distance based on which centroids the vector's sub-vectors are closest to, not the vector's actual original values, so there's an inherent quantization error. This is why PQ is almost always combined with a coarse-search step (like IVF) that narrows candidates first, and often a final re-ranking step using full-precision vectors on the small shortlist to recover accuracy — the same "quantize for speed, full-precision for the final shortlist" pattern from Day 2.

**Why it matters in practice:** PQ (typically as IVF-PQ) is what makes billion-scale vector search feasible on a single machine's memory — without it, a billion 768-dim float32 vectors would need ~2.86 TB of RAM (`1e9 × 768 × 4 bytes`), which is impractical; with PQ compression, that drops to a few GB, fitting comfortably in memory.

---

## 5. LSH (Locality-Sensitive Hashing) — Brief but Interview-Relevant

**Mechanism:** Use hash functions specifically designed so that **similar vectors are likely to land in the same hash bucket**, while dissimilar vectors are likely to land in different buckets (the opposite goal of a normal cryptographic hash function, which is designed to scatter similar inputs unpredictably). At query time, only compare against vectors in the query's bucket(s).

**Where it fits vs. HNSW/IVF:** LSH was historically important and is still used in some systems, but has generally been outperformed by HNSW and IVF-PQ on recall/speed trade-offs for modern high-dimensional embedding search — worth knowing it exists and roughly how it works (for breadth), but HNSW and IVF(-PQ) are what you'll actually encounter in nearly every modern production vector database.

---

## Algorithm comparison table (the master summary — memorize this)

| | Flat (exact) | HNSW | IVF | IVF-PQ |
|---|---|---|---|---|
| **Accuracy** | Perfect | High (tunable via `ef_search`) | High (tunable via `nprobe`) | Slightly lower (compression adds error) |
| **Query speed at scale** | Unusable past ~1M vectors | Fast, consistent | Fast, depends on `nprobe` | Fastest at very large scale |
| **Memory** | Highest (full precision, no structure) | High (vectors + graph edges) | Moderate (vectors + centroid list) | Lowest (compressed codes) |
| **Build time** | None (nothing to build) | Slower | Faster | Moderate (clustering + codebook training) |
| **Handles incremental updates** | Trivially (just append) | Well | Poorly (centroid drift over time) | Poorly (same IVF limitation) |
| **Best for** | Small datasets, ground-truth baseline | Frequently-updated, latency-sensitive production RAG | Large static datasets, memory-conscious | Billion-scale datasets where memory is the binding constraint |

---

## Real Vector Database Products — How the Algorithms Show Up in Practice

| Product | Type | Notable characteristics |
|---|---|---|
| **FAISS** | Library (not a managed DB) | Meta's library implementing Flat, IVF, HNSW, PQ, and combinations (IVF-PQ) — the algorithmic reference point most other tools build on or benchmark against |
| **Pinecone** | Managed cloud vector DB | Fully managed, abstracts index-type choice significantly, strong at metadata filtering and multi-tenancy at scale |
| **Weaviate** | Open-source, self-hostable or managed | HNSW-based, strong hybrid (sparse+dense) search support, GraphQL-style query interface |
| **Milvus** | Open-source, distributed-first | Supports multiple index types (HNSW, IVF, IVF-PQ), built with horizontal scaling/sharding as a first-class concern |
| **Qdrant** | Open-source, Rust-based | HNSW-based, known for strong filtering performance (filtered HNSW variant) |
| **pgvector** | Postgres extension | Adds vector search *into* an existing relational database — attractive when you want vector search alongside normal SQL/joins in one system rather than a separate specialized store, at some ceiling on scale/performance vs. purpose-built vector DBs |

> **Why This Matters callout:** If asked to choose a vector database, the strong interview move isn't naming a favorite product — it's naming the **actual requirements that should drive the choice**: expected scale (vectors, QPS), update frequency, filtering/multi-tenancy needs, whether you want a managed service vs. self-hosted control, and whether you already have infrastructure (e.g., "we're already heavily invested in Postgres" is a legitimate real reason to reach for pgvector over standing up a new specialized system, even if it's not the highest-performance option at extreme scale).

---

## Scaling a Vector Index: Sharding, Replication, and Capacity Planning

**Sharding** — splitting the index across multiple machines, typically because a single machine can't hold the full index in memory or can't sustain the required QPS alone.
- Common strategy: shard by a natural partition key (e.g., by tenant, by document category, or just by hash of vector ID for even distribution).
- Trade-off: a query may need to fan out to multiple shards and merge results (scatter-gather), adding coordination overhead and tail-latency risk (the query is only as fast as the *slowest* shard it touches).

**Replication** — duplicating a shard (or the whole index) across multiple machines.
- Purpose: read throughput (more replicas = more QPS capacity) and fault tolerance (a replica can serve reads if one node goes down).
- Consistency trade-off: updates need to propagate to all replicas — most vector databases favor **eventual consistency** for this (a newly inserted vector might not be immediately searchable on every replica), which is usually an acceptable trade-off for RAG use cases where sub-second staleness on newly added documents is rarely business-critical, in exchange for much better write/read scalability than strict consistency would allow.

**Worked back-of-envelope capacity planning example (a realistic system-design-style question):**

*Scenario:* You need to serve a RAG index of **200 million vectors**, 768 dimensions, using HNSW, targeting **2,000 QPS** at under 50ms p99 latency.

```
Step 1 — Raw vector memory:
  200,000,000 × 768 × 4 bytes = 614,400,000,000 bytes ≈ 572 GB (just the raw vectors)

Step 2 — HNSW graph overhead (rough rule of thumb: ~1.5-2x raw vector size
          for graph edges at typical M settings):
  572 GB × ~1.7 ≈ 972 GB total memory needed

Step 3 — This clearly exceeds a single machine's practical RAM budget
          (e.g., a large single instance might offer ~512GB–1TB) →
          sharding is required, not optional, at this scale.

Step 4 — Sharding decision: split into, say, 4 shards of 50M vectors
          each (~243 GB per shard, comfortably fits a single large-memory
          instance with headroom).

Step 5 — Replication for throughput: if a single shard replica sustains
          ~600 QPS at the target latency, and you need 2000 QPS system-wide,
          you'd want ~4 replicas per shard (2000/600 ≈ 3.3, round up) for
          headroom and fault tolerance.

Total nodes ≈ 4 shards × 4 replicas = 16 nodes
```

**Why this matters in practice:** This is exactly the style of estimation an Apple systems-design interview segment wants to see — not a memorized "right answer," but the ability to move from a data-scale + throughput requirement to a rough memory footprint, recognize when a single machine is insufficient, and reason through sharding and replication counts with actual arithmetic, even with approximate/rule-of-thumb constants. Getting the exact multiplier right matters far less than demonstrating the reasoning chain.

---

## Filtering at Scale — Revisited with More Depth

**Pre-filtering vs. post-filtering (recap from Module 1, with more mechanism detail):**
- **Post-filtering** is simple but risky: run the ANN search first, ignoring metadata, then discard non-matching results afterward. If the filter is highly selective (e.g., only 0.1% of vectors match), you might retrieve top-k and have almost none survive the filter — a very common production bug.
- **Pre-filtering (naive)** restricts the candidate set by metadata *before* running ANN search — but naively, this can force something close to brute-force search *within* the filtered subset if the ANN index structure (e.g., an HNSW graph built over the *whole* corpus) doesn't have an efficient way to traverse only the filtered subset — the graph's shortcuts were built assuming the whole corpus is eligible.
- **Filtered ANN indexes (the real solution at scale):** Modern vector databases increasingly implement **filter-aware** ANN search — e.g., HNSW variants that can prune the graph traversal using the filter condition mid-search, or maintaining separate sub-indexes per common filter value (e.g., one HNSW graph per tenant, if multi-tenancy filtering is the dominant filter pattern) — trading index-build complexity and potentially higher memory (multiple indexes) for much better filtered-query performance.

> **Gotcha:** Don't describe metadata filtering as solved by simply "adding a WHERE clause" — at scale, *how* the filter interacts with the ANN index structure is a real, actively-evolving area of vector database engineering, and this nuance is exactly the kind of depth an Apple MLE interview is listening for.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** Why does brute-force exact nearest-neighbor search stop being viable as corpus size grows, and roughly where's the tipping point?

<details>
<summary>Show answer</summary>

Brute-force search is `O(N × d)` per query — linear in the number of vectors — so query latency grows directly with corpus size. At small scale (tens of thousands of vectors) this is fine, often single-digit milliseconds. Past roughly hundreds of thousands to a million vectors, latency crosses from "interactive" into "too slow for production," which is the practical tipping point where ANN algorithms (HNSW, IVF, etc.) become necessary instead of optional.
</details>

---

**Q2 (Easy — calculation).** A corpus of 5 million vectors is split into IVF with `nlist = 500` clusters. At `nprobe = 5`, roughly how many vectors get compared per query, and what's the trade-off of increasing nprobe to 50?

<details>
<summary>Show answer</summary>

```
vectors per cluster ≈ 5,000,000 / 500 = 10,000
nprobe = 5:  ~5 × 10,000 = 50,000 vectors compared
nprobe = 50: ~50 × 10,000 = 500,000 vectors compared (10x more)
```
Increasing nprobe from 5 to 50 improves recall (catches more boundary cases where the true nearest neighbor sits in a nearby-but-not-closest cluster) but costs roughly 10x more comparisons per query, directly increasing latency — the standard IVF speed/recall trade-off.
</details>

---

**Q3 (Medium — conceptual).** Explain what Product Quantization does differently from simple int8 quantization, and why it achieves a much larger compression ratio.

<details>
<summary>Show answer</summary>

Simple int8 quantization reduces each individual dimension's numerical precision uniformly (float32 → int8, a fixed ~4x reduction regardless of the data). Product Quantization instead splits each vector into several sub-vectors, and for each sub-vector "slot," learns a small codebook of representative centroids via clustering across the whole corpus — then represents each vector not by its raw values but by which centroid ID it's closest to in each slot. Because centroid IDs are small integers that exploit actual structure/redundancy in the data (via clustering) rather than just truncating precision, PQ can achieve much larger compression ratios (often 100x+) than uniform quantization, at the cost of approximate (not exact) distance calculations.
</details>

---

**Q4 (Medium — conceptual).** Why does HNSW generally handle frequent incremental updates better than IVF?

<details>
<summary>Show answer</summary>

HNSW inserts a new vector by using the same greedy-search mechanism used for queries to find where it belongs in the existing graph, then wiring it into the relevant layers — an incremental, local operation that doesn't require touching the rest of the graph. IVF's cluster centroids, by contrast, were computed from the original data distribution via clustering (e.g., k-means); as new data arrives and the distribution shifts, those centroids gradually become stale (misrepresenting where the data actually is), degrading recall over time, and eventually require a full re-clustering pass over the corpus to fix — a much heavier, more disruptive operation than HNSW's local insertion.
</details>

---

**Q5 (Medium — system design).** You're serving a vector index that needs sub-100ms p99 latency and must support metadata filtering by `tenant_id` for strict data isolation across customers. What are the trade-offs between a single shared index with a tenant_id filter vs. one index per tenant?

<details>
<summary>Show answer</summary>

**Shared index + filter:** Cheaper and simpler to operate (one index to maintain, scales storage more efficiently by pooling all tenants together), but introduces real risk — a filtering bug is a data-leak incident across tenants, and if the ANN index isn't filter-aware, a highly selective tenant filter (a tenant with few documents in a huge shared index) can suffer poor recall or effectively fall back toward brute-force-like search within the filtered subset, hurting both latency and accuracy for smaller tenants. **Per-tenant index:** Much stronger isolation by construction (no filtering logic to get wrong), and predictable, consistent performance per tenant regardless of other tenants' data volume — but infrastructure and operational overhead scale roughly linearly with tenant count, which becomes expensive and operationally heavy at large tenant counts. The right choice depends on tenant count and isolation requirements: for a small number of large, high-value tenants (e.g., enterprise customers with strict compliance needs), per-tenant indexes are often worth the overhead; for a large number of small tenants, a shared filtered index (ideally on a filter-aware ANN implementation) is usually more practical.
</details>

---

**Q6 (Hard — calculation).** You need to index 1 billion vectors at 768 dimensions. Compute the memory required for (a) raw float32 storage, and (b) PQ-compressed storage using m=8 sub-vectors with 256-entry codebooks per slot. What does this tell you about when PQ becomes necessary rather than optional?

<details>
<summary>Show answer</summary>

```
(a) Raw float32: 1,000,000,000 × 768 × 4 bytes = 3,072,000,000,000 bytes ≈ 2.86 TB

(b) PQ-compressed: 8 sub-vectors × 1 byte each (256 codebook entries → 1 byte per slot)
    = 8 bytes/vector
    1,000,000,000 × 8 bytes = 8,000,000,000 bytes ≈ 7.45 GB
```
Raw float32 storage for 1 billion vectors (2.86 TB) exceeds what's practical to hold in memory on typical single machines or even modest multi-node setups without heavy sharding. PQ compression brings this down to under 8 GB — comfortably fitting in memory on a single machine. This demonstrates why PQ (typically combined with IVF for the search-narrowing step, i.e. IVF-PQ) isn't just an optimization but becomes effectively **necessary**, not optional, once corpus size reaches the hundreds-of-millions-to-billions scale, if you want to avoid a very large and expensive sharded cluster just to hold raw vectors in memory.
</details>

---

**Q7 (Hard — system design synthesis).** Walk through, with rough numbers, how you'd estimate the number of machines needed to serve a 300-million-vector, 768-dimension HNSW index at 3,000 QPS with sub-50ms p99 latency.

<details>
<summary>Show answer</summary>

Step 1 — raw vector memory: `300,000,000 × 768 × 4 bytes ≈ 858 GB`. Step 2 — apply an HNSW graph-overhead multiplier (rule of thumb ~1.5-2x raw vectors for edges): `858 GB × 1.7 ≈ 1.46 TB` total memory needed. Step 3 — this exceeds a single reasonable machine's memory, so sharding is required; splitting into, say, 6 shards of 50M vectors each gives ~243 GB per shard, fitting a large-memory instance comfortably. Step 4 — for throughput, estimate single-replica QPS capacity at the target latency (say a shard replica sustains ~500 QPS at sub-50ms), then compute replicas needed: `3000 QPS / 500 QPS per replica ≈ 6 replicas per shard` for headroom and fault tolerance. Step 5 — total nodes ≈ `6 shards × 6 replicas = 36 nodes`. The exact constants (overhead multiplier, per-replica QPS) would need real benchmarking to nail down precisely, but the reasoning chain — raw memory → overhead-adjusted memory → sharding decision → per-shard throughput → replication count → total node estimate — is the actual skill being tested.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Treating HNSW and IVF as if they're competing with Product Quantization — PQ is an orthogonal compression layer, commonly combined with IVF (or even HNSW variants) rather than an alternative to them.
- ❌ Assuming index choice is purely about "which algorithm is best" instead of reasoning from actual requirements: scale, update frequency, latency budget, memory budget, filtering needs.
- ❌ Describing metadata filtering as a free "WHERE clause" without acknowledging how it interacts with (and can degrade) ANN index performance at scale.
- ❌ Forgetting that IVF's biggest weakness isn't search speed — it's that cluster centroids drift stale as new data arrives, unlike HNSW's graceful incremental updates.
- ❌ Not knowing any concrete numbers when asked a capacity-planning question — even rough, clearly-labeled estimates ("rule of thumb ~1.5-2x overhead") demonstrate far more competence than refusing to estimate.
- ❌ Assuming replication only helps fault tolerance — it's equally (often primarily) about read throughput/QPS scaling.

---

# 📌 Cheat Sheet (Day 4)

**Algorithm landscape:** Flat (exact, `O(N×d)`, unusable past ~1M vectors) → HNSW (graph, great updates, more memory, `M`/`ef_construction`/`ef_search` knobs) → IVF (clusters, faster build, less memory, poor at updates, `nlist`/`nprobe` knobs) → PQ (orthogonal compression, ~100x+ reduction via sub-vector codebooks, approximate distances) → IVF-PQ (the common production combo at billion-scale).

**Recall/latency dial:** `ef_search` (HNSW) and `nprobe` (IVF) both trade latency for recall with diminishing returns — early increases are cheap, later increases cost much more for less gain.

**Products:** FAISS (library/reference), Pinecone (managed, filtering/multi-tenancy strength), Weaviate (HNSW + hybrid search), Milvus (distributed-first), Qdrant (filtered-HNSW strength), pgvector (Postgres-native, good when already SQL-invested).

**Scaling:** Shard when memory or QPS exceeds one machine; replicate for throughput + fault tolerance (usually eventual consistency across replicas). Capacity planning = raw vector memory → index-overhead-adjusted memory → sharding decision → per-replica QPS → replication count.

**Filtering:** pre-filter > post-filter for selective filters, but naive pre-filtering can degrade toward brute-force within the filtered subset unless the ANN index itself is filter-aware (filtered-HNSW, per-tenant sub-indexes, etc.).

---

*End of Day 4. Next up — Day 5: Metadata Filtering, Hybrid Storage & Multi-Tenancy.*
