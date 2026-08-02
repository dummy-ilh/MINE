# RAG Module 3 — Indexing & Vector Databases

---

## 3.1 The core problem: exact NN doesn't scale

Given a query vector, the "correct" answer is exact k-nearest-neighbor (kNN) search: compute distance to *every* vector in the index, sort, take top-k. This is O(N·d) per query — for N in the millions/billions, this is too slow for interactive latency (tens of ms).

**Approximate Nearest Neighbor (ANN)** search trades a small amount of recall (you might miss the true #7 nearest neighbor and get #9 instead) for orders-of-magnitude speedup. Every vector DB and index type in this module exists to approximate kNN faster than brute force, with different tradeoffs on speed/memory/recall.

---

## 3.2 ANN algorithm families

### HNSW (Hierarchical Navigable Small World)
- Builds a multi-layer graph: top layers are sparse "highways" connecting distant regions, bottom layer is dense with fine-grained connections
- Search starts at the top layer, greedily navigates toward the query vector, drops down a layer once no closer neighbor is found, repeats until the bottom layer gives the final candidate set
- **Tradeoffs**: excellent recall/speed tradeoff, but the graph must be held (mostly) in memory — memory footprint is a real constraint at billion-scale. Insertions are online/incremental (unlike IVF, which needs periodic retraining of cluster centroids), making HNSW a good fit for indexes with frequent updates.
- Tunable params to know: `M` (max connections per node — higher = better recall, more memory), `efConstruction` (build-time search depth), `efSearch` (query-time search depth — the main recall/speed knob at serving time)

### IVF (Inverted File Index)
- Partition the vector space into `nlist` clusters via k-means, each vector assigned to its nearest centroid
- At query time, only search within the `nprobe` closest clusters to the query (not the whole index) — this is the core speedup
- **Tradeoff**: `nprobe` is the recall/speed knob — probe more clusters for higher recall at higher cost. Requires a training step (running k-means) before use, and cluster boundaries can degrade as data distribution shifts post-training (a real production concern — mention this proactively).

### IVF-PQ (IVF + Product Quantization)
- Adds **Product Quantization**: instead of storing full-precision vectors, split each vector into sub-vectors and quantize each sub-vector to a small codebook index (e.g. compress a 768-dim float32 vector down to a few dozen bytes)
- Massively reduces memory footprint (this is the mechanism that makes billion-scale indexes fit in RAM), at the cost of reduced precision — distances computed on quantized vectors are approximate
- **Common pattern**: use IVF-PQ for the initial fast/cheap candidate retrieval, then re-rank the shortlist using full-precision vectors (a second, small-scale exact distance computation) to recover accuracy lost to quantization — this is itself a mini preview of the two-stage retrieve-then-rerank pattern in Module 5.

### ScaNN (Google)
- Uses **anisotropic vector quantization** — unlike standard PQ which minimizes *overall* quantization error uniformly, ScaNN's quantization is specifically optimized to minimize error in the *direction that matters for inner-product ranking* (errors along the ranking-relevant axis are penalized more than errors orthogonal to it)
- Consistently near the top of ANN benchmark leaderboards (recall vs queries-per-second) — good to know it exists and why it beats naive PQ, don't need deep implementation detail for most interviews.

---

## 3.3 FAISS — index types, when to use which

FAISS (Facebook AI Similarity Search) is a library, not a managed service — you self-host and choose your index type explicitly.

| Index type | Description | When to use |
|---|---|---|
| `IndexFlatL2` / `IndexFlatIP` | Brute-force exact search | Small corpora (≤~100K vectors), or as a ground-truth baseline to measure ANN recall loss against |
| `IndexIVFFlat` | IVF clustering + exact distance within probed clusters | Medium corpora, need better recall than PQ, memory less constrained |
| `IndexIVFPQ` | IVF + product quantization | Large corpora where memory is the binding constraint |
| `IndexHNSWFlat` | HNSW graph, full-precision vectors stored | Best recall/speed tradeoff when memory allows full-precision storage |

**Interview framing**: FAISS gives you the *building blocks* and you own the ops (persistence, sharding, filtering, updates) — contrast this directly with managed vector DBs (3.4) which wrap ANN indexing with production infrastructure. This distinction ("library vs system") is a common interview framing question.

---

## 3.4 Managed vector databases — feature comparison

| | Pinecone | Weaviate | Qdrant | Milvus | pgvector |
|---|---|---|---|---|---|
| Model | Fully managed, proprietary | Open-source + managed cloud option | Open-source + managed cloud option | Open-source + managed cloud option | Postgres extension |
| Metadata filtering | Yes, strong | Yes, strong (GraphQL-based) | Yes, strong | Yes | Yes (SQL `WHERE`) |
| Hybrid search (dense+sparse) | Built-in (sparse-dense vectors) | Built-in (BM25 + vector fusion) | Built-in | Supported | Requires manual combination with Postgres full-text search |
| Best fit | Teams wanting zero ops, fast to production | Teams wanting open-source + rich schema/graph features | Teams wanting open-source + strong filtering performance | Very large scale, high customization | Teams already on Postgres wanting to avoid a new system dependency |

**Interview-relevant point, not just a feature checklist**: the real decision driver is usually *operational* — do you already have Postgres in your stack and want to avoid adding a new database (pgvector), do you need proprietary-grade zero-ops scaling (Pinecone), or do you need full control/self-hosting for compliance reasons (Weaviate/Qdrant/Milvus self-hosted)? Interviewers are often testing whether you reach for "it depends on infra constraints" rather than reciting a single "best" vector DB.

---

## 3.5 Metadata filtering: pre-filter vs post-filter vs hybrid

Say a query needs both vector similarity *and* a metadata constraint (e.g. "find similar docs, but only from `department=legal`").

- **Post-filtering**: run ANN search first (get top-k by similarity), then filter the results by metadata. **Failure mode**: if very few of the top-k happen to match the metadata filter, you can end up with far fewer (or zero) results than requested — the ANN search wasn't aware of the filter, so it didn't prioritize matching candidates.
- **Pre-filtering**: apply the metadata filter first (reduce the candidate set to only matching vectors), then run ANN search only within that filtered subset. **Failure mode**: if the filtered subset is small, you lose the benefit of ANN index structures built over the *full* dataset (may fall back to brute-force over the filtered subset, or require a separate index per filter value — doesn't scale to high-cardinality filters).
- **Hybrid/integrated filtering** (what most modern vector DBs actually implement): filtering is pushed *into* the graph/cluster traversal itself — e.g. HNSW graph traversal skips non-matching nodes during the search rather than filtering before or after. This avoids both failure modes above but requires the index structure to support filter-aware traversal natively (not all do).

**Interview trap**: candidates often assume "just filter then search" (pre-filtering) is obviously correct — the interesting answer is naming the *high-cardinality filter* failure mode (e.g. filtering by `user_id` where each user has few docs) as the case where naive pre-filtering breaks down and hybrid/integrated filtering becomes necessary.

---

## 3.6 Index update strategies

- **Real-time upsert**: insert/update vectors into a live index as new documents arrive — supported natively by HNSW-based systems (graph insertion is incremental). Good for corpora with continuous, small-volume updates (e.g. a live support ticket system).
- **Batch rebuild**: periodically re-embed and rebuild the entire index from scratch — necessary when using IVF (cluster centroids trained on a snapshot of the data go stale as data distribution shifts) or when doing a full embedding-model migration (old and new embeddings are **not comparable** — you cannot mix vectors from two different embedding model versions in the same index, this must always be a full re-embed + rebuild).
- **Handling deletes**: most ANN structures don't support fast true deletion (removing a node from an HNSW graph is nontrivial). Common pattern: **soft delete via a tombstone flag** filtered out at query time, with periodic compaction/rebuild to actually reclaim space and remove tombstoned entries from the underlying structure.

**Interview trap to flag proactively**: "can you just add new documents to your existing index without downtime?" — the honest answer depends entirely on index type. HNSW: yes, easily. IVF-based: technically yes (inserted into existing clusters) but recall degrades over time as the data distribution drifts from the original cluster centroids, so periodic retraining/rebuild is still needed even though it's not strictly required for correctness.

---

## 3.7 Scaling: sharding, quantization, memory math

**Sharding**: split the index across multiple machines/nodes, either by:
- **Random/hash sharding**: distribute vectors arbitrarily — simple, but every query must fan out to *all* shards (scatter-gather), since there's no way to know which shard holds the relevant vectors
- **Semantic/cluster-based sharding**: assign vectors to shards based on a coarse clustering, so queries can be routed to only the most relevant shard(s) first — reduces fan-out but adds routing complexity and risk of imbalanced shard sizes if clusters are uneven

**Quantization for memory**: back-of-envelope math worth being able to do live in an interview —
- A 768-dim float32 vector = 768 × 4 bytes = 3072 bytes ≈ 3KB per vector, *before* any graph/index overhead
- At 100M vectors: 3KB × 100M ≈ 300GB just for raw vectors — often exceeds a single machine's RAM, motivating either PQ compression (can shrink this by 10-30x) or sharding across machines (or both)
- Product Quantization example: compressing to 8-bit codes across 96 sub-vectors → 96 bytes/vector instead of 3072 bytes — roughly a 32x reduction, at the cost of approximate (not exact) distance computation

**Interview signal**: being able to casually do this napkin math (vectors × dims × bytes-per-float) when asked "how would you scale this to N documents" is a strong differentiator — most candidates talk about scaling only qualitatively.

---

## Interview Q&A drill

**Q: Your team wants to add real-time document updates to a RAG system currently using FAISS IndexIVFPQ. What's the issue and what would you recommend?**
A: IVF-based indexes are trained on a snapshot of the data (k-means cluster centroids fit once); new vectors can technically be inserted into existing clusters, but as more new data arrives, the original centroids stop representing the actual data distribution well, degrading recall over time without warning. For a system needing frequent live updates, I'd recommend HNSW-based indexing instead (natively incremental, no periodic retraining required for correctness) or, if staying on IVF/PQ for its memory efficiency at scale, scheduling periodic full index rebuilds and monitoring recall on a held-out eval set to catch degradation before it becomes user-visible.

**Q: Walk me through pre-filtering vs post-filtering and when each breaks.**
A: Post-filtering runs ANN search first, then discards results that fail the metadata filter — breaks when the filter is highly selective, since you might filter away most or all of the top-k, ending up with too few results despite plenty of matching documents existing elsewhere in the index. Pre-filtering restricts the candidate set to matching vectors first, then searches — breaks when the filter has high cardinality (many distinct filter values, each with few matching vectors), since you either lose ANN index benefits over such tiny filtered subsets or need a separate index per filter value, which doesn't scale. Most production vector DBs solve this with filter-aware traversal integrated into the ANN algorithm itself, avoiding both failure modes.

**Q: You need to migrate from embedding model A to embedding model B. Walk through what breaks if you do this naively.**
A: Vectors from two different embedding models live in *different, incompatible geometric spaces* — there's no shared coordinate system between them, so cosine similarity between an old-model query vector and a new-model document vector (or vice versa) is meaningless, not just slightly degraded. A naive incremental migration (embed only new documents with model B, leave old documents on model A) silently produces garbage retrieval for any query that should match an old document. The correct approach is a full re-embed of the entire corpus with model B and a full index rebuild, typically done as a blue-green swap (build the new index fully offline, validate recall against an eval set, then cut over) rather than an in-place migration.

**Q: When would pgvector be the right choice over a dedicated vector DB like Pinecone, even though it's less specialized?**
A: When the team already runs Postgres for the application's primary data and wants to avoid introducing a new database dependency, operational surface, and data-sync problem (keeping metadata in Postgres and vectors in a separate vector DB in sync is itself an engineering cost). pgvector is the right tradeoff at moderate scale where you don't yet need Pinecone-grade horizontal scaling or advanced ANN tuning, and where transactional consistency between metadata and vectors (native to a single Postgres instance) matters more than raw ANN performance.

---

**Next up: Module 4 — Retrieval strategies (dense, sparse, hybrid, query transformation).** Say the word when ready.
