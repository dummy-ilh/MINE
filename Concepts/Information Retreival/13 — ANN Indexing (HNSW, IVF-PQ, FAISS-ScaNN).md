# Chapter 13 — ANN Indexing (HNSW, IVF-PQ, FAISS/ScaNN)

## What is it?

Dense retrieval (Chapter 4) gives you a query vector and needs the top-k nearest document vectors out of a corpus. At small scale, this is a linear scan: compute a similarity score against every vector, sort, done. At Apple/Google/Meta scale — hundreds of millions to billions of vectors — a linear scan is computationally impossible per-query at production latency.

**Approximate Nearest Neighbor (ANN) indexing** is the family of data structures and algorithms that let you find the (approximately) closest vectors in sub-linear time, trading a small amount of recall for orders-of-magnitude speedup. This is the single most common "how would you actually ship this at scale" follow-up to any dense retrieval interview answer.

---

## The intuition

Three observations that motivate everything in this chapter:

**Observation 1 — exact search doesn't scale.** Computing cosine similarity between 1 query and 1 billion documents, each a 768-dim vector, is ~1.5 trillion floating point multiplications per query. At any real QPS this is not feasible on CPU, and even on specialized hardware it's wasteful.

**Observation 2 — you don't need the exact top-k, you need a *very good approximation* of it, fast.** Users can't tell the difference between the true #7 and #9 most relevant result. So we can trade a small, controlled amount of recall for a massive latency win.

**Observation 3 — there are two fundamentally different ways to prune the search space:**
- **Graph-based (HNSW):** build a navigable graph over the vectors so you can "walk" toward the query's neighborhood instead of scanning everything.
- **Partition + compress (IVF-PQ):** cluster the vectors into buckets so you only scan a few relevant buckets, and compress each vector so more of it fits in memory/cache.

FAISS and ScaNN are libraries that implement both families (and hybrids of them); they are not themselves separate algorithms.

---

## HNSW (Hierarchical Navigable Small World)

### The equations / mechanics

HNSW builds a **multi-layer graph**. Think of it like a skip list generalized to high dimensions:

```
Layer 2 (sparse):   A ────────────── D
Layer 1 (medium):   A ──── B ──── D ──── F
Layer 0 (all pts):  A─B─C─D─E─F─G─H─I─J─K
```

- Every vector lives in layer 0 (the base layer, containing *all* points).
- A vector is *also* inserted into layer 1, 2, 3... with exponentially decreasing probability:

```
P(vector assigned to layer ≥ l) = exp(-l / mL)
```

where `mL` is a normalization constant (typically `1/ln(M)`, with `M` = max connections per node).

- Each node connects to its `M` nearest neighbors already in the graph at that layer (found via greedy search from the entry point).

**Search procedure:**
1. Start at a fixed entry point in the top (sparsest) layer.
2. Greedily walk to the neighbor closest to the query at this layer, until no neighbor is closer (a local minimum for this layer).
3. Drop down one layer, using the current position as the new starting point, and repeat.
4. At layer 0, do a wider beam search using a parameter `ef_search` (how many candidates to keep in the priority queue) to get the final top-k.

**Key parameters:**
- `M` — max neighbors per node per layer. Higher M = better recall, more memory, slower build.
- `ef_construction` — beam width during index build. Higher = better graph quality, slower build.
- `ef_search` — beam width during query. Higher = better recall, higher query latency. **This is the main recall/latency knob at serving time.**

**Complexity:** Build is `O(N log N)`. Search is `O(log N)` expected — this is the entire point: you go from scanning N vectors to touching roughly `log N` of them.

---

## IVF-PQ (Inverted File + Product Quantization)

### The equations / mechanics

**Step 1 — IVF (coarse partitioning).** Run k-means over the corpus to get `nlist` centroids (e.g. 4096 or 16384 clusters). Every document vector is assigned to its nearest centroid — this is exactly like building an inverted index, except the "postings list" for each centroid contains vectors instead of doc-ids for a term.

At query time, compare the query only against the `nprobe` closest centroids (not all `nlist` of them), then only scan the documents inside those `nprobe` buckets.

```
scanned_fraction ≈ nprobe / nlist
```

This alone gives a huge speedup: with `nlist = 4096` and `nprobe = 8`, you scan roughly `8/4096 ≈ 0.2%` of the corpus.

**Step 2 — PQ (compression).** Even 0.2% of a billion vectors is still 2 million 768-dim float32 vectors = ~6GB just for that scan, per query, repeated at high QPS. So each vector is compressed:

- Split the `D`-dimensional vector into `m` sub-vectors (e.g. `D=768`, `m=96` → 8 dims per sub-vector).
- Run k-means separately on each of the `m` sub-vector spaces to get `k` centroids (typically `k=256`, so each sub-vector index fits in 1 byte).
- Store each original vector as `m` centroid-ID bytes instead of `D` floats.

```
compression_ratio = (D × 4 bytes) / (m × 1 byte) = (768 × 4) / 96 = 32×
```

**Distance computation** at query time uses **Asymmetric Distance Computation (ADC):** keep the *query* uncompressed, precompute the distance from each query sub-vector to all 256 centroids in each of the `m` sub-spaces (a small lookup table), then approximate the full distance as a sum of table look-ups:

```
d(q, x)² ≈ Σ_{i=1}^{m} ||q_i − centroid(x_i)||²
```

This turns a 768-dim float distance computation into `m` table lookups + additions — dramatically cheaper.

---

## Worked numeric example

```
Corpus: N = 1,000,000 vectors, D = 768 dimensions, float32

Brute-force scan cost:
  1,000,000 × 768 multiply-adds ≈ 768M FLOPs per query

IVF setup:
  nlist  = 1,000  (centroids)
  nprobe = 10     (buckets scanned per query)

Average bucket size = N / nlist = 1,000,000 / 1,000 = 1,000 vectors

Vectors scanned per query = nprobe × avg_bucket_size = 10 × 1,000 = 10,000

Speedup from IVF alone = 1,000,000 / 10,000 = 100×
```

Now add PQ compression on top:

```
m = 96 sub-vectors, k = 256 centroids per sub-space, 1 byte per sub-vector

Memory per vector:
  Uncompressed: 768 × 4 bytes = 3,072 bytes
  PQ-compressed: 96 × 1 byte   = 96 bytes

Compression ratio = 3,072 / 96 = 32×

Total corpus memory:
  Uncompressed: 1,000,000 × 3,072 bytes ≈ 2.93 GB
  PQ-compressed: 1,000,000 × 96 bytes   ≈ 91.6 MB
```

**Recall/latency tradeoff — what changing nprobe does:**

```
nprobe = 1   → scan 1,000 vectors   → fastest,  lowest recall  (~70% recall@10 typical)
nprobe = 10  → scan 10,000 vectors  → balanced                  (~92% recall@10 typical)
nprobe = 100 → scan 100,000 vectors → slower,   near-exhaustive (~99% recall@10 typical)
```

This is the exact curve you'd sketch in an interview: **recall rises steeply then plateaus as nprobe increases, while latency rises roughly linearly** — the "knee" of that curve is where you pick your operating point.

---

## Why it works / why it fails

**Why it works:**
- IVF exploits the fact that real embeddings cluster — nearby documents in meaning end up geometrically nearby, so coarse clustering is a meaningful pre-filter, not a random partition.
- PQ exploits the fact that you don't need exact distances, only distances accurate enough to preserve *ranking order* among the top candidates.
- HNSW exploits small-world graph theory — most real point clouds admit navigable graphs where greedy routing gets you close to the true nearest neighbor in very few hops.

**Why it fails / where it breaks:**
- **IVF cluster imbalance.** If the embedding space has a few dense "hub" regions, some buckets become huge and others near-empty, so a fixed `nprobe` gives very uneven recall across queries.
- **PQ accuracy degrades with intrinsic dimensionality.** If the true signal doesn't compress well into `m` independent sub-spaces (dimensions are correlated), reconstruction error grows and recall drops — this is why OPQ (Optimized PQ, which rotates the space before splitting) is often layered on top.
- **HNSW build cost and memory.** The graph itself takes substantial RAM (each node stores `M` neighbor pointers per layer) and is slow to update incrementally — this is exactly why "retrieval freshness" (a later chapter) is its own hard problem: you can't just insert one vector into a mature HNSW graph as cheaply as you'd like.
- **Both approaches are approximate by design** — for domains where missing the single true best match is unacceptable (e.g., exact-match legal/medical retrieval), pure ANN indexing needs a re-ranking or brute-force fallback stage.

---

## The one thing to remember

ANN indexing is always a **three-way trade between recall, latency, and memory** — HNSW spends memory on graph structure to buy speed, IVF-PQ spends a controlled recall hit (via `nprobe` and quantization error) to buy both speed and memory savings; there is no free lunch, only where you choose to sit on the curve.

---

## Formulas used in this chapter

| Formula | Meaning |
|---|---|
| `P(layer ≥ l) = exp(-l / mL)` | Probability a vector is inserted into HNSW layer l |
| `scanned_fraction ≈ nprobe / nlist` | Fraction of corpus scanned under IVF |
| `compression_ratio = (D × 4) / m` | Memory savings from PQ (m = number of sub-vectors, 1 byte each) |
| `d(q,x)² ≈ Σᵢ ‖qᵢ − centroid(xᵢ)‖²` | Approximate distance via PQ lookup tables (ADC) |

---

## Interview Q&A

**Q1. Why does increasing `nprobe` improve recall but hurt latency, and how do you pick the right value in production?**

Increasing `nprobe` means scanning more IVF buckets, so more true nearest neighbors that happen to fall in nearby-but-not-closest clusters get a chance to be found — recall goes up. But you're linearly scanning more vectors per query, so latency goes up roughly linearly too. In production, you pick `nprobe` empirically: run offline recall@k evaluation (against a brute-force ground truth on a sample) at several `nprobe` values, plot recall vs. p99 latency, and pick the point at the "knee" of the curve where recall gains flatten out relative to latency cost — then validate against your actual SLA.

**Q2. Why can't you just increase `M` (neighbors per node) arbitrarily in HNSW to get better recall?**

Higher `M` means each node stores more neighbor edges, so the graph captures the true neighborhood structure more faithfully and greedy search is less likely to get stuck in a bad local minimum — recall improves. But memory grows linearly with `M` (you're storing `M` pointers per node per layer), and build time grows because each insertion does a more expensive neighbor search. Beyond a certain `M` (commonly 16–64 in practice), the marginal recall gain per additional edge shrinks sharply while memory keeps growing linearly, so it stops being worth it.

**Q3. What's the difference between what IVF does and what PQ does — why do you need both?**

IVF is about *not looking at* most of the corpus (a pruning/routing problem — reduce the number of candidates). PQ is about *making each candidate cheap to compare* once you do look at it (a compression/distance-computation problem). They solve different bottlenecks: IVF reduces how many vectors you touch; PQ reduces the cost of touching each one and how much RAM they occupy. Production ANN systems (FAISS's `IndexIVFPQ`, ScaNN) almost always combine both because neither alone is sufficient at billion-scale — IVF alone still leaves you scanning millions of uncompressed vectors, and PQ alone would still mean scanning the entire corpus, just more cheaply per vector.

**Q4. A teammate suggests using flat brute-force search but on GPU instead of ANN indexing — when would that actually be a reasonable choice?**

Brute-force on GPU is reasonable when the corpus is small-to-medium (up to a few million vectors) and you have batch or lower-QPS requirements, because GPUs offer enough raw FLOPs to make exact search fast enough, and you get *exact* results with zero recall loss and zero index-build complexity. It stops being reasonable once the corpus reaches hundreds of millions to billions of vectors and/or QPS is high, because even massively parallel exact search scales linearly with corpus size — no amount of extra GPU throughput changes that asymptotic, whereas ANN structures change the asymptotic itself (sub-linear in N).

**Q5. Why is HNSW harder to update incrementally than IVF-PQ, and why does this matter for a search system with fast-changing content?**

In HNSW, inserting a new node requires finding its nearest neighbors in the existing graph (an expensive greedy search) and potentially rewiring nearby nodes' edge lists to keep the graph well-connected — deletions are worse, since removing a node can fragment its neighbors' connectivity. In IVF-PQ, inserting a new vector is comparatively cheap: assign it to its nearest existing centroid and append it to that bucket's list; deletion is a soft-delete/tombstone. For a search system where content changes constantly (news, social content, product catalogs), this difference drives real architecture decisions — many production systems use IVF-style structures (or periodic HNSW rebuilds) specifically because true incremental HNSW updates at scale remain an open, actively-researched problem.
