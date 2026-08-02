# RAG Interview Prep — Day 6
## Review Day: Foundations Week (Days 1–5) — Closed Book

---

## 📋 How to run this review

1. **No notes.** Close Days 1–5 before starting.
2. Go question by question, write or say your answer out loud, *then* expand the `<details>` to check.
3. Use the **Weak Spot Tracker** at the bottom to log anything you got wrong or hesitated on — that's your Day 26-style repair list, and worth revisiting before Day 12's Retrieval review builds on top of this.
4. Target: get through all 25 questions in 60–90 minutes, including checking answers.

---

## Section A — Day 1: RAG vs. Fine-Tuning vs. Long-Context

**A1.** A company wants their support bot to always respond in a specific brand voice and JSON-structured format. Is this a fine-tuning problem, a RAG problem, or both? Why?

<details>
<summary>Show answer</summary>
Fine-tuning. This is a *behavior/format* problem, not a knowledge problem — no new facts are involved, just how the model expresses itself. RAG would only enter the picture if the bot also needed access to knowledge that's large or changes frequently.
</details>

**A2.** Name three distinct reasons (not just "context isn't big enough") why RAG remains valuable even with 1M+ token context windows.

<details>
<summary>Show answer</summary>
(1) Cost — you pay for the full context on every query even if a tiny fraction is relevant, scaling with corpus size. (2) Latency — processing huge prompts is slow. (3) Lost-in-the-middle — models attend less reliably to information buried in the middle of a long context, so "it's in there somewhere" doesn't guarantee correct use. (4) Access control — RAG can filter what's even eligible to be retrieved per user/permission; a static long context can't do this cleanly.
</details>

**A3 (calculation).** A knowledge base is 1,000,000 tokens. API cost is $2.50/million input tokens. Compare per-query cost for long-context (whole KB every call) vs. RAG (retrieve 6 chunks of 300 tokens + 100 tokens overhead).

<details>
<summary>Show answer</summary>

```
Long-context: 1,000,000/1,000,000 × $2.50 = $2.50/query
RAG: (6×300)+100 = 1900 tokens → 1900/1,000,000 × $2.50 = $0.00475/query
```
Long-context costs ~526x more per query.
</details>

---

## Section B — Day 2: Embeddings

**B1.** What is pooling, and name the two common strategies.

<details>
<summary>Show answer</summary>
Pooling collapses the per-token vectors a transformer encoder produces into a single fixed-length vector representing the whole input. Common strategies: mean pooling (average all token vectors) and CLS-token pooling (use one designated token's vector) — CLS pooling only works well if the model was actually trained with that objective in mind.
</details>

**B2 (calculation).** Compute cosine similarity between `A = [3, 4]` and `B = [6, 8]`. What property of cosine similarity does this demonstrate?

<details>
<summary>Show answer</summary>

```
A·B = 18+32 = 50
‖A‖ = √25 = 5, ‖B‖ = √100 = 10
cosine = 50/(5×10) = 1.0
```
B is A scaled by 2 — same direction, different magnitude. Result of exactly 1.0 demonstrates cosine similarity's magnitude-invariance.
</details>

**B3.** When does dot product become mathematically identical to cosine similarity, and why does this matter operationally?

<details>
<summary>Show answer</summary>
When both vectors are pre-normalized to unit length (‖A‖=‖B‖=1). It matters because dot product is cheaper to compute (skips the normalization division), so production systems often normalize once at indexing time and use fast dot product at query time to get cosine-equivalent rankings.
</details>

**B4.** Retrieval quality is mediocre on a legal-document corpus using a general-purpose embedding model. What's a likely high-leverage fix, and what's the operational cost of making that change?

<details>
<summary>Show answer</summary>
Switching to a domain-specific (legal) embedding model — general web-trained embeddings often underrepresent specialized vocabulary/semantic relationships in legal text, capping retrieval quality regardless of downstream tuning. The cost: this requires re-embedding the entire corpus (embedding drift — different models produce incompatible vector spaces), which is more disruptive than a chunk-size or k tweak but often higher-leverage.
</details>

---

## Section C — Day 3: Chunking

**C1.** Why is chunking considered one of the highest-leverage decisions in the RAG pipeline?

<details>
<summary>Show answer</summary>
Every downstream stage (embedding, retrieval, generation) operates on chunks, not raw documents. A chunk with one coherent idea produces a sharp embedding; a chunk with multiple unrelated ideas produces a blurred, ambiguous embedding that matches poorly against any single-topic query — capping quality regardless of how good the rest of the pipeline is.
</details>

**C2 (calculation).** A 5000-token document uses chunk size 500, overlap 125. Compute stride, approximate chunk count, and redundancy percentage.

<details>
<summary>Show answer</summary>

```
stride = 500-125 = 375
N ≈ ⌈(5000-125)/375⌉ = ⌈13⌉ = 13 chunks
total indexed = 13×500 = 6500 tokens
redundancy = (6500-5000)/5000 = 30%
```
</details>

**C3.** What problem does small-to-big (parent-document) retrieval solve, and how?

<details>
<summary>Show answer</summary>
It resolves the tension between search precision (favoring small, focused chunks) and generation completeness (favoring larger chunks with more context). Small chunks are indexed and searched for precision; each stores a pointer to a larger parent chunk. At query time, search happens over small chunks, but the larger parent chunks get passed to the generator — decoupling the search unit from the generation unit instead of compromising on one size.
</details>

**C4.** How does semantic chunking decide where to place boundaries, and what's its main cost trade-off?

<details>
<summary>Show answer</summary>
It embeds small units (typically sentences) and computes similarity between consecutive sentence embeddings; a sharp drop in similarity signals a topic shift, and a boundary is placed there. Main trade-off: requires an embedding call per sentence just to find boundaries, meaningfully more expensive at ingestion than fixed-size or recursive splitting — usually reserved for high-value corpora.
</details>

---

## Section D — Day 4: Vector Databases & Indexing

**D1.** Why does brute-force exact search stop being viable at scale, and roughly where's the tipping point?

<details>
<summary>Show answer</summary>
It's O(N×d) per query — linear in corpus size. Fine at tens of thousands of vectors (single-digit ms), but crosses from interactive to unusable somewhere around hundreds of thousands to a million vectors for latency-sensitive applications.
</details>

**D2 (calculation).** IVF with `nlist=800` over 8,000,000 vectors. At `nprobe=4`, how many vectors are compared, and what happens at `nprobe=40`?

<details>
<summary>Show answer</summary>

```
per cluster ≈ 8,000,000/800 = 10,000
nprobe=4:  4×10,000 = 40,000 vectors compared
nprobe=40: 40×10,000 = 400,000 vectors compared (10x more)
```
Higher nprobe improves recall (catches boundary-case neighbors in nearby clusters) at the cost of ~10x more comparisons and higher latency.
</details>

**D3.** Explain why HNSW generally handles incremental updates better than IVF.

<details>
<summary>Show answer</summary>
HNSW inserts new vectors via the same greedy-search mechanism as queries, wiring them locally into the existing graph without touching the rest of it. IVF's cluster centroids were computed from the original data distribution; as new data shifts that distribution, centroids go stale and recall degrades, eventually requiring a full re-clustering — a much heavier operation than HNSW's local insertion.
</details>

**D4.** What does Product Quantization do differently from simple int8 quantization, and why does it achieve much higher compression?

<details>
<summary>Show answer</summary>
Int8 quantization uniformly reduces each dimension's numerical precision (a fixed ~4x reduction). PQ splits each vector into sub-vectors and learns a small codebook of centroids per sub-vector slot via clustering, then represents each vector as centroid IDs (small integers) instead of raw values — exploiting actual data structure rather than uniform truncation, achieving much larger compression (often 100x+) at the cost of approximate distance calculations.
</details>

**D5 (system design).** You need to serve 100 million 768-dim vectors via HNSW at 1,500 QPS, sub-50ms p99. Walk through a rough capacity estimate.

<details>
<summary>Show answer</summary>

```
raw memory = 100,000,000 × 768 × 4 bytes ≈ 286 GB
HNSW overhead (~1.7x) ≈ 486 GB
```
This likely fits on a large single instance, but for QPS/fault-tolerance you'd still want replicas — e.g., if one replica sustains ~500 QPS at target latency, 1500/500 ≈ 3 replicas needed. If memory or QPS needs grow further, shard first (split vectors across machines), then replicate each shard for throughput.
</details>

---

## Section E — Day 5: Metadata Filtering & Multi-Tenancy

**E1.** Why isn't metadata filtering on a vector index "free" the way a relational `WHERE` clause is?

<details>
<summary>Show answer</summary>
ANN index structures (HNSW graphs, IVF clusters) are built assuming the full corpus is eligible for traversal. A filter that makes most of the corpus ineligible can break those structural assumptions, forcing degraded (sometimes near-brute-force) performance within the filtered subset unless the index is specifically designed to be filter-aware.
</details>

**E2 (calculation).** A filter matches 3% of a corpus. You retrieve top-15 by similarity, then post-filter. How many results would you expect on average, and what does this imply?

<details>
<summary>Show answer</summary>

```
Expected survivors ≈ 15 × 0.03 = 0.45
```
Post-filtering is unsafe here — you'd almost always get far fewer than 15 results. This is a highly selective filter, which calls for pre-filtering or a filter-aware/partitioned index instead.
</details>

**E3.** Why must RAG access control be enforced at the retrieval layer rather than just the UI layer?

<details>
<summary>Show answer</summary>
If retrieval queries the full corpus regardless of permissions and access control is only enforced at display time, unauthorized content can still be retrieved and fed into the generator's context — producing an answer that reveals or is conditioned on content the user was never authorized to see, silently, through fluent generated text rather than a visible document.
</details>

**E4.** Why is deleting a user's data from an HNSW index harder than deleting a row from a relational table?

<details>
<summary>Show answer</summary>
A vector in HNSW isn't an isolated record — it's a graph node wired to neighbors via multiple edges. Clean removal can require re-wiring neighbor connections. A common shortcut ("tombstoning" — marking deleted, filtering at query time) leaves the actual data physically present in the graph, which may not satisfy compliance requirements like GDPR's right to be forgotten.
</details>

---

## Section F — Cross-Day Synthesis (the hardest section — these mix concepts across days)

**F1.** Explain why a "chunk size too large" mistake (Day 3) might not show up clearly in Recall@k, but would show up in generation-stage metrics instead.

<details>
<summary>Show answer</summary>
A large multi-topic chunk's embedding (Day 2) becomes a blurred average of multiple ideas — it can still land "close enough" to a query about any one of those ideas to be retrieved in a generous top-k, so Recall@k can look fine. But the retrieved chunk is diluted with irrelevant surrounding content, which shows up downstream as lower context relevance and more hallucination risk at the generation stage — not as an obvious retrieval-stage metric drop.
</details>

**F2.** You're designing a multi-tenant RAG system (Day 5) at very large scale (500M+ vectors total). How does the choice between shared-index-with-filter vs. per-tenant-namespace interact with the HNSW vs. IVF choice (Day 4)?

<details>
<summary>Show answer</summary>
If you go with per-tenant namespaces/partitions, each tenant effectively gets a smaller, self-contained index — for tenants with frequently-updated data, HNSW's graceful incremental updates are attractive per-namespace. If tenants are large enough individually that memory becomes the binding constraint, IVF-PQ's compression becomes more relevant within a given tenant's partition. If instead you use one giant shared filtered index, you need the index to be genuinely filter-aware to avoid post-filtering waste (Day 5's selectivity math) — and a single graph over hundreds of millions of vectors with filtering baked in is a harder engineering problem than several smaller per-tenant HNSW graphs. The namespace/partition approach often sidesteps a lot of the filtering-at-scale complexity by construction.
</details>

**F3.** Why does the embedding drift problem (Day 2) make "switching to a better embedding model" a much bigger decision than it initially sounds, and how does this interact with a production vector database serving live traffic (Day 4/5)?

<details>
<summary>Show answer</summary>
Different embedding models produce incompatible vector spaces, so upgrading requires re-embedding the *entire* corpus, not just new documents — you can't mix old and new embeddings in the same index without producing meaningless similarity scores. In a production system serving live traffic, this means the migration needs a real strategy (e.g., dual-writing to old and new indexes during a transition window, or a scheduled cutover), rather than an incremental swap — directly connecting a modeling decision (embedding model choice) to an infrastructure/operations problem (zero-downtime index migration).
</details>

**F4.** A RAG system for a healthcare company needs: (a) strict per-customer data isolation, (b) documents updated continuously throughout the day, (c) users within each customer having different permission levels. Name the Day 3/4/5 concepts you'd combine and why.

<details>
<summary>Show answer</summary>
(a) Points to per-tenant namespaces or fully separate indexes (Day 5) given the compliance stakes. (b) Points to HNSW over IVF (Day 4) for its graceful incremental-update behavior, since frequent updates would cause IVF's cluster centroids to go stale. (c) Requires enforcing user-level permission filtering at the retrieval layer within each tenant's namespace (Day 5's access-control point), not just at the UI. Chunking strategy (Day 3) would also matter for a healthcare corpus specifically — likely recursive or semantic chunking to preserve clinical instruction coherence, given how costly a badly-split medical instruction chunk could be.
</details>

---

## 📊 Weak Spot Tracker

Fill this in honestly — anything you hesitated on or got wrong goes here. Revisit these specific items before Day 12 (Retrieval review), since Retrieval builds directly on Foundations.

| Question # | Topic | Got it cold? | Needs repair? |
|---|---|---|---|
| A1–A3 | RAG vs FT vs LC | ☐ | ☐ |
| B1–B4 | Embeddings | ☐ | ☐ |
| C1–C4 | Chunking | ☐ | ☐ |
| D1–D5 | Vector DB / Indexing | ☐ | ☐ |
| E1–E4 | Filtering / Multi-tenancy | ☐ | ☐ |
| F1–F4 | Cross-day synthesis | ☐ | ☐ |

**Rule from the curriculum:** if a synthesis question (Section F) tripped you up, that's more important to fix than a single-day recall question — synthesis is what interviewers actually probe for once they know you have the definitions down.

---

*Foundations week complete. Next up — Day 7: Sparse Retrieval (BM25, TF-IDF), starting the Retrieval week.*
