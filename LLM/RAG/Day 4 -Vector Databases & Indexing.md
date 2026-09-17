RAG Interview Prep — Day 4 (Refined)
Vector Databases & Indexing — Full Deep Dive
Refined version: Q&A unhidden (no collapsed <details>), content tightened and de-duplicated, and a new Google-specific angle section + extra practice questions added, since ScaNN, Vertex AI Matching Engine, and Google-scale system design are fair game in a Google ML interview.

🚀 Quick Summary
A vector database exists to answer one question fast, at scale: "which of my millions/billions of vectors are closest to this query vector?" Doing that fast means trading a small amount of accuracy for a huge amount of speed, via Approximate Nearest Neighbor (ANN) algorithms. Three orthogonal levers production systems pull:

How do I avoid comparing against every vector? → graph navigation (HNSW) or cluster pruning (IVF)
How do I avoid storing every vector at full size? → compression (Product Quantization / ScaNN)
How do I avoid one machine being the bottleneck? → sharding + replication
Everything else hangs off those three questions.

Warehouse analogy: walking every aisle to find one box (brute-force) works but doesn't scale. HNSW builds hub-and-spoke shortcuts through the warehouse. IVF pre-sorts boxes into labeled zones and only searches nearby zones. PQ shrinks every box to a compressed summary so more fit on the shelf. Which filing system you pick depends on corpus size, update rate, memory budget, and latency requirement.

🧠 How Indexing Actually Works — Plain-Language Walkthrough
Step 1 — What "search" means for vectors. Every chunk gets embedded (say, 768 numbers). A query gets embedded into a vector of the same length. "Relevant" becomes "geometrically close" (cosine similarity, dot product, or Euclidean distance). Retrieval becomes: given a point in 768-dim space, find the nearest points among millions of others.

Step 2 — Why you can't just "look it up." A hash map or B-tree works because it rules out most data with one comparison. Nearest-neighbor search has no such shortcut in high dimensions — there's no ordering where "close in space" maps to "close in a sorted list." This is the curse of dimensionality: as dimensions grow, classic exact-search shortcuts degrade toward "compare against everything." Exact search is O(N × d) — linear, no way around it.

Step 3 — The ANN insight. Accept "almost certainly the nearest neighbor, via a good heuristic" instead of "guaranteed nearest neighbor," and you can answer most queries by touching only a small fraction of the data. Every algorithm below is a different heuristic for narrowing candidates before doing the real distance comparison.

Step 4 — Two independent problems people conflate:

Search-narrowing (which vectors do I even compare against?) → HNSW's graph or IVF's clusters.
Storage-shrinking (how small can each vector be in memory?) → Product Quantization, ScaNN, or simpler float16/int8 quantization.
You can mix any search-narrowing method with any storage-shrinking method — that's literally IVF-PQ: IVF narrows candidates, PQ shrinks what's stored.

Step 5 — What "building the index" means concretely:

HNSW: as each vector is inserted, a greedy search over the graph built so far finds where it belongs, then wires edges to nearby existing nodes at each randomly-assigned layer. Incremental by construction — no separate training phase.
IVF: run k-means once over a representative sample to fix nlist centroids. Every vector (existing and new) is assigned to its nearest centroid. This is a training phase — centroids are fit to a snapshot and don't move afterward.
PQ: independently per sub-vector "slot," run k-means to learn a small codebook (e.g., 256 centroids). Each sub-vector segment is replaced by "which codebook entry is closest" — an integer ID instead of raw floats. Also a training phase, fit once and reused.
Step 6 — Query time:

(If IVF) compare query to nlist centroids, pick the nprobe closest.
(If HNSW) greedily walk down through graph layers from an entry point.
Within the resulting candidate set, compute real (or PQ-approximated) distances, return top-k.
Optionally re-rank top-k' candidates using full-precision vectors to undo compression error (two-stage pattern below).
The Full Indexing Landscape
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

               COMPRESSION LAYER (orthogonal — combine with any of the above)
                                     │
                    PRODUCT QUANTIZATION (PQ)  ──or──  ScaNN (anisotropic PQ)
                    shrinks memory footprint, often paired with IVF as "IVF-PQ"
Key framing: HNSW and IVF both answer "how do I avoid comparing against every vector"; PQ/ScaNN answer "how do I avoid storing every vector at full precision." Production systems often combine them (IVF-PQ).

Algorithm-by-Algorithm Mechanics
1. Flat / Brute-Force (Exact Search)
O(N × d) per query. Fine under ~100K–1M vectors, or as the ground-truth baseline for measuring ANN recall.

Time estimate (d=768, ~1μs/comparison):

N = 10,000:      10 ms   → fine
N = 1,000,000:   1 s     → too slow
N = 100,000,000: 100 s   → unusable
GFLOPS-based framing:

N = 10,000,000 docs, d = 768
Total ops/query = 10M × 768 = 7.68 billion ops
At 10 GFLOPS (1 CPU core) ≈ 0.77s/query
Target: <50ms → off by ~15x, before any other overhead
Recall@k definition:

Recall@k = (relevant vectors in ANN top-k) / (relevant vectors in exact top-k)
0.90–0.97 recall is typically acceptable for RAG — the downstream LLM tolerates occasional missed chunks.

2. HNSW (Hierarchical Navigable Small World)
Structure: multi-layer graph — sparse "highway" layers on top, dense "local streets" at the bottom (every vector lives in the bottom layer).

Search: enter top layer → greedily hop toward the query until no neighbor is closer → drop a layer → repeat → final local search at layer 0.

Param	Controls	Effect of increasing
M	max connections/node	↑ recall, ↑ memory, slower build
ef_construction	build-time search effort	better graph, much slower build
ef_search	query-time search effort	↑ recall, ↑ latency — main serving knob, no rebuild needed
Recall/latency curve (diminishing returns):

ef_search=10:   recall≈0.85, latency≈1.2ms
ef_search=50:   recall≈0.95, latency≈3.5ms
ef_search=200:  recall≈0.99, latency≈9.0ms
Memory: raw vectors (N × d × 4 bytes) + graph edges. Two rules of thumb:

Simple: graph overhead ≈ 1.5–2× raw vector size.
Granular: graph overhead ≈ N × M × 4 bytes/ID × ~2 layers-average — e.g. 50M nodes × 16 × 4 × 2 ≈ 6.4 GB on top of ~307 GB raw — a much smaller fraction than 1.5–2× when M is modest. Real multiplier depends on M and layer distribution.
Updates: incremental insert via the same greedy search, no rebuild — HNSW's single biggest practical advantage for continuously-changing RAG corpora.

Deletion is nontrivial: removing a node means repairing its neighbors' edges — O(M·log N) per deletion. See the tombstone pattern below.

3. IVF (Inverted File Index)
Build: k-means over the corpus → nlist centroids; each vector assigned to nearest centroid. Query: compare query to nlist centroids (cheap), then full search only within the nprobe closest clusters.

Param	Controls	Effect of increasing
nlist	# clusters	finer partitioning, faster per-cluster search, but boundary-case recall risk
nprobe	# clusters searched	↑ recall, ↑ latency (direct analogue of ef_search)
Worked example (10M vectors, nlist=1000):

nprobe=1:   ~10,000 vectors searched  → fast, misses boundary cases
nprobe=10:  ~100,000 vectors searched → catches more boundary cases
nprobe=100: ~1,000,000 vectors searched → near-brute-force recall, slow
Compact general formula:

Total vectors searched = (N / nlist) × nprobe
Speedup vs brute force ≈ nlist / nprobe
Why boundary vectors get missed: a vector near a cluster boundary is assigned to only one centroid; a query landing just across that boundary won't find it unless nprobe also covers the neighboring cluster.

The staleness problem (say this proactively — a classic interview gotcha): centroids are trained on a snapshot. As new data arrives and the distribution shifts, centroids stop matching reality — vectors get assigned to suboptimal clusters and nprobe's nearest centroids increasingly miss relevant vectors. This degradation is silent — no error, just slowly worsening recall. Fix: monitor recall on a held-out eval set; rebuild/retrain on a schedule or when recall drops below a threshold.

4. Product Quantization (PQ)
Problem it solves: memory, not search speed — that's IVF's job.

Mechanism:

Split each d-dim vector into m sub-vectors.
Per slot, k-means over the whole corpus → small codebook (e.g. 256 centroids).
Replace each sub-vector with its nearest codebook centroid's ID.
Two worked compression examples (know the shape, not one magic number):

m=8, 256-entry codebooks, d=768:
  Raw: 768×4 = 3072 bytes → PQ: 8×1 byte = 8 bytes → 384× compression

m=96, 256-entry codebooks, d=768:
  Raw: 3072 bytes → PQ: 96×1 byte = 96 bytes → 32× compression
Takeaway: compression ratio scales with how many sub-vectors you split into (m) — fewer, larger sub-vectors (small m) compress harder but lose more fidelity per slot; more, smaller sub-vectors (large m) compress less but preserve more structure. No single "correct" m — it's a tuned trade-off.

Accuracy cost: distances become approximate (quantization error). Standard mitigation — the two-stage pattern:

Stage 1: IVF-PQ search over full index → fast, approximate, returns top-k' (e.g. k'=100)
Stage 2: exact re-score of just those k' with full-precision vectors → tiny, cheap, recovers accuracy
This is the same pattern as bi-encoder (cheap, approximate) + cross-encoder (expensive, precise) reranking — just applied at the index/storage level instead of the retrieval-scoring level. Drawing this connection out loud is a strong interview signal.

Why it matters: 1B vectors at float32 ≈ 2.86 TB (impractical); PQ-compressed ≈ single-digit GB (comfortably in RAM on one machine).

5. LSH (Locality-Sensitive Hashing)
Hash functions designed so similar vectors collide into the same bucket (opposite goal of a cryptographic hash). Only compare within the query's bucket(s). Historically important, generally outperformed by HNSW/IVF-PQ on modern high-dim embeddings — know it exists, don't over-invest.

6. ScaNN — Google's Anisotropic Quantization
Standard PQ minimizes quantization error uniformly across all dimensions. ScaNN's insight: not all quantization error matters equally for ranking. For inner-product search, error parallel to the query direction (which shifts the dot-product ranking) hurts far more than error orthogonal to the query (which barely changes relative ranking). ScaNN penalizes parallel-direction error more heavily during codebook training, preserving ranking order better than standard PQ at the same compression ratio.

Result: consistently near the top of ann-benchmarks.com on recall-vs-QPS. Implemented in Google Vertex AI Vector Search (formerly Matching Engine). You need the why, not implementation depth, for most interviews — but naming it unprompted when asked "besides PQ, what else compresses vectors" is a strong signal, especially in a Google interview, since it's Google Research's own contribution to this space (published at ICML 2020, "Accelerating Large-Scale Inference with Anisotropic Vector Quantization").

🎯 Google-Specific Angle (worth knowing cold for a Google ML interview)
ScaNN is Google's own algorithm — open-sourced, used internally and exposed via Vertex AI Vector Search. Interviewers may specifically probe whether you know why it beats plain PQ (the anisotropic/parallel-error insight above), not just that it exists.
Vertex AI Vector Search is Google Cloud's managed ANN service — conceptually parallel to Pinecone, but built on ScaNN under the hood, and integrated with the rest of Vertex AI (embeddings API, feature store, pipelines).
Google's internal infra lineage for this problem includes Google's original ANN work at web-search scale (this is the same "search-narrowing at planet scale" problem that shows up in web index sharding) — the interviewer likely cares more that you can reason about sharding/replication/capacity trade-offs from first principles than that you memorize product names.
If asked to "design a semantic search system for Google-scale data," the expected shape is: embedding generation → ANN index (ScaNN/HNSW) → sharding by hash or semantic cluster → replication for QPS and fault tolerance → a re-ranking stage → freshness/update strategy (tombstones, scheduled rebuilds) → monitoring recall drift. Structuring your answer exactly along these six beats is itself a signal of seniority.
Be ready to connect this topic to embeddings quality (two-tower models, dual encoders) since Google interviewers often pair "how do you index vectors" with "how do you produce good vectors in the first place" (matching Module 1 material on bi-encoders/cross-encoders).
Master Comparison Table
Flat (exact)	HNSW	IVF	IVF-PQ	ScaNN
Accuracy	Perfect	High (ef_search)	High (nprobe)	Slightly lower	High at same compression as PQ
Speed at scale	Unusable >~1M	Fast, consistent	Fast, depends on nprobe	Fastest at billion-scale	Fast, near top of benchmarks
Memory	Highest	High (vectors+graph)	Moderate	Lowest	Low
Build time	None	Slower	Faster	Moderate	Moderate-high (codebook training)
Incremental updates	Trivial	Well	Poorly (stale centroids)	Poorly	Poorly (same IVF-family limitation)
Best for	Small / ground-truth	Frequently-updated latency-sensitive RAG	Large static, memory-conscious	Billion-scale, memory-bound	Billion-scale, recall-per-byte-optimized
FAISS index-type cheat sheet:

FAISS Index	Mechanism	Memory	Speed	Recall	Use when
IndexFlatL2/IndexFlatIP	brute-force	High	Slow	Perfect	≤100K vectors, or ground truth
IndexIVFFlat	IVF + exact within cluster	Medium	Fast	High	Medium corpora, recall > memory savings
IndexIVFPQ	IVF + PQ	Very low	Very fast	Lower	100M+ vectors, memory-bound
IndexHNSWFlat	HNSW, full precision	High	Very fast	Very high	Best recall/speed if RAM allows
Library vs. system — the distinction interviewers probe for: FAISS is a library, not a database. It gives you the ANN algorithms; you own persistence (indexes are in-memory — you serialize/load them), metadata storage (only vectors + integer IDs), filtering (no native support), sharding (single-node only), and updates (limited deletion, you manage tombstoning). This is exactly why managed vector databases exist — they wrap FAISS-equivalent algorithms with production infrastructure around all five of those gaps.

Real Vector Database / Search Products
Product	Type	Notable characteristics
FAISS	Library	Meta's reference implementation (Flat, IVF, HNSW, PQ, combinations) most tools build on or benchmark against
Pinecone	Managed SaaS	Fully managed, abstracts index-type choice, strong metadata filtering + multi-tenancy at scale
Weaviate	OSS / managed	HNSW-based, strong hybrid (sparse+dense) search, GraphQL-style interface
Milvus	OSS, distributed-first	HNSW/IVF/IVF-PQ, horizontal sharding as a first-class concern
Qdrant	OSS, Rust	HNSW-based, standout filtered-HNSW traversal performance
pgvector	Postgres extension	Vector search inside your existing relational DB — no second system, transactional consistency with metadata
Vertex AI Vector Search	Google Cloud managed	Built on ScaNN, integrates with Vertex embeddings/pipelines
pgvector concrete example — worth having memorized:

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    department TEXT,
    updated_at TIMESTAMPTZ,
    embedding VECTOR(768)
);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

SELECT content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE department = 'legal' AND updated_at > NOW() - INTERVAL '1 year'
ORDER BY embedding <=> $1
LIMIT 10;
Value proposition in one sentence: metadata and vectors share the same transaction — no sync problem between a separate metadata store and a separate vector index.

Decision framework as a flowchart:

Already running Postgres in production?
  → Yes → pgvector (avoid adding a new system dependency)
  → No ↓
Need zero-ops / fully managed / fastest time-to-prod?
  → Yes → Pinecone (or Vertex AI Vector Search if already on GCP)
  → No ↓
Need self-hosting (compliance / data residency)?
  → Yes → Weaviate / Qdrant / Milvus
       → rich schema + graph features → Weaviate
       → strongest filter performance → Qdrant
       → billion-scale + deep customization → Milvus
Why This Matters: the strong interview move is never naming a favorite product — it's naming the requirements that should drive the choice (scale, QPS, update frequency, filtering/multi-tenancy needs, managed vs. self-hosted, existing infra investment). "We're already heavily invested in Postgres" is a legitimate reason to pick pgvector even when it's not the highest-performance option at extreme scale.

🆕 Additional Standard Choices to Evaluate: MongoDB & OpenSearch
These aren't purpose-built vector databases, but both are extremely common in real production RAG stacks because teams already run them for other reasons — the same "avoid a new system dependency" logic that makes pgvector attractive.

MongoDB Atlas Vector Search
What it is: a vector index type (built on Lucene HNSW under the hood, via Atlas Search) added to MongoDB Atlas, letting you store embeddings as a field in a normal document and query with a $vectorSearch aggregation stage.

Mechanism: HNSW-based ANN, integrated into the existing aggregation pipeline — a single query can combine $vectorSearch with normal $match filters, similar in spirit to pgvector's SQL WHERE + vector ORDER BY.

Strengths: no new system if your app data is already in MongoDB; native pre-filtering + vector search in one query; fully managed on Atlas.

Weaknesses: locked into Atlas for the full feature set; less battle-tested at extreme scale than FAISS-lineage systems; no choice of underlying ANN algorithm (you get Atlas's implementation).

When it's the right call: teams already on MongoDB, small-to-medium vector scale, wanting filtering + vector search unified without adding a new system.

OpenSearch (k-NN plugin)
What it is: ANN search bolted onto OpenSearch (the Elasticsearch fork) via its k-NN plugin, wrapping FAISS, Lucene HNSW, and (historically) nmslib.

Mechanism: because it wraps FAISS/Lucene, OpenSearch gives you a choice of underlying algorithm (HNSW or IVF-family via FAISS engine) — closer to FAISS's flexibility than most managed vector DBs, while still being a full search engine around it.

Strengths: best-in-class for hybrid dense+sparse search — OpenSearch is a text search engine (BM25/inverted index) first, so combining lexical and semantic scoring is a first-class use case, arguably its single strongest differentiator; mature filtering/aggregations/access-control from its search-engine lineage.

Weaknesses: operationally heavier than a purpose-built vector DB if vector search is your only need; HNSW-in-OpenSearch generally trails dedicated vector-native systems at very large vector-only scale.

When it's the right call: genuine hybrid search (keyword + semantic) is a hard requirement — e.g. exact product SKUs or legal citation numbers must match alongside semantic similarity — and/or you already run an Elastic/OpenSearch cluster.

Where Mongo & OpenSearch slot into the decision flowchart
Already running Postgres?        → pgvector
Already running MongoDB?         → MongoDB Atlas Vector Search
Already running OpenSearch/ELK?  → OpenSearch k-NN
                                     (especially if hybrid lexical+semantic search matters)
None of the above / greenfield?  → Pinecone / Vertex AI Vector Search (zero-ops)
                                     or Weaviate/Qdrant/Milvus (self-hosted, by feature need)
One-line synthesis: pgvector, MongoDB Atlas Vector Search, and OpenSearch k-NN are all instances of the same rule — "extend the database you already operate rather than add a new specialized one" — differentiated mainly by which database you already operate, plus OpenSearch's edge when hybrid search is a hard requirement.

Updated Full Feature Comparison Table
Pinecone	Weaviate	Qdrant	Milvus	pgvector	MongoDB Atlas	OpenSearch
Ops model	Fully managed SaaS	OSS + managed	OSS + managed	OSS + managed	Postgres extension	Managed (Atlas)	OSS + managed
Underlying algorithm	Proprietary (abstracted)	HNSW	HNSW (filter-aware)	HNSW/IVF/IVF-PQ	HNSW (via extension)	HNSW (Lucene-based)	HNSW or IVF (FAISS/Lucene engines)
Hybrid search	Native	Native (BM25+vector)	Native	Supported	Manual	Supported via aggregation	Best-in-class (core strength)
Metadata filtering	Strong	Strong (GraphQL)	Strong, filter-aware traversal	Strong	SQL WHERE	Native via $match	Mature (inherited from search-engine lineage)
Horizontal scale	Automatic	Distributed mode	Distributed mode	Cloud-native distributed	Limited	Atlas-managed sharding	Cluster sharding (heavier ops)
Best for	Zero-ops teams	Rich schema/graph	Filter-heavy workloads	Very large scale	Already on Postgres	Already on MongoDB	Need hybrid lexical+semantic, or already on ELK/OpenSearch
Metadata Filtering at Scale
Post-filtering: search full index first, discard non-matches after. Breaks on highly selective filters — a 0.1%-match filter can leave your top-k nearly empty even though 50 relevant matches sit just outside the searched window.

Pre-filtering (naive): restrict candidates by metadata before ANN search. Breaks on high-cardinality filters — filtering a 10M-vector HNSW graph down to 500 docs for one user_id essentially disables the index (the graph's shortcuts assumed the whole corpus was eligible), and you'd need one index per user at scale, which doesn't work.

Filter-aware / hybrid traversal (the real production answer): push the filter into the graph traversal itself. During HNSW walk, a node failing the filter is skipped (doesn't count toward your ef_search budget) but its neighbors are still explored — so the search self-routes toward the relevant region without ever pre-restricting the candidate pool or wasting result slots. Qdrant and Weaviate implement this natively; OpenSearch's filtering (inherited from its search-engine core) is also mature here.

Gotcha: don't describe filtering as solved by "adding a WHERE clause" — how the filter interacts with the ANN structure is real, actively-evolving engineering.

Index Update Strategies
Real-time upsert (HNSW): incremental insert, no retraining. Good for continuous small-volume ingestion. Caveat: very high concurrent insert load can slightly degrade graph quality; implementations use locking/lock-free structures to mitigate.

Batch rebuild (IVF-family): centroids trained on a snapshot, don't self-update. The silent decay pattern:

T=0:   train on 1M docs → good boundaries
T=3mo: insert 200K new docs → assigned to old centroids
T=6mo: recall@10 silently drops 0.95 → 0.87
T=?:   users notice degraded answers — hard to diagnose without eval monitoring
Fix: instrument recall on a held-out eval set, alert on drift, rebuild on schedule or threshold.

Tombstone pattern for deletes: most ANN structures can't cheaply delete a single node — repairing an HNSW node's neighbors' edges is O(M·log N) and can't be done cheaply under high-throughput writes. Standard pattern instead:

1. Mark the vector as a tombstone in metadata (soft delete) — O(1)
2. Filter tombstoned vectors out at query time (check metadata before returning)
3. Periodic compaction: rebuild the index skipping tombstoned vectors, reclaiming storage/graph space
Mandatory rebuild trigger — embedding model migration (one of the highest-signal gotchas): if you swap embedding model A for model B and only embed new documents with B while old documents stay embedded with A, you now have two geometrically incompatible vector spaces in one index. Cosine similarity between a model-B query and a model-A document is meaningless — the retriever silently returns garbage for anything that should match old documents, with no error signal, and standard monitoring won't catch it unless your eval set specifically includes old-document queries. Correct fix: full re-embed of the entire corpus with model B, full index rebuild, then a blue-green swap — build the new index offline, validate recall against the eval set, cut over traffic atomically. This is why embedding-model migrations are expensive projects, not incremental changes.

Scaling: Sharding, Replication, Capacity Planning
Sharding: split the index across machines when one machine can't hold it in memory or sustain required QPS. Trade-off: fan-out + tail-latency risk (query is only as fast as its slowest shard).

Two concrete sharding strategies:

Random / hash sharding	Semantic / cluster-based sharding
Mechanism	shard = hash(vector_id) % num_shards	coarse clustering assigns clusters to shards; route query to shards whose centroids are near it
Simplicity	Much simpler	Complex routing layer
Load balance	Perfectly balanced	Can be skewed by uneven cluster sizes
Fan-out	Always full fan-out (every shard queried every time)	Partial fan-out — only relevant shards queried
Use when	Small shard counts (≤10)	Many shards, query latency is critical and you can afford routing complexity
Replication: duplicate shards for read throughput + fault tolerance. Most vector DBs favor eventual consistency across replicas (a new insert might not be immediately searchable everywhere) — an acceptable trade for RAG, where sub-second staleness on brand-new documents rarely matters.

Memory math — the core formula:

Memory (raw vectors) = N × d × bytes_per_float
  float32 → 4 bytes/dim, float16 → 2 bytes/dim
Scaled example table:

Scenario	N	d	Format	Raw memory
Small startup	1M	384	float32	1.5 GB
Medium product	10M	768	float32	30 GB
Large enterprise	100M	768	float32	300 GB (exceeds 1 machine)
Web-scale	1B	1536	float32	6 TB (needs PQ + sharding)
Capacity-planning walkthrough #1 (200M vectors / 2000 QPS):

1. Raw memory: 200M × 768 × 4B ≈ 572 GB
2. HNSW overhead (~1.7×): ≈ 972 GB total
3. Exceeds single-machine RAM → sharding required
4. 4 shards of 50M vectors ≈ 243 GB/shard, fits comfortably
5. If 1 replica sustains ~600 QPS, need 2000/600 ≈ 3.3 → round to 4 replicas/shard
Total nodes ≈ 4 shards × 4 replicas = 16 nodes
Capacity-planning walkthrough #2 (50M vectors, 1536-dim, more granular overhead math):

1. Raw: 50M × 1536 × 4B = 307 GB
2. HNSW graph overhead (M=16, ~2 layers avg): 50M × 16 × 4B × 2 ≈ 6.4 GB
3. Total ≈ 313 GB — exceeds a typical 128–256GB memory-optimized instance
4. Options, in order of preference: float16 (halves to ~154GB, minimal precision loss)
   → then PQ (16-32× reduction) if still over budget → then shard as last resort
Having both a "graph overhead is 1.5–2× raw" rule of thumb and a granular per-node-edge derivation lets you pick whichever the interviewer's framing suggests — and explicitly flagging the estimate as a rule of thumb either way is itself part of the signal.

Interview Q&A Practice Set
Q1 (Easy). Why does brute-force search stop scaling, and roughly where's the tipping point?

A: O(N×d) per query — linear in corpus size. Fine at tens of thousands of vectors (single-digit ms). Past roughly hundreds of thousands to a million vectors, latency crosses from interactive to production-unacceptable — that's the point ANN becomes necessary rather than optional.

Q2 (Easy — calculation). 5M vectors, IVF nlist=500, nprobe=5 vs nprobe=50 — vectors compared, and the trade-off?

A: ~10,000 vectors/cluster. nprobe=5 → ~50,000 compared; nprobe=50 → ~500,000 (10× more). Higher nprobe improves recall on boundary cases at ~linear latency cost — the standard IVF dial.

Q3 (Medium). How does PQ differ from simple int8 quantization, and why the bigger compression ratio?

A: int8 quantization uniformly truncates precision on every dimension (fixed ~4×). PQ instead learns per-sub-vector-slot codebooks via clustering across the whole corpus, then stores just a centroid ID per slot — exploiting actual data structure/redundancy rather than uniform truncation, yielding much larger ratios (32×–384×+ depending on m), at the cost of approximate distances.

Q4 (Medium). Why does HNSW handle frequent updates better than IVF?

A: HNSW inserts via the same greedy search used for queries — local, incremental, no rebuild. IVF's centroids are fit once to a snapshot; as the distribution shifts they go silently stale, degrading recall until a full re-clustering pass is needed — a much heavier operation.

Q5 (Medium — system design). Sub-100ms p99, filter by tenant_id for strict isolation — shared filtered index vs. per-tenant index?

A: Shared index+filter: cheaper, simpler, pools storage — but a filtering bug is a cross-tenant data leak, and if the ANN structure isn't filter-aware, small tenants get poor recall/near-brute-force behavior inside a huge shared corpus. Per-tenant index: strong isolation by construction, predictable per-tenant performance — but ops overhead scales linearly with tenant count. Choose per-tenant for a small number of large/high-compliance tenants; shared filter-aware index for many small tenants.

Q6 (Hard — calculation). 1B vectors, d=768 — raw float32 memory vs. PQ (m=8, 256-entry codebooks)?

A: Raw: 1B × 768 × 4B ≈ 2.86 TB. PQ: 8 bytes/vector × 1B ≈ 7.45 GB. This is why PQ becomes necessary, not optional, once you're at hundreds-of-millions-to-billions scale — the alternative is a very large, expensive sharded cluster just to hold raw vectors in memory.

Q7 (Hard — synthesis). 300M vectors, d=768, HNSW, 3000 QPS, sub-50ms p99 — how many machines?

A: Raw: 300M×768×4B ≈ 858 GB. ×1.7 graph overhead ≈ 1.46 TB → exceeds one machine → shard into 6×50M (~243GB/shard). If a replica sustains ~500 QPS, need 3000/500=6 replicas/shard. Total ≈ 6×6 = 36 nodes. The reasoning chain matters far more than nailing the exact constants.

Q8 (Medium). Team wants real-time updates on a system currently using FAISS IndexIVFPQ — what's wrong, and what do you recommend?

A: Two issues: IVF centroids silently go stale as new data shifts the distribution, and FAISS itself has no native persistence, filtering, or deletion — you'd be building that infrastructure yourself regardless. Recommend switching to an HNSW-based system (FAISS IndexHNSWFlat, or a managed system like Qdrant/Weaviate with production infra built in) for incremental inserts with no retraining. If IVF-PQ's memory efficiency must be kept, instrument recall monitoring, alert on drift, and batch-buffer new documents for scheduled rebuilds rather than expecting real-time freshness.

Q9 (Medium). When would pgvector, MongoDB Atlas Vector Search, or OpenSearch beat a purpose-built vector DB?

A: All three share the same logic: if you already operate that database for your primary application data, adding vector search to it avoids a second system, a sync problem between separate metadata and vector stores, and extra ops burden — at the cost of some ceiling on raw ANN performance/scale versus a dedicated vector-native system. OpenSearch has one further specific edge: if hybrid lexical+semantic search (exact keyword/entity matches alongside embedding similarity) is a hard requirement, its search-engine lineage makes it a stronger fit than any of the vector-native options.

Q10 (Hard). What breaks if you migrate embedding models without a full rebuild?

A: Old and new documents end up embedded in two geometrically incompatible vector spaces sharing one index. Similarity scores between a new-model query and an old-model document are meaningless, so the retriever silently returns garbage for anything that should match older content — with no error signal, and standard monitoring misses it unless the eval set specifically covers old-document queries. Correct approach: full re-embed of the whole corpus, full rebuild, then a validated blue-green cutover.

Q11 (Medium — new). Why does ScaNN outperform standard PQ at the same compression ratio, and where would you mention it if not asked directly?

A: Standard PQ minimizes quantization error uniformly across all dimensions. But for ranking purposes, error parallel to the query direction distorts the dot-product ranking far more than error orthogonal to it. ScaNN's anisotropic loss penalizes parallel-direction error more heavily during codebook training, so it preserves the relative order of nearest neighbors better than PQ at equal compression. Mention it whenever asked "besides PQ, what other compression techniques exist" or "how would Google's own systems handle this" — it signals depth beyond the FAISS-only mental model.

Q12 (Hard — new, system design). Design a semantic search backend for 5 billion product embeddings (d=768) that needs sub-100ms p99 and daily catalog updates. Walk through your reasoning.

A:

Raw memory: 5B × 768 × 4B ≈ 15.4 TB float32 — far beyond any single machine or reasonable cluster if kept raw.
Compression is mandatory, not optional: apply PQ or ScaNN (16–64× range) to get into the hundreds-of-GB range, e.g. ScaNN at ~32× → ~480 GB total — now shardable across a small cluster.
Search-narrowing: combine with IVF (coarse clusters) or HNSW-over-compressed-vectors for the "which vectors do I compare against" problem — IVF-PQ or ScaNN's own partitioning is the standard billion-scale combo.
Sharding: hash-based if updates are frequent and uniform (simplicity), semantic/cluster-based if query latency dominates and you can afford routing complexity.
Replication: size for QPS and fault tolerance, accept eventual consistency.
Update strategy: daily catalog updates → batch rebuild window (IVF/PQ retraining) is realistic at daily cadence; monitor recall drift between rebuilds; use tombstones for intra-window deletes (discontinued products).
Re-ranking: two-stage — compressed ANN for candidate generation, then exact re-score of top-k' with full-precision vectors (or a learned re-ranker) to recover precision before returning results.
Monitoring: held-out eval set tracking recall@k over time, alerting on drift — since both centroid staleness and embedding-model mismatches fail silently.
Q13 (Medium — new). Interviewer asks: "Why not just always use HNSW, since it has the best recall/speed and handles updates well?" How do you respond?

A: Push back on "always" — HNSW's weakness is memory: it stores full-precision vectors plus graph edges, so at billion-scale (multi-TB raw) it's often not the memory-optimal choice compared to IVF-PQ or ScaNN. It also has expensive point deletions (graph repair), unlike simple tombstoning elsewhere. The right framing: HNSW wins when memory budget allows full-precision storage and updates are frequent; IVF-PQ/ScaNN win when memory is the binding constraint and update cadence can tolerate batch rebuilds. Naming the actual trade-off (memory vs. update latency) rather than declaring a universal winner is the answer an interviewer is fishing for.

🧠 Gotchas — Full Recap
❌ Treating PQ as competing with HNSW/IVF instead of an orthogonal compression layer commonly combined with either.
❌ Picking an index "by which algorithm is best" instead of from requirements: scale, update frequency, latency budget, memory budget, filtering needs.
❌ Calling metadata filtering a free "WHERE clause" — how it interacts with the ANN structure is real, evolving engineering.
❌ Forgetting IVF's real weakness isn't speed — it's silent centroid staleness as new data arrives.
❌ Refusing to estimate on a capacity-planning question — a clearly-labeled rough estimate beats no answer.
❌ Assuming replication is only about fault tolerance — it's equally (often primarily) about QPS/read throughput.
❌ Assuming you can incrementally migrate to a new embedding model — impossible without a full re-embed + rebuild.
❌ Assuming FAISS is a database — it's a library; persistence, filtering, sharding, and deletion are all on you.
❌ Assuming HNSW node deletion is cheap — it requires graph repair; use tombstone + periodic compaction instead.
❌ Reaching for OpenSearch/Elasticsearch purely as "a vector DB" without naming hybrid search as the actual reason it might win.
❌ New: Declaring one algorithm a universal "best" — every answer should end with the trade-off (memory vs. update latency vs. accuracy), not a single winner.
❌ New: Mentioning ScaNN as "just another PQ variant" without being able to explain the anisotropic/parallel-error insight — that's the actual differentiator.
📌 Cheat Sheet
Landscape: Flat → HNSW (graph, best updates, M/ef_construction/ef_search) → IVF (clusters, nlist/nprobe, stale-centroid risk) → PQ (orthogonal compression, ~32×–384×+, approximate distances) → ScaNN (anisotropic PQ, better recall-per-byte) → IVF-PQ (the common billion-scale combo).

Two independent axes: search-narrowing (HNSW/IVF) vs. storage-shrinking (PQ/ScaNN/float16) — mix and match.

Recall/latency dial: ef_search and nprobe both trade latency for recall, diminishing returns.

Products, one-line differentiators: FAISS (library/reference) · Pinecone (managed, filtering/multi-tenancy) · Weaviate (HNSW + hybrid, GraphQL) · Milvus (distributed-first) · Qdrant (filtered-HNSW) · pgvector (already-Postgres) · MongoDB Atlas Vector Search (already-Mongo) · OpenSearch k-NN (already-ELK, or hybrid lexical+semantic is a hard requirement) · Vertex AI Vector Search (Google-managed, built on ScaNN).

The universal decision rule: extend the database you already operate before adding a new specialized one — pgvector/MongoDB/OpenSearch are all instances of this; deviate only for greenfield builds or genuinely extreme scale/performance requirements.

Updates: HNSW = incremental insert, no rebuild. IVF-family = batch rebuild, silent staleness — monitor recall. Deletes = tombstone + periodic compaction, never in-place graph repair. Embedding-model swaps = always full re-embed + rebuild + blue-green cutover, never incremental.

Scaling: shard when memory/QPS exceeds one machine (random = simple+full fan-out, semantic = complex+partial fan-out); replicate for throughput and fault tolerance (eventual consistency, usually fine for RAG). Capacity planning = raw memory → overhead-adjusted memory → sharding decision → per-replica QPS → replica count.

Filtering: pre-filter beats post-filter for selective filters, but naive pre-filtering can gut the index at high cardinality — filter-aware graph traversal (Qdrant/Weaviate/OpenSearch) is the real production answer.

Google-specific: ScaNN = anisotropic quantization (penalizes ranking-relevant, parallel-direction error more) → powers Vertex AI Vector Search. Frame any Google-scale system design answer as: embeddings → ANN index → sharding → replication → re-ranking → update/freshness strategy → recall-drift monitoring.

End of Day 4 (Refined). Next up — Day 5: Metadata Filtering, Hybrid Storage & Multi-Tenancy.
