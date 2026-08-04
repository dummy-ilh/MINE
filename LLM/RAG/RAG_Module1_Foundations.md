# RAG Module 1 — Foundations
### Master Interview Prep Notes

---

## 🚀 Quick Summary

Before you can evaluate or optimize a RAG system (Module 7), you need to know what's actually inside it: **why RAG exists as an alternative to fine-tuning and long-context**, **how text becomes searchable vectors (embeddings)**, **how documents get split into retrievable units (chunking)**, and **how millions of those vectors get searched in milliseconds (vector databases & indexing)**. Get these five pieces solid and the rest of the curriculum — retrieval, generation, evaluation — all snaps onto a system you can actually picture.

**Think of it like building a library from scratch.** Before you can argue about which search algorithm finds books fastest (retrieval, Module 2), you need to decide: do you even need a library, or should the librarian just memorize everything (fine-tuning)? How do you translate a book's content into something searchable (embeddings)? Do you file whole books, or chapters, or paragraphs (chunking)? And how do you organize the shelves so a search doesn't require walking past every single book (indexing)? This module is the librarian's setup work — everything else depends on it being right.

---

## 🔑 Key Concepts (Glossary — skim first, reference later)

| Term | One-line definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — fetch relevant external evidence at query time, then generate an answer conditioned on it |
| **Fine-tuning** | Updating a model's internal weights on a custom dataset so knowledge becomes baked into the parameters |
| **Long-context** | Feeding an entire knowledge base directly into the prompt instead of retrieving a subset of it |
| **Embedding** | A dense numerical vector representation of text, where semantic similarity ≈ geometric closeness |
| **Cosine similarity** | A measure of the angle between two vectors — the standard way to compare embeddings |
| **Chunking** | Splitting long documents into smaller retrievable units before embedding and indexing |
| **Vector database** | A storage system optimized for fast similarity search over embeddings, not exact-match lookup |
| **ANN (Approximate Nearest Neighbor)** | Search algorithms that trade a small amount of accuracy for a large amount of speed at scale |
| **HNSW** | Hierarchical Navigable Small World — a graph-based ANN indexing algorithm |
| **IVF** | Inverted File Index — a clustering-based ANN indexing algorithm |
| **Metadata filtering** | Narrowing a vector search using structured fields (date, category, user ID) alongside similarity |

---

# PHASE 1 — Intuition & Mental Map

## 1.1 RAG vs. Fine-Tuning vs. Long-Context

**Analogy — three ways to make an employee good at their job.**

1. **Fine-tuning** is like sending the employee to a multi-week training course where the material gets absorbed into their actual knowledge/instincts. They come back changed — they don't need to look anything up, they just *know* it now.
2. **Long-context** is like handing the employee the entire company handbook every single time they answer a question, and trusting them to skim the right page under time pressure.
3. **RAG** is like giving the employee a smart assistant who, the moment a question comes in, quickly pulls out *just the relevant pages* of the handbook and hands them over — the employee never has to read the whole book, and the book can be updated overnight without retraining the employee at all.

```
                    ┌─────────────────────────────────────────┐
                    │         HOW DOES THE MODEL KNOW?          │
                    └─────────────────────────────────────────┘
                                      │
        ┌─────────────────┬──────────┴──────────┬─────────────────┐
        ▼                  ▼                     ▼
  FINE-TUNING         LONG-CONTEXT               RAG
  bake knowledge      stuff everything          fetch only what's
  into weights         into the prompt          relevant, at query time
```

### Comparison table (memorize this — it's asked almost every time)

| Dimension | Fine-tuning | Long-context | RAG |
|---|---|---|---|
| **Knowledge freshness** | Stale until you retrain (expensive, slow) | Fresh — just change what you paste in | Fresh — just update the index |
| **Cost per query** | Low (no extra tokens) | High (paying for huge context every call) | Moderate (retrieval + smaller context) |
| **Latency** | Fast (no retrieval step) | Slow (huge prompt to process) | Moderate (retrieval adds a hop, but context stays small) |
| **Best for** | Teaching a *style*, *format*, or *skill* (tone, output structure, domain jargon) | Small, static, occasionally-changing corpora that fit the context window | Large, frequently-changing, or per-user knowledge bases |
| **Attribution/citations** | Hard — can't point to a source, knowledge is baked in | Possible but you're citing from a giant blob | Natural — you know exactly which chunk was retrieved |
| **Hallucination risk** | Higher on facts outside training data | Lower if relevant info is in context, but "lost in the middle" risk | Lower, but only as good as retrieval (garbage in → garbage out) |
| **Data privacy/access control** | Hard to scope per-user (weights are global) | Hard to scope (whole context is one blob) | Natural — filter retrieval per user/permission at query time |

**When to use which — the interview-ready framing:**
- Use **fine-tuning** when the problem is "the model doesn't know *how* to respond" (wrong tone, wrong format, doesn't follow domain conventions) — this is a *behavior* problem, not a *knowledge* problem.
- Use **long-context** when your knowledge base is small and mostly static (e.g., a single 50-page PDF) and you don't need retrieval infrastructure at all.
- Use **RAG** when the problem is "the model doesn't know *what*" — facts, documents, or knowledge that's large, changes often, or needs to be scoped per user/permission level.
- **They're not mutually exclusive.** A common real-world (and interview-favorite) answer: fine-tune the model to be good at *using* retrieved context well (following citation format, refusing when evidence is missing), and use RAG to supply the actual facts. Fine-tuning teaches the skill, RAG supplies the knowledge.

> **Why This Matters callout:** If asked "why not just use a 1M-token context window and skip RAG entirely?" — the strong answer isn't "context windows aren't big enough" (they increasingly are). It's: (1) cost — you pay for those tokens on *every single query* even if only 2% of the context is relevant, (2) latency — processing a huge prompt is slow, (3) lost-in-the-middle — models still attend unevenly across very long contexts, so dumping everything in doesn't guarantee the model actually *uses* the relevant part, and (4) access control — RAG lets you filter *what's even eligible to be retrieved* per user, which a static long context can't do cleanly.

---

## 1.2 Embeddings — Turning Text into Searchable Geometry

**Analogy — a map of meaning.** Imagine every word, sentence, or document as a city on a map, where cities that mean similar things are placed close together and cities that mean different things are placed far apart. "Return policy" and "refund process" would be neighboring cities; "return policy" and "battery specifications" would be across the map from each other. An embedding model's entire job is to draw this map — converting text into coordinates (a vector) such that geometric distance mirrors semantic distance.

**Why this matters for RAG specifically:** Once text is turned into coordinates, "finding relevant documents for a query" becomes "finding the nearest cities to the query's location on the map" — a pure geometry problem that computers are extremely fast at, instead of a fuzzy language-understanding problem for every single comparison.

### When to use it / when NOT to use it
- ✅ Use embeddings when you need **semantic** matching — "cheap flights" should match "affordable airfare" even with zero shared words.
- ✅ Use embeddings when your corpus is large enough that keyword search alone produces too much noise or misses paraphrases.
- ❌ Don't rely on embeddings alone for exact-match needs — product SKUs, error codes, legal citation numbers. Embeddings are bad at exact string/token matching; that's what sparse retrieval (BM25, Module 2) is for.
- ❌ Don't assume a bigger/fancier embedding model always wins — domain-specific fine-tuned embeddings often beat generic large ones on specialized corpora (medical, legal, code).

---

# PHASE 2 — Math & Formulas

## Notation table

| Symbol | Meaning |
|---|---|
| `A`, `B` | Two embedding vectors being compared |
| `A · B` | Dot product of A and B |
| `‖A‖` | Magnitude (length) of vector A |
| `θ` | The angle between vectors A and B |
| `d` | The dimensionality of the embedding (e.g. 768, 1536) |

---

### Cosine Similarity

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

**Plain English:** "How aligned in *direction* are these two vectors, ignoring their length?" It measures the angle between two vectors, not the distance between their tips — two vectors pointing the same direction get a similarity of 1, even if one is much "longer" than the other.

**Term-by-term:**
- `A · B` (dot product) — multiply the vectors' corresponding coordinates together and sum them up. This single number captures both alignment *and* magnitude combined, which is why we can't stop here.
- `‖A‖ × ‖B‖` — the product of each vector's own length (magnitude). Dividing by this **normalizes away magnitude**, leaving purely the directional (angular) similarity.
- The whole expression is mathematically equal to `cos(θ)` — hence the name. It ranges from **-1** (pointing in exactly opposite directions) to **1** (pointing in exactly the same direction), with **0** meaning perpendicular/unrelated.

**What happens if each term changes:**
- If `A` and `B` point in the exact same direction but `A` is scaled to be 10x longer, cosine similarity is **unchanged** — this is the key property that makes it preferred over raw dot product or Euclidean distance for text embeddings, since embedding magnitude often reflects things like text length or model confidence rather than meaning, and we usually don't want that polluting the similarity score.
- As the angle `θ` between two vectors grows from 0° to 90°, cosine similarity shrinks smoothly from 1 to 0.

**Worked numerical example.** Suppose we have tiny (unrealistic, but easy-to-hand-compute) 3-dimensional embeddings for two sentences:

```
A = "How do I reset my Apple ID password?"  → [0.8, 0.6, 0.0]
B = "Steps to recover a forgotten Apple ID"  → [0.6, 0.8, 0.0]
C = "What is the battery capacity of AirPods?" → [0.0, 0.1, 0.9]
```

**Step 1 — compute A · B:**
```
A · B = (0.8 × 0.6) + (0.6 × 0.8) + (0.0 × 0.0)
      = 0.48 + 0.48 + 0.0
      = 0.96
```

**Step 2 — compute magnitudes:**
```
‖A‖ = √(0.8² + 0.6² + 0.0²) = √(0.64 + 0.36 + 0) = √1.0 = 1.0
‖B‖ = √(0.6² + 0.8² + 0.0²) = √(0.36 + 0.64 + 0) = √1.0 = 1.0
```

**Step 3 — compute cosine similarity:**
```
cosine_similarity(A, B) = 0.96 / (1.0 × 1.0) = 0.96
```
A and B (both about password reset) are highly similar — 0.96, close to the max of 1.0.

**Now compare A and C:**
```
A · C = (0.8 × 0.0) + (0.6 × 0.1) + (0.0 × 0.9) = 0 + 0.06 + 0 = 0.06
‖C‖  = √(0² + 0.1² + 0.9²) = √(0 + 0.01 + 0.81) = √0.82 ≈ 0.906

cosine_similarity(A, C) = 0.06 / (1.0 × 0.906) ≈ 0.066
```
A (password reset) and C (battery capacity) are almost unrelated — 0.066, close to 0.

**Why it matters in practice:** Cosine similarity is the default comparison function for nearly every vector database and RAG retrieval pipeline because it's magnitude-invariant, cheap to compute, and empirically correlates well with human judgments of semantic similarity for the embedding models in common use. When you set up a vector index, "similarity metric: cosine" is usually the very first configuration choice you make.

> **Gotcha:** Some embedding models (and some vector DBs) actually default to **dot product** instead of cosine — this is fine (and even faster, since it skips the normalization step) *if and only if* the embeddings are pre-normalized to unit length, in which case dot product and cosine similarity become mathematically identical. If you use raw dot product on non-normalized embeddings, you'll get results skewed by vector magnitude, which usually isn't what you want. Know this distinction — it's a common "gotcha" interview question.

### Cosine similarity vs. Euclidean distance — quick comparison

| | Cosine Similarity | Euclidean (L2) Distance |
|---|---|---|
| **Measures** | Angle between vectors | Straight-line distance between vector tips |
| **Sensitive to magnitude?** | No | Yes |
| **Typical use in RAG** | Default for text embeddings | Sometimes used for image embeddings or when magnitude carries meaningful signal |
| **Range** | -1 to 1 | 0 to ∞ |

---

## 1.3 Chunking Strategies

**Analogy — cutting a cake for a buffet.** If you serve the whole cake as one giant slab, nobody can grab a manageable piece (this is "no chunking" — indexing whole documents, which usually retrieves way more irrelevant content per hit than needed). If you cut it into crumbs, each piece is meaningless on its own and you need to grab twenty crumbs to get a coherent bite (chunks too small, losing context). The goal is slices — big enough to be a complete, meaningful unit, small enough that a person only needs one or two to get what they came for.

### Strategy comparison table

| Strategy | How it works | Pros | Cons | When to use |
|---|---|---|---|---|
| **Fixed-size** | Split every N tokens/characters, regardless of content boundaries | Simple, fast, predictable chunk sizes for indexing/cost | Can split mid-sentence or mid-idea, hurting coherence | Quick prototypes, homogeneous unstructured text |
| **Sliding window (with overlap)** | Fixed-size chunks, but consecutive chunks share an overlapping region | Reduces the "important sentence split across a chunk boundary" problem | More storage/compute (redundant content indexed multiple times) | Most production systems as a baseline — good default |
| **Recursive** | Try splitting on natural boundaries first (paragraphs → sentences → words), falling back to smaller units only if a piece is still too big | Respects natural document structure much better than fixed-size | Slightly more complex to implement, variable chunk sizes | General-purpose production default (e.g. LangChain's `RecursiveCharacterTextSplitter`) |
| **Semantic chunking** | Use embeddings to detect where topic/meaning shifts, and split at those boundaries rather than a fixed size | Chunks are topically coherent, not arbitrarily cut | More expensive (requires embedding calls just to decide split points), harder to reason about resulting chunk sizes | High-value corpora where coherence matters a lot (legal, medical) and cost of extra embedding calls is acceptable |

### Worked numerical example — sliding window overlap math

Suppose a document is **1000 tokens** long, and you choose a **chunk size of 200 tokens** with a **50-token overlap** between consecutive chunks.

**Step 1 — compute the effective stride** (how far forward each new chunk starts):
```
stride = chunk_size - overlap = 200 - 50 = 150 tokens
```

**Step 2 — compute number of chunks:**
```
num_chunks ≈ ⌈(document_length - overlap) / stride⌉
           = ⌈(1000 - 50) / 150⌉
           = ⌈950 / 150⌉
           = ⌈6.33⌉
           = 7 chunks
```

**Step 3 — total tokens actually indexed (including redundancy from overlap):**
```
total_indexed_tokens ≈ num_chunks × chunk_size = 7 × 200 = 1400 tokens
```
That's **1400 tokens indexed from a 1000-token document** — a 40% storage/embedding-cost overhead purely from the overlap. This is the direct, quantifiable trade-off of using overlap: you pay ~40% more in embedding calls and storage in exchange for reducing the chance that an important sentence gets split awkwardly across a chunk boundary and loses coherence in both halves.

**Why it matters in practice:** Chunk size and overlap are two of the highest-leverage hyperparameters in the entire RAG pipeline (this is exactly what Module 2.8's chunk-size sweep, referenced in your evaluation notes, is tuning). Too large → chunks contain multiple unrelated ideas, diluting embedding quality and retrieval precision (a query about one idea in the chunk pulls back irrelevant text about another idea in the same chunk). Too small → chunks lose necessary context, and answers requiring even a little synthesis become impossible without multi-hop retrieval.

> **Business example:** Indexing Apple's support documentation, a fixed-size 100-token chunk might cut a troubleshooting step in half between "Step 3: Hold the button for 10 seconds" and "...until the light flashes amber" — retrieved independently, Step 3 alone is nearly useless. A recursive or semantic strategy would keep that instruction intact as one chunk.

### Gotchas
- ❌ Chunking purely by character/token count with zero overlap is the most common beginner mistake — it *will* eventually split a critical sentence in half, and you won't notice until a specific query mysteriously fails.
- ❌ Don't assume smaller chunks are always safer — very small chunks lose surrounding context that the embedding needs to represent meaning accurately (a 10-token chunk saying "14 days" is meaningless without knowing it's about a return window).
- ❌ Forgetting to store chunk metadata (source document, position, surrounding chunk IDs) at chunking time — you'll want this later for citation, re-assembly, and debugging, and it's expensive to reconstruct after the fact.

---

## 1.4 Vector Databases & Indexing

**Analogy — finding a friend's house in a city with no addresses.** If the city had a million houses and no organizing system, "find the house most similar to this description" means walking past every single house (this is **brute-force / exact nearest neighbor search** — perfectly accurate, but painfully slow at scale). A vector database's indexing algorithm is like organizing the city into neighborhoods-of-neighborhoods so you can jump straight to the right area and only check a handful of houses — trading a tiny bit of accuracy for a massive speedup. This is **Approximate Nearest Neighbor (ANN) search**.

### Exact vs. Approximate Nearest Neighbor

| | Exact NN (brute-force) | Approximate NN (ANN) |
|---|---|---|
| **Accuracy** | Perfect — guaranteed to find the true nearest neighbors | Approximate — might miss the true #1 result occasionally, in exchange for speed |
| **Speed at scale** | Linear in the number of vectors — becomes unusable past ~100K–1M vectors | Sub-linear — scales to billions of vectors |
| **When to use** | Small datasets (thousands of vectors), or when you need a ground-truth baseline to evaluate ANN accuracy against | Virtually all production RAG systems at any real scale |

### HNSW (Hierarchical Navigable Small World)

**Plain English:** HNSW builds a **multi-layer graph** where each vector is a node, connected to a handful of its nearest neighbors. The top layer is sparse (few nodes, long-range connections — like highways), and each layer down gets denser (more nodes, short-range connections — like local streets). A search starts at the sparse top layer, quickly narrows down to the right general neighborhood via long jumps, then descends layer by layer, taking smaller and smaller steps to pinpoint the actual nearest neighbors.

**Analogy:** Think of it like a flight route — you don't drive city-to-city, you fly into a regional hub (top layer, coarse), then take a connecting flight to a smaller airport (middle layer), then drive the last few miles (bottom layer, fine-grained). HNSW search does the geometric equivalent: coarse-to-fine navigation instead of exhaustive search.

**Key hyperparameters and what happens when they change:**
- **`M`** (number of connections per node) — higher M → better recall/accuracy, but more memory and slower index build time. Lower M → faster, smaller index, but worse accuracy.
- **`ef_construction`** (search effort during index building) — higher → better quality graph (more accurate neighbor connections), but much slower to build the index.
- **`ef_search`** (search effort at query time) — higher → better recall at query time, but slower per-query latency. This is the main knob you tune *after* the index is already built, since it doesn't require rebuilding.

**When to use:** HNSW is the most common choice for production RAG vector databases (Pinecone, Weaviate, Milvus, FAISS all support it) because it gives excellent recall at low query latency and updates reasonably well when you add new vectors incrementally — a big deal for RAG corpora that change over time.

### IVF (Inverted File Index)

**Plain English:** IVF first **clusters** all vectors into a fixed number of groups (using something like k-means), then at query time, only searches within the clusters closest to the query vector — instead of comparing against every single vector in the database.

**Analogy:** Like organizing a library by genre section first. If you're looking for a mystery novel, you don't scan every shelf in the building — you go straight to the mystery section (the cluster nearest your query) and only search within it.

**Key hyperparameters:**
- **`nlist`** (number of clusters) — more clusters → finer-grained partitioning, faster individual cluster search, but risk of the true nearest neighbor being in a *different* cluster than the one searched (hurting recall) if clustering isn't clean.
- **`nprobe`** (number of clusters searched per query) — higher `nprobe` → search more clusters → better recall, but slower (this is IVF's main speed/accuracy dial, directly analogous to HNSW's `ef_search`).

**When to use:** IVF is often faster to build on very large static datasets and can be more memory-efficient than HNSW, but handles incremental updates (new documents arriving continuously) less gracefully than HNSW, since re-clustering may eventually be needed as data distribution shifts.

### HNSW vs. IVF — comparison table (frequently asked directly)

| | HNSW | IVF |
|---|---|---|
| **Underlying structure** | Multi-layer navigable graph | Cluster-based partitioning (inverted lists) |
| **Query speed** | Very fast, consistently | Fast, depends on `nprobe` |
| **Memory usage** | Higher (graph edges take space) | Generally lower |
| **Build time** | Slower to build | Faster to build |
| **Handles incremental updates well?** | Yes — a common reason it's preferred for RAG, where documents get added continuously | Less naturally — may need periodic re-clustering |
| **Main tuning knob at query time** | `ef_search` | `nprobe` |

> **Why This Matters callout:** In an Apple MLE interview, "which indexing algorithm would you pick" isn't really the question — the real question is whether you understand the **speed/accuracy/memory trade-off triangle** and can reason about it given constraints (e.g., "we need sub-50ms query latency and the corpus updates every hour" → HNSW, because IVF's re-clustering story is worse for frequently-updated data; "we need to index 2 billion static vectors on a memory budget" → IVF, because it's typically more memory-efficient).

---

## 1.5 Metadata Filtering, Hybrid Storage, and Multi-Tenancy

**Analogy — a library with locked sections.** Pure vector search finds the books most similar to your query, but sometimes you also need hard constraints: "only books published after 2023," "only books in the medical wing," "only books this specific patron is allowed to check out." Metadata filtering is how you combine *fuzzy* similarity search with *hard* structured constraints in the same query.

### Key ideas

- **Metadata filtering:** Every vector can be stored alongside structured fields (date, category, author, permission tags, language). A query becomes: "find the nearest vectors *among only those matching filter X*" — e.g., `category = "return_policy" AND region = "US"`.
- **Pre-filtering vs. post-filtering:**
  - **Pre-filtering** narrows the candidate set by metadata *before* running the similarity search — more efficient when the filter is highly selective (few matching documents), but can hurt recall if the ANN index isn't filter-aware (some implementations degrade badly when the filter shrinks the eligible set far below what the index structure expects).
  - **Post-filtering** runs similarity search first, then discards results that don't match the metadata filter — simpler, but risks retrieving *fewer than k* useful results if too many top-ranked candidates get filtered out afterward (you asked for top-10, but 8 of them get discarded by the filter, leaving you with only 2).
- **Hybrid storage:** Combining a vector index with a traditional structured database (or filterable index) so a single query can jointly leverage semantic similarity *and* exact-match/range constraints without two separate round trips.
- **Multi-tenancy:** In systems serving multiple users/customers/orgs from one shared vector database, tenant isolation is usually done via metadata filtering (`tenant_id = X`) or fully separate namespaces/indexes per tenant — the trade-off being shared-index-with-filter (cheaper, simpler ops) vs. per-tenant-index (stronger isolation, more predictable performance, but more infrastructure to manage at scale).

> **Business example:** An enterprise support-RAG product serving multiple companies needs strict data isolation — Company A's documents must never leak into Company B's retrieved context. The interview-strong answer distinguishes *how* you'd enforce this (metadata filter on `tenant_id` vs. separate indexes) and the trade-off (shared index = cheaper and easier to maintain, but a filtering bug is a serious data leak; separate indexes = safer by construction, but multiplies infrastructure and ops overhead linearly with tenant count).

> **Gotcha:** Don't say "just filter by metadata" as if it's a free operation. If asked a follow-up about performance, mention that pre-filtering interacts with the ANN index structure (some ANN indexes handle filters natively and efficiently — filtered HNSW variants exist — while others essentially fall back toward brute-force search when filters are highly restrictive) — this is a real, current, actively-discussed engineering problem in vector database design, not a solved non-issue.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** Your team is deciding between fine-tuning and RAG for a customer support bot whose product catalog changes weekly. Which would you recommend, and why?

<details>
<summary>Show answer</summary>

RAG. The core signal is *knowledge that changes frequently* — fine-tuning bakes knowledge into weights, so a weekly-changing catalog would require weekly retraining, which is slow and expensive. RAG only requires updating the vector index (re-embedding and inserting new/changed documents), which is far cheaper and can happen continuously. Fine-tuning would still be worth layering on top if you also wanted the model to follow a specific tone or output format, but the knowledge freshness problem specifically points to RAG.
</details>

---

**Q2 (Easy — calculation).** Two embedding vectors are `A = [1, 0]` and `B = [0, 1]`. Compute their cosine similarity and interpret it.

<details>
<summary>Show answer</summary>

```
A · B = (1×0) + (0×1) = 0
‖A‖ = √(1²+0²) = 1
‖B‖ = √(0²+1²) = 1

cosine_similarity = 0 / (1×1) = 0
```
A similarity of 0 means the vectors are perpendicular (90° apart) — completely unrelated in direction, which for text embeddings typically means semantically unrelated content.
</details>

---

**Q3 (Medium — conceptual).** Why is cosine similarity generally preferred over raw dot product for comparing text embeddings?

<details>
<summary>Show answer</summary>

Raw dot product is sensitive to vector magnitude, not just direction — two vectors could point in a very similar direction but produce a small dot product simply because one has small magnitude, or two dissimilar-direction vectors could produce a large dot product because both have large magnitude. Cosine similarity normalizes by dividing by both vectors' magnitudes, isolating pure directional (angular) similarity, which correlates better with semantic similarity for typical text embedding models. Note: if embeddings are pre-normalized to unit length, dot product and cosine similarity become mathematically identical — some systems exploit this to skip the normalization step at query time for speed.
</details>

---

**Q4 (Medium — calculation).** A 2000-token document is split with chunk size 300 and overlap 100. Compute the stride and approximate number of chunks.

<details>
<summary>Show answer</summary>

```
stride = 300 - 100 = 200
num_chunks ≈ ⌈(2000 - 100) / 200⌉ = ⌈1900/200⌉ = ⌈9.5⌉ = 10 chunks
```
</details>

---

**Q5 (Medium — conceptual).** What's the practical difference between fixed-size chunking and recursive chunking, and when would you pick one over the other?

<details>
<summary>Show answer</summary>

Fixed-size chunking splits at a strict token/character count regardless of content structure, which is simple and predictable but frequently cuts sentences or ideas in half. Recursive chunking tries to split at natural boundaries first (paragraphs, then sentences, then words only if necessary), producing more coherent chunks at the cost of variable chunk sizes and slightly more implementation complexity. Pick fixed-size for quick prototyping or highly homogeneous unstructured text where structure doesn't carry much meaning; pick recursive (or semantic, for even higher stakes) as the production default for most real document corpora, since coherent chunks directly improve both embedding quality and downstream answer quality.
</details>

---

**Q6 (Hard — conceptual + trade-off).** You're designing a vector index for a RAG system with 500 million vectors that gets ~10,000 new documents added every hour, and needs sub-100ms query latency. Would you lean toward HNSW or IVF, and what's the reasoning?

<details>
<summary>Show answer</summary>

HNSW. The two dominant constraints here are (1) frequent incremental updates and (2) tight query latency. HNSW handles incremental insertion more gracefully than IVF, since new nodes can be added into the existing graph structure without necessarily requiring a full re-clustering step — IVF's cluster boundaries can degrade as data distribution shifts over time with continuous inserts, eventually requiring re-clustering, which is disruptive at 500M-vector scale. HNSW also delivers consistently fast, tunable query latency via the `ef_search` parameter without rebuilding the index. IVF might still be considered if memory budget were the dominant constraint instead (IVF is generally more memory-efficient), but the update-frequency signal in this scenario points toward HNSW.
</details>

---

**Q7 (Hard — "spot the bug" scenario).** A team filters vector search results by `region = "EU"` *after* retrieving the top-10 nearest neighbors, and complains that sometimes only 2-3 results come back even though there are plenty of EU documents in the index. What's happening, and how would you fix it?

<details>
<summary>Show answer</summary>

This is a **post-filtering** problem: the system retrieves the globally top-10 nearest neighbors first (ignoring region), and only *afterward* discards any that aren't tagged EU. If most of the true top-10 nearest neighbors happen to be non-EU documents, filtering post-hoc can leave very few results even though relevant EU documents exist further down the ranked list. The fix is to move to **pre-filtering** (restrict the candidate pool to `region = "EU"` before or during the similarity search, e.g., using a filter-aware ANN index, or filtering within a metadata-partitioned index/namespace) so the top-k similarity search is actually performed within the already-narrowed EU-only candidate set, guaranteeing up to k EU results if that many exist.
</details>

---

**Q8 (Hard — synthesis across the module).** Explain, in a way that connects chunking, embeddings, and indexing together, why a "chunk size too large" mistake is hard to detect just by looking at Recall@k in isolation (tie back to Module 7 if helpful).

<details>
<summary>Show answer</summary>

If a chunk is too large, it likely contains multiple distinct ideas bundled together. Because embeddings compress an entire chunk into a single vector, a large multi-topic chunk's embedding becomes a blurred average of all the ideas inside it — it may still land "close enough" to a query about *any one* of those ideas to be retrieved as a top-k result, especially with a generous k. This means Recall@k can look deceptively fine (the relevant chunk technically *was* retrieved) while context relevance and faithfulness (Module 7) suffer, because the retrieved chunk is diluted with irrelevant surrounding content that confuses the generator or gets ignored/misattributed. In other words, chunk-size problems often show up downstream as generation-stage symptoms (low context relevance, more hallucination) rather than as an obvious retrieval-stage Recall@k drop — which is exactly why Module 7 emphasizes using the full metric triad together rather than trusting any single number in isolation.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Treating RAG, fine-tuning, and long-context as mutually exclusive rather than complementary (fine-tuning for skill/format, RAG for knowledge, and they're often combined).
- ❌ Using raw dot product on non-normalized embeddings and being surprised results skew toward longer/higher-magnitude vectors.
- ❌ Chunking with zero overlap and no respect for natural document boundaries — the single most common beginner mistake in RAG pipelines.
- ❌ Assuming smaller chunks are always safer — too-small chunks lose the context needed for the embedding to represent meaning accurately.
- ❌ Picking an ANN algorithm by familiarity rather than by reasoning through the speed/accuracy/memory/update-frequency trade-off triangle.
- ❌ Describing metadata filtering as a "free" operation with no performance implications — pre- vs. post-filtering has real recall and latency consequences.
- ❌ Forgetting that a chunk-size problem often surfaces as a *generation*-stage symptom (low faithfulness/context relevance), not an obvious retrieval-stage Recall@k drop.

---

# 📌 One-Page Cheat Sheet

**RAG vs. fine-tuning vs. long-context:** Fine-tuning = teach a skill/format (bakes into weights, stale, expensive to update). Long-context = stuff everything in the prompt (fresh but costly + lost-in-the-middle risk). RAG = fetch just what's relevant at query time (fresh, cheaper per-query, natural citations/access control). Often combined, not either/or.

**Embeddings:** Text → dense vector where geometric closeness ≈ semantic closeness. Cosine similarity = `(A·B)/(‖A‖‖B‖)`, magnitude-invariant, range -1 to 1. Dot product ≈ cosine similarity only if vectors are pre-normalized.

**Chunking:** Fixed-size (simple, cuts sentences) → sliding window w/ overlap (reduces boundary loss, costs storage) → recursive (respects structure, production default) → semantic (topically coherent, most expensive). Overlap trade-off is quantifiable: `stride = chunk_size - overlap`, more overlap = more redundant storage.

**Indexing:** Exact NN = perfect but doesn't scale. ANN trades tiny accuracy loss for massive speed. HNSW = multi-layer graph, great for frequent updates + low latency, more memory (`ef_search`/`ef_construction`/`M` are the knobs). IVF = clustering-based, faster to build, more memory-efficient, worse at handling frequent incremental updates (`nprobe`/`nlist` are the knobs).

**Metadata filtering:** Pre-filter (narrow candidates before similarity search — better recall, needs filter-aware index) vs. post-filter (search first, discard after — simpler, risks returning fewer than k results). Multi-tenancy = shared index + filter (cheap, filter-bug risk) vs. per-tenant index (safer, more infra).

---

*End of Module 1. Next up: Module 2 — Retrieval (sparse, dense, hybrid search, reranking, query transformation).*
