# Chapter 4 — Dense Retrieval (Embeddings)

## What is it?

Dense retrieval is a fundamentally different approach to `score(q, d)`. Instead of matching exact words (like TF-IDF and BM25), it converts both the query and every document into a **dense vector** — a list of hundreds of floating point numbers — and measures relevance as **geometric closeness** between those vectors in high-dimensional space.

The word "dense" contrasts with "sparse." A TF-IDF vector for a document has one dimension per vocabulary word — typically 100,000+ dimensions — and almost all of them are zero (the document only contains a tiny fraction of all possible words). A dense vector has 768 or 1536 dimensions and **every single dimension has a non-zero value**. That density is what allows it to encode meaning, not just word presence.

This is what powers Google Search's semantic understanding, OpenAI's embedding APIs, and every RAG pipeline you'll encounter in FAANG interviews.

---

## The intuition

TF-IDF and BM25 are blind to meaning. They see strings, not concepts. "Car" and "automobile" are completely different tokens — no overlap, no connection.

Now imagine a different approach. You train a neural network on billions of sentences. It learns that "car" and "automobile" always appear in similar contexts — similar sentences, similar surrounding words. So it maps them to **nearby points in vector space**. Not identical points — they are still different words — but close enough that a nearest-neighbor search for one retrieves the other.

This is the core insight: **meaning ≈ context ≈ proximity in vector space.**

A query like "what causes high blood sugar?" and a document containing "diabetes is characterized by hyperglycemia" will have very similar vectors even though they share zero words. That's what dense retrieval buys you.

---

## The equations

### Cosine similarity

```
cosine_sim(q, d) = (q · d) / (|q| × |d|)
```

Where:
- `q · d` = dot product = Σ qᵢ × dᵢ (multiply element-wise, sum up)
- `|q|` = magnitude of q = √(Σ qᵢ²)
- `|d|` = magnitude of d = √(Σ dᵢ²)

Range is [-1, 1]:
- `1`  = vectors point in exactly the same direction → maximally similar
- `0`  = vectors are perpendicular → no similarity
- `-1` = vectors point in opposite directions → maximally dissimilar

**Why direction and not distance?** Two documents about "cats" — one short, one long — will have vectors pointing in the same direction but at different magnitudes. Euclidean distance would make them look different. Cosine similarity ignores magnitude and only looks at direction, so both documents score the same. Direction encodes topic; magnitude encodes something closer to document length.

---

### Dot product (when vectors are normalized)

Most embedding models (BERT, OpenAI ada, sentence-transformers) output **unit vectors** — vectors with magnitude = 1. For unit vectors:

```
cosine_sim(q, d) = q · d = Σ qᵢ × dᵢ
```

The denominator becomes 1 × 1 = 1 and disappears. This matters enormously in practice because **dot product is much faster** — no square roots, no division. FAISS, ScaNN, and HNSW all exploit this.

---

### Euclidean distance (L2)

```
L2(q, d) = √( Σ (qᵢ - dᵢ)² )
```

Smaller = more similar. For unit-normalized vectors, L2 and cosine similarity are mathematically equivalent — one is a monotonic transformation of the other — so the choice doesn't matter when vectors are normalized.

---

## The bi-encoder architecture

```
query    →  [BERT encoder]  →  q_vector (768-dim)
document →  [BERT encoder]  →  d_vector (768-dim)

score(q, d) = cosine_sim(q_vector, d_vector)
```

The key insight: **query and document are encoded independently.** This means you pre-compute and store all document vectors offline. At query time, encode only the query (one forward pass) then do a nearest-neighbor search over pre-computed document vectors.

---

## Worked numeric example — bi-encoder scoring

Using 2D vectors so the arithmetic is visible. Real embeddings are 768D but the math is identical.

```
embeddings (2D, unit-normalized):
  query q  = [0.90, 0.10]   → "dog"
  doc D1   = [0.85, 0.15]   → "canine companion"
  doc D2   = [0.10, 0.90]   → "feline friend"
  doc D3   = [0.60, 0.40]   → "animal behaviour"
```

### Step 1 — dot product scores (= cosine sim for unit vectors)

```
score(q, D1) = (0.90 × 0.85) + (0.10 × 0.15) = 0.765 + 0.015 = 0.780
score(q, D2) = (0.90 × 0.10) + (0.10 × 0.90) = 0.090 + 0.090 = 0.180
score(q, D3) = (0.90 × 0.60) + (0.10 × 0.40) = 0.540 + 0.040 = 0.580
```

### Step 2 — rank

```
D1 (0.780) > D3 (0.580) > D2 (0.180)
```

D1 "canine companion" ranks first despite sharing zero words with "dog." The model has learned that "dog" and "canine" are semantically close — nearby vectors. D2 "feline friend" ranks last. BM25 would score all three at zero (no word overlap with "dog"). That's the fundamental win of dense retrieval.

---

## The cross-encoder — worked numeric example

A cross-encoder sees query and document **together**, not independently. It produces a single relevance score by attending across both simultaneously.

```
architecture:
  input:  [CLS] query tokens [SEP] document tokens [SEP]
  model:  BERT (or similar transformer)
  output: single scalar relevance score from [CLS] token
```

### Why this is more accurate

In a bi-encoder, "dog" gets encoded without knowing what document it will be compared to. In a cross-encoder, the model sees "dog" and "canine companion" at the same time — it can attend from "dog" directly to "canine" and learn their interaction. This cross-attention is what makes it more powerful.

### Numeric example

```
query: "dog training tips"

candidate documents (from bi-encoder first stage):
  D1 = "how to train your canine companion effectively"
  D2 = "dogs make great household pets"
  D3 = "tips for training puppies and adult dogs"

cross-encoder input and scores:
  [CLS] dog training tips [SEP] how to train your canine companion [SEP]  → 0.91
  [CLS] dog training tips [SEP] dogs make great household pets [SEP]      → 0.43
  [CLS] dog training tips [SEP] tips for training puppies and adult dogs  → 0.87
```

### Re-ranking result

```
bi-encoder order:   D1 (0.81) > D2 (0.74) > D3 (0.69)   ← couldn't see full interactions
cross-encoder order: D1 (0.91) > D3 (0.87) > D2 (0.43)  ← D3 jumps, D2 drops correctly
```

D2 "dogs make great pets" was rated fairly high by the bi-encoder (it shares "dog" context) but the cross-encoder correctly demotes it — it's not about training at all. D3 jumps because the cross-encoder can see that "tips for training" directly matches the query intent.

### Why you can't use cross-encoders alone

```
corpus: 10,000,000 documents
cross-encoder inference: ~20ms per (query, doc) pair on GPU

total latency if used alone: 10,000,000 × 20ms = 55 hours per query
total latency with bi-encoder first stage:
  bi-encoder ANN search → top 1,000 candidates → 50ms
  cross-encoder re-rank 1,000 docs → 1,000 × 20ms = 20 seconds  ← still too slow

realistic pipeline:
  bi-encoder → top 100 → cross-encoder re-rank → return top 10
  latency: 50ms + 2,000ms = ~2 seconds  ← acceptable
```

This two-stage architecture — **bi-encoder for recall, cross-encoder for precision** — is the standard production pattern at Google, Meta, and Microsoft.

---

## ANN search — why exact search doesn't scale

```
corpus:       1,000,000 documents
vector dim:   768
exact search: compute dot product with every doc → 1,000,000 × 768 multiplications per query

on modern CPU: ~500ms per query  ← too slow for real-time search
```

ANN (Approximate Nearest Neighbor) trades a small recall loss for massive speed gains. Three main approaches:

---

## HNSW — worked numeric example

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where each node is a document vector.

```
structure (simplified to 8 docs, 3 layers):

layer 2 (sparse, long-range):    D1 ←→ D5
layer 1 (medium):                D1 ←→ D3 ←→ D5 ←→ D7
layer 0 (dense, all nodes):      D1-D2-D3-D4-D5-D6-D7-D8 (fully connected locally)
```

### Search walkthrough — query q

```
step 1: enter at layer 2, current node = D1
        compute sim(q, D1) = 0.61
        compute sim(q, D5) = 0.74   ← D5 is closer, move to D5

step 2: drop to layer 1 at D5
        neighbours of D5: D3 (0.69), D7 (0.82)
        D7 is closest → move to D7

step 3: drop to layer 0 at D7
        neighbours of D7: D6 (0.79), D8 (0.91)  ← D8 is closest
        check D8 neighbours: D4 (0.85)
        no improvement from D4 → stop

result: D8 (0.91) → top candidate returned
```

### Why it's fast

```
exact search: checked all 8 nodes
HNSW:         checked 5 nodes (D1, D5, D3, D7, D8)

at 1,000,000 docs:
exact search: O(N)      → ~500ms
HNSW:         O(log N)  → ~1-5ms
recall@10:    ~95-99%   (small fraction of true nearest neighbors missed)
```

The long-range connections at upper layers let you skip huge portions of the graph in early steps. Lower layers zoom in for precision. This mimics how you'd navigate a map — continent first, then country, then city.

**HNSW tradeoffs:**
```
pros: very high recall, fast queries, good for static corpora
cons: large RAM footprint (graph stored in memory), slow index build,
      adding new nodes requires graph rewiring → best for mostly-static corpora
```

---

## FAISS IVF — worked numeric example

FAISS IVF (Inverted File Index) clusters all document vectors first, then only searches the relevant clusters at query time.

### Index build phase

```
corpus: 12 document vectors (2D for visibility)
step 1: run k-means with K=3 centroids

centroids after k-means:
  C1 = [0.85, 0.15]   ← cluster of "animal/dog" docs
  C2 = [0.10, 0.88]   ← cluster of "cat/feline" docs
  C3 = [0.50, 0.50]   ← cluster of "general animal" docs

assignments:
  C1 → {D1, D2, D4}
  C2 → {D5, D6, D8}
  C3 → {D3, D7, D9, D10, D11, D12}
```

### Query phase

```
query q = [0.88, 0.12]   → "dog"

step 1: find nearest centroids to q
  sim(q, C1) = (0.88×0.85) + (0.12×0.15) = 0.748 + 0.018 = 0.766  ← closest
  sim(q, C2) = (0.88×0.10) + (0.12×0.88) = 0.088 + 0.106 = 0.194
  sim(q, C3) = (0.88×0.50) + (0.12×0.50) = 0.440 + 0.060 = 0.500

step 2: search top nprobe=1 cluster → only search C1 = {D1, D2, D4}
  sim(q, D1) = 0.780
  sim(q, D2) = 0.751
  sim(q, D4) = 0.698

step 3: return top-k from searched clusters
  result: D1 > D2 > D4
```

### The nprobe tradeoff

```
nprobe = 1 → search 1 cluster  → fastest, lowest recall
             risk: true nearest neighbor might be in C3, missed entirely

nprobe = 2 → search C1 + C3   → slower, higher recall
nprobe = 3 → search all clusters → same as exact search, slowest

typical setting: nprobe = 8-64 out of K=1000 clusters
recall@10 at nprobe=32: ~90-95%
speedup vs exact:       ~20-50×
```

**FAISS IVF tradeoffs:**
```
pros: much smaller memory than HNSW (no graph structure),
      easy to add new vectors to existing clusters,
      nprobe is a runtime tradeoff (tune without rebuilding)
cons: lower recall than HNSW at same speed budget,
      cluster quality depends on k-means initialization,
      needs periodic retraining if data distribution shifts
```

---

## HNSW vs FAISS IVF — head to head

| Property | HNSW | FAISS IVF |
|----------|------|-----------|
| Query speed | Very fast (log N graph traversal) | Fast (search subset of clusters) |
| Recall@10 | 95–99% | 85–95% (tunable via nprobe) |
| Memory | High (graph in RAM) | Lower (cluster lists only) |
| Index build | Slow | Faster |
| Dynamic updates | Hard (graph rewiring) | Easier (append to cluster) |
| Runtime tuning | ef_search parameter | nprobe parameter |
| Best for | Static corpora, max recall | Large corpora, memory-constrained |

---

## Why it works / why it fails

**Why it works:**
- Handles synonyms, paraphrases, and cross-lingual queries naturally
- Learns from data — embedding model captures what "similar" actually means in your domain
- A single vector encodes full document semantics, not just keyword presence
- ANN makes billion-document search practical at millisecond latency

**Why it fails:**
- **Exact match weakness** — product IDs, rare proper nouns, error codes. Embedding space may retrieve semantically similar but factually wrong results
- **Needs labeled data** — embedding model must be fine-tuned on your domain. Generic embeddings underperform on medical, legal, or technical text
- **Cold start** — new documents must be encoded and inserted into ANN index. More complex operationally than appending to an inverted index
- **Black box** — unlike BM25, hard to explain why two vectors are close. Matters for auditing and compliance
- **Infrastructure cost** — billions of 768-dim float32 vectors require terabytes of RAM for HNSW

---

## The one thing to remember

Dense retrieval converts matching from a **word overlap problem** into a **geometry problem** — relevant documents are nearby vectors in embedding space, found without sharing a single word with the query.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `cosine_sim(q,d) = (q·d) / (\|q\|×\|d\|)` | Similarity as angle between vectors. Range [-1, 1] |
| `q·d = Σ qᵢ×dᵢ` | Dot product — fast similarity for unit-normalized vectors |
| `\|v\| = √(Σ vᵢ²)` | Vector magnitude (L2 norm) |
| `L2(q,d) = √(Σ(qᵢ-dᵢ)²)` | Euclidean distance — equivalent to cosine for unit vectors |
| `score(q,d) = q·d` | Final retrieval score when embeddings are unit-normalized |
| `HNSW complexity = O(log N)` | Query time scales logarithmically with corpus size |
| `IVF speedup = N / (nprobe × (N/K))` | Docs searched = nprobe clusters × avg cluster size |

---

## Interview Q&A

**Q1. What is the difference between a bi-encoder and a cross-encoder? When do you use each?**

A bi-encoder encodes query and document independently into separate vectors scored with a dot product. Document vectors are pre-computed, so retrieval is fast — one query encode then ANN search. A cross-encoder concatenates query and document and passes them through a model together, producing a single relevance score via cross-attention. It's far more accurate because it sees term interactions between query and document, but it cannot pre-compute anything. The standard production pipeline is: bi-encoder for first-stage retrieval (top-100 candidates fast), cross-encoder for second-stage re-ranking (top-100 → top-10 accurately). Used in DPR, Google's dual-encoder, and every serious production search system.

**Q2. Why do we prefer dot product over cosine similarity in production?**

For unit-normalized vectors they are mathematically identical — cosine similarity with |q|=|d|=1 reduces to the dot product. Dot product requires N multiplications and N additions. Cosine similarity requires all that plus two square root computations and a division. At billions of comparisons, eliminating square roots and division is a meaningful latency saving. More importantly, ANN libraries like FAISS and ScaNN are heavily optimized for dot product using SIMD instructions. The standard practice is to L2-normalize all vectors at indexing time and use dot product at query time.

**Q3. What is the curse of dimensionality and how does it affect dense retrieval?**

In high-dimensional spaces, distances between random points converge — the nearest neighbor is barely closer than the farthest neighbor. This makes ANN search harder because the signal-to-noise ratio in distance comparisons shrinks. HNSW graph traversal and IVF cluster assignments become less reliable guides to the true nearest neighbor. In practice, embedding models are trained to pack meaningful geometry into 768 or 1536 dimensions — high enough to represent rich semantics but not so high that distance becomes meaningless. Dimensionality reduction (PCA to 256D) is sometimes applied before indexing to speed up ANN with minimal recall loss.

**Q4. When would you choose HNSW over FAISS IVF and vice versa?**

Choose HNSW when you need maximum recall and have RAM to spare — it consistently achieves 95-99% recall@10 and query latency of 1-5ms, but the graph structure lives entirely in RAM and can be multiple terabytes for large corpora. Choose FAISS IVF when memory is constrained or the corpus is very large — it uses far less memory because it only stores cluster lists, not a full graph. IVF also supports runtime tuning via nprobe without rebuilding the index, which is useful when latency vs recall tradeoffs need adjusting after deployment. In practice, large systems often combine both: HNSW for a hot recent-docs tier, FAISS IVF+PQ for the cold full corpus tier.

**Q5. When would you choose BM25 over dense retrieval despite dense retrieval being more advanced?**

Several scenarios favor BM25. First, exact keyword matching — product IDs, SKUs, error codes, rare proper nouns. Embedding models may retrieve semantically similar but factually wrong results. Second, no labeled training data — dense retrieval requires domain-specific fine-tuning to work well; without it a generic embedding model often underperforms well-tuned BM25. Third, latency and infrastructure constraints — BM25 requires no GPU inference and no multi-terabyte ANN index in RAM. Fourth, interpretability requirements — in legal or medical settings where every retrieval decision must be auditable, BM25's explicit scoring is far easier to explain than embedding similarity.
