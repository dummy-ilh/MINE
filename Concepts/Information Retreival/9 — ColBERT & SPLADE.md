# Chapter 9 — ColBERT & SPLADE

## What is it?

Chapters 2–4 gave you three retrieval models that sit on a spectrum:

```
BM25          →    bi-encoder      →    cross-encoder
fastest            middle               slowest
least accurate     middle               most accurate
sparse             dense                dense
```

ColBERT and SPLADE both live in the gaps on this spectrum — they are architecturally different answers to the same question: **can we get closer to cross-encoder accuracy while staying closer to bi-encoder speed?**

**ColBERT** (Contextualized Late Interaction over BERT) keeps full token-level embeddings for both query and document and scores them with a cheap interaction at query time — faster than a cross-encoder, more accurate than a bi-encoder.

**SPLADE** (Sparse Lexical and Expansion Model) learns to produce sparse vectors — like BM25 but the sparsity pattern is learned from data, not hand-crafted. It gets the interpretability and exact-match strength of sparse retrieval with the semantic understanding of dense retrieval.

Both are increasingly common in FAANG interviews as the follow-up to "what comes after bi-encoders?"

---

## The intuition

### Why bi-encoders lose information

A bi-encoder compresses an entire document into a single 768-dim vector. Think about what that means for a long document:

```
document: "Python is great for data science. Java is better for enterprise.
           C++ is fastest for systems programming. Ruby is elegant for web."

bi-encoder: [0.31, 0.87, 0.12, ..., 0.54]  ← one vector for ALL of this
```

If the query is "systems programming language," the single vector is a blurry average of all four topics. The C++ information is diluted by Python, Java, and Ruby. The bi-encoder loses the ability to match query terms to specific parts of the document.

A cross-encoder sees query and document together and can attend directly from "systems programming" to the C++ sentence. That's why it's more accurate — but also why it can't pre-compute anything.

**ColBERT's idea:** don't compress to one vector. Keep one vector per token. Match query tokens to document tokens at query time using a cheap operation. You pre-compute all the document token vectors — the expensive part — offline. The matching at query time is fast.

### Why dense retrieval misses exact matches

A bi-encoder for "GTX-4090-A1 driver error code 43" might retrieve documents about similar GPUs because they're semantically nearby. But the user wants that exact product. BM25 nails it. Dense retrieval misses it.

**SPLADE's idea:** learn a sparse vector where each dimension corresponds to a vocabulary term — like TF-IDF — but the weights are learned by a neural network, not computed by a formula. The network also learns to expand the sparse vector with related terms (query expansion built-in). You get semantic understanding AND exact match in one model.

---

## ColBERT — the equations

### Standard bi-encoder (recap)

```
q_vec = BERT(query)          → single vector ∈ R^768
d_vec = BERT(document)       → single vector ∈ R^768
score = q_vec · d_vec
```

### ColBERT late interaction

```
Q = BERT(query)     → matrix ∈ R^(|q| × 128)   one 128-dim vector per query token
D = BERT(document)  → matrix ∈ R^(|d| × 128)   one 128-dim vector per doc token

score(q, d) = Σᵢ max_j (Qᵢ · Dⱼ)
```

Where:
- `|q|` = number of query tokens
- `|d|` = number of document tokens
- `Qᵢ` = embedding of the i-th query token
- `Dⱼ` = embedding of the j-th document token
- `max_j (Qᵢ · Dⱼ)` = find the document token most similar to query token i
- `Σᵢ` = sum these max similarities across all query tokens

This is called **MaxSim** — for each query token, find its best matching document token, then sum those best matches.

**Why 128-dim instead of 768-dim?** ColBERT projects from 768 to 128 to reduce storage. Each document token is a 128-dim vector stored on disk. This is still much larger than a single 768-dim vector per document but smaller than 768-dim per token.

---

## ColBERT — worked numeric example

```
query:    "fast sorting"           → 2 tokens: ["fast", "sorting"]
document: "quicksort runs quickly" → 3 tokens: ["quicksort", "runs", "quickly"]

token embeddings (2-dim for visibility, normally 128-dim):

query token vectors Q:
  Q["fast"]    = [0.9, 0.2]
  Q["sorting"] = [0.3, 0.8]

document token vectors D (pre-computed, stored offline):
  D["quicksort"] = [0.4, 0.7]
  D["runs"]      = [0.5, 0.5]
  D["quickly"]   = [0.8, 0.3]
```

### Step 1 — compute all dot products between query and document tokens

```
Q["fast"] · D["quicksort"] = (0.9×0.4) + (0.2×0.7) = 0.36 + 0.14 = 0.50
Q["fast"] · D["runs"]      = (0.9×0.5) + (0.2×0.5) = 0.45 + 0.10 = 0.55
Q["fast"] · D["quickly"]   = (0.9×0.8) + (0.2×0.3) = 0.72 + 0.06 = 0.78 ← max

Q["sorting"] · D["quicksort"] = (0.3×0.4) + (0.8×0.7) = 0.12 + 0.56 = 0.68 ← max
Q["sorting"] · D["runs"]      = (0.3×0.5) + (0.8×0.5) = 0.15 + 0.40 = 0.55
Q["sorting"] · D["quickly"]   = (0.3×0.8) + (0.8×0.3) = 0.24 + 0.24 = 0.48
```

### Step 2 — MaxSim: for each query token, take the max over all document tokens

```
MaxSim("fast")    = max(0.50, 0.55, 0.78) = 0.78  ← matched to "quickly"
MaxSim("sorting") = max(0.68, 0.55, 0.48) = 0.68  ← matched to "quicksort"
```

### Step 3 — sum MaxSims to get final score

```
score("fast sorting", "quicksort runs quickly") = 0.78 + 0.68 = 1.46
```

### What this reveals

"fast" matched to "quickly" — not "fast" itself — because "quickly" is the token most semantically similar to "fast" in this document. "sorting" matched to "quicksort" — which is a sorting algorithm — without sharing the exact word. A bi-encoder would have compressed all three document tokens into one blurry average vector. ColBERT preserves the ability to match query tokens to the most relevant part of the document.

### Comparing ColBERT to bi-encoder and cross-encoder

```
bi-encoder:    compress D to one vector offline
               query time: 1 dot product  ← fastest, least accurate

ColBERT:       store all token vectors of D offline
               query time: |q| × |d| dot products + MaxSim  ← middle
               for query of 10 tokens, doc of 100 tokens: 1000 dot products

cross-encoder: no precomputation
               query time: full BERT forward pass over [query; doc]  ← slowest, most accurate
```

---

## ColBERT — storage cost

The tradeoff ColBERT makes is accuracy for storage:

```
corpus: 1M documents, avg 100 tokens per doc, 128-dim vectors, float32

bi-encoder:  1M × 768 × 4 bytes          = 3.07 GB
ColBERT:     1M × 100 × 128 × 4 bytes    = 51.2 GB   ← 16× larger
cross-encoder: nothing stored (no precomputation)
```

ColBERT requires significantly more disk and RAM than a bi-encoder. This is its primary practical limitation. ColBERT v2 addresses this with compression techniques (residual compression) that bring storage down ~6–10×.

---

## SPLADE — the equations

SPLADE produces a sparse vector over the full vocabulary — one weight per vocabulary term, most weights zero.

```
BERT(d) → [CLS] + token representations → MLM head → logits over vocabulary

SPLADE weight for term t in document d:

w(t, d) = Σⱼ log(1 + ReLU(Eⱼₜ))
```

Where:
- `Eⱼₜ` = the logit for vocabulary term t at token position j (from the MLM head)
- `ReLU` = max(0, x) — ensures non-negative weights
- `log(1 + x)` — compresses weights, same motivation as IDF's log
- `Σⱼ` = aggregate across all token positions in the document

The result is a sparse vector of shape `[vocab_size]` (e.g. 30,000 dimensions for BERT's vocabulary) where most entries are zero. Non-zero entries represent terms the model thinks are important — both terms that actually appear in the document AND terms the model has learned to expand into.

### Retrieval with SPLADE

```
query sparse vector:    q ∈ R^V   (V = vocab size, mostly zeros)
document sparse vector: d ∈ R^V   (mostly zeros)

score(q, d) = q · d = Σₜ q(t) × d(t)
```

This is an inverted index dot product — exactly like BM25 — but the weights come from a learned neural model rather than a formula.

---

## SPLADE — worked numeric example

```
vocabulary (simplified to 8 terms):
  idx: 0="the"  1="dog"  2="canine"  3="fast"  4="quick"  5="run"  6="animal"  7="slow"

document: "the dog runs fast"

SPLADE produces sparse vector for this document:
  w(0, "the")    = 0.00  ← stopword, suppressed
  w(1, "dog")    = 1.82  ← appears in doc, high weight
  w(2, "canine") = 0.94  ← doesn't appear but model expands: dog → canine
  w(3, "fast")   = 1.45  ← appears in doc
  w(4, "quick")  = 0.71  ← expansion: fast → quick
  w(5, "run")    = 1.31  ← appears (as "runs"), stemmed
  w(6, "animal") = 0.43  ← expansion: dog → animal
  w(7, "slow")   = 0.00  ← opposite of fast, suppressed

document sparse vector d = [0, 1.82, 0.94, 1.45, 0.71, 1.31, 0.43, 0]
```

Now score two queries:

```
query 1: "canine speed"
SPLADE query vector q1 = [0, 0.21, 1.54, 0.62, 0.89, 0.11, 0.30, 0]

score(q1, d) = (0×0) + (0.21×1.82) + (1.54×0.94) + (0.62×1.45) +
               (0.89×0.71) + (0.11×1.31) + (0.30×0.43) + (0×0)
             = 0 + 0.382 + 1.448 + 0.899 + 0.632 + 0.144 + 0.129 + 0
             = 3.634

query 2: "slow turtle"
SPLADE query vector q2 = [0, 0, 0, 0, 0, 0, 0.12, 0.95]

score(q2, d) = (0.12×0.43) + (0.95×0)
             = 0.052 + 0
             = 0.052   ← very low, correct
```

Query 1 "canine speed" scores high even though neither "canine" nor "speed" appears in the document. The SPLADE expansion built the bridge: dog→canine, fast→quick/speed. Query 2 "slow turtle" scores near zero — correctly, the document is about something fast, not slow.

**This is what BM25 cannot do.** BM25 would score both queries at zero (no word overlap). SPLADE gets semantic understanding while still using an inverted index for retrieval — fast and interpretable.

---

## SPLADE — sparsity and efficiency

The key to SPLADE's efficiency is that most weights are exactly zero:

```
vocabulary size: 30,000 terms
avg non-zero terms per document: ~100–200 (0.3–0.7% of vocabulary)

inverted index lookup: same O(1) per term as BM25
storage per document: ~100–200 (term, weight) pairs ← sparse, tiny
```

SPLADE can use the same inverted index infrastructure as BM25 — Elasticsearch, Lucene — with no ANN index required. This is a massive operational advantage.

---

## ColBERT vs SPLADE vs bi-encoder vs cross-encoder

| Property | Bi-encoder | ColBERT | SPLADE | Cross-encoder |
|----------|-----------|---------|--------|---------------|
| Pre-computation | Full doc vector | All token vectors | Sparse doc vector | None |
| Query time op | 1 dot product | MaxSim (|q|×|d| dots) | Sparse dot product | Full forward pass |
| Exact match | Poor | Moderate | Excellent | Excellent |
| Semantic match | Good | Very good | Good | Excellent |
| Storage overhead | Low | Very high (16×) | Low (sparse) | None |
| Index type | ANN (FAISS/HNSW) | ColBERT-specific index | Inverted index | N/A |
| Interpretable | No | Partial | Yes (term weights) | No |
| Speed rank | 1st (fastest) | 3rd | 2nd | 4th (slowest) |
| Accuracy rank | 3rd | 2nd | 2nd–3rd | 1st (best) |

---

## Why they work / why they fail

### ColBERT

**Why it works:**
- Preserves token-level interactions — no information lost to single-vector compression
- MaxSim is a principled way to find the most relevant part of a document for each query term
- Document token vectors are pre-computed offline — query time is fast relative to cross-encoder
- Significantly outperforms bi-encoders on passage retrieval benchmarks (MS MARCO)

**Why it fails:**
- Storage cost is prohibitive at scale — 16× larger than bi-encoder index
- ColBERT-specific index infrastructure (not Elasticsearch or standard FAISS) — engineering overhead
- Slower than bi-encoder at query time — MaxSim across |q|×|d| token pairs adds up
- ColBERT v2 reduces storage with residual compression but adds complexity

### SPLADE

**Why it works:**
- Combines exact match (inverted index) with semantic expansion (learned weights) in one model
- Uses standard inverted index infrastructure — drop-in for BM25 operationally
- Interpretable — you can inspect which terms the model expanded and why
- Strong performance on out-of-domain queries because expansion handles vocabulary mismatch
- Sparse vectors are tiny compared to dense vectors — efficient storage

**Why it fails:**
- Expansion quality depends on training data — poor domain coverage → poor expansion
- Slower to index than BM25 (requires neural forward pass per document vs. simple counting)
- Less effective on long queries where the dense semantic space is richer
- SPLADE models are not as widely available or battle-tested as standard bi-encoders

---

## The one thing to remember

ColBERT keeps one vector per token instead of one per document, matching query tokens to the most relevant document tokens at query time (MaxSim) — more accurate than bi-encoders at the cost of storage. SPLADE learns sparse vectors with built-in query expansion — semantic understanding using an inverted index, bridging the gap between BM25 and dense retrieval without ANN infrastructure.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `Q = BERT(q) ∈ R^(\|q\|×128)` | ColBERT: matrix of query token embeddings |
| `D = BERT(d) ∈ R^(\|d\|×128)` | ColBERT: matrix of document token embeddings |
| `score(q,d) = Σᵢ max_j(Qᵢ·Dⱼ)` | ColBERT MaxSim scoring |
| `w(t,d) = Σⱼ log(1 + ReLU(Eⱼₜ))` | SPLADE weight for term t in document d |
| `score(q,d) = q·d = Σₜ q(t)×d(t)` | SPLADE retrieval: sparse dot product |
| `storage_ColBERT = N×\|d\|×128×4` | ColBERT storage: tokens × dim × bytes |
| `storage_SPLADE ≈ N×200×8` | SPLADE storage: ~200 non-zero terms × (term_id + weight) |

---

## Interview Q&A

**Q1. What problem does ColBERT solve that a bi-encoder cannot?**

A bi-encoder compresses an entire document into a single vector — all semantic content must fit into 768 numbers. For long or topically diverse documents, this compression loses the ability to match a query term to a specific relevant passage. ColBERT keeps one 128-dim vector per token, so every token in the document is independently matchable to every query token. The MaxSim operation finds the best document token for each query token, then sums those best matches. This means ColBERT can retrieve a document about "C++ systems programming" for a query about "fast systems language" by matching "fast" to "C++" via their token embeddings — without averaging C++ information with the other topics in the document. The tradeoff is 16× more storage.

**Q2. Explain MaxSim. Why sum the maxes instead of taking the max of all dot products?**

MaxSim computes, for each query token, the maximum similarity to any document token, then sums across all query tokens. Taking the max of all dot products would give you the single most similar (query token, document token) pair — this ignores most of the query. If the query is "fast sorting algorithm," you want evidence that the document is relevant to "fast," to "sorting," and to "algorithm" — not just to whichever one has the highest single dot product. Summing the per-query-token maxes ensures every query term contributes to the score. It's conceptually similar to how BM25 sums TF-IDF scores across all query terms — every term must be accounted for.

**Q3. How does SPLADE differ from BM25 if both produce sparse vectors and use an inverted index?**

BM25 weights are computed by a fixed formula — TF normalized by document length, multiplied by log(N/df). The weights only reflect terms that literally appear in the document. SPLADE weights are learned by a neural network trained on relevance data — the network learns which terms are important and also expands into related terms that don't appear in the document. A SPLADE vector for a document about "dogs" will have non-zero weights for "canine," "pet," "animal" even if those words don't appear. BM25 would have zero weights for those terms. SPLADE also suppresses uninformative terms (stopwords get near-zero weights from the ReLU) without an explicit stopword list. The retrieval operation is the same — sparse dot product over an inverted index — but the quality of the weights is much higher.

**Q4. Where would you place ColBERT in a production retrieval pipeline?**

ColBERT fits best as a re-ranker in the second stage — not as a first-stage retriever. As a first-stage retriever over 100M documents, ColBERT's storage cost (50+ GB for token vectors) and query-time MaxSim computation are manageable but non-trivial. As a re-ranker over the top-200 candidates from a BM25 + bi-encoder first stage, ColBERT is excellent: you only compute MaxSim for 200 documents (not 100M), the storage for 200 token matrices is trivial, and you get near cross-encoder accuracy at much lower latency. The pipeline becomes: BM25 + bi-encoder ANN → RRF → top 200 → ColBERT re-rank → top 10. This gives you first-stage recall from the hybrid retrieval and second-stage precision from ColBERT, avoiding the full cross-encoder latency.

**Q5. A team proposes replacing your BM25 + bi-encoder hybrid with SPLADE alone. What are the tradeoffs?**

Arguments for SPLADE alone: it handles both exact match and semantic expansion in one model, uses standard inverted index infrastructure (no ANN index needed), is interpretable, and is operationally much simpler — one system instead of two. Arguments against: SPLADE's semantic coverage is not as rich as a well-trained bi-encoder for purely conceptual queries — it works through vocabulary expansion, which is limited to terms in the training vocabulary, not the full continuous semantic space. For highly semantic or cross-lingual queries, a bi-encoder in a dense vector space generalizes better. SPLADE also requires neural inference at indexing time, which is slower than BM25's term counting. The right answer depends on your query distribution: if your queries are mostly keyword-style with some vocabulary mismatch, SPLADE alone may be sufficient and much simpler to operate. If you have many conceptual or cross-lingual queries, keep the bi-encoder.

---

Ready for your comments — what stays, what changes, what's missing?
