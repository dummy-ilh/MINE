# RAG Interview Prep — Day 2
## Embeddings & Vector Representations

---

## 🚀 Quick Summary

An embedding is a dense numerical vector that represents the *meaning* of a piece of text, produced by running it through a trained model, such that semantically similar text ends up close together in vector space and dissimilar text ends up far apart. This single idea is what turns "find documents relevant to this query" from a fuzzy language-understanding problem into a fast geometry problem — nearest-neighbor search — which is the entire foundation retrieval (Day 7+) is built on. Today is about how that vector gets created, how to compare two vectors, and how to choose an embedding model — three decisions that quietly determine the ceiling on your entire RAG system's quality.

**Think of it like a map of meaning.** Every sentence becomes a city with GPS coordinates. Cities about "return policies" cluster in one region of the map; cities about "battery specs" cluster somewhere else entirely. An embedding model's whole job is drawing this map accurately — and a bad map (wrong model, wrong pooling, wrong normalization) means your search will confidently point you to the wrong neighborhood, no matter how good your search algorithm is.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Embedding** | A fixed-length dense vector representing the semantic content of text |
| **Embedding dimensionality** | The length of that vector (e.g. 384, 768, 1536, 3072) |
| **Tokenization** | Splitting raw text into sub-word units the model actually processes |
| **Pooling** | Collapsing per-token vectors (one per token) into a single vector representing the whole input |
| **Bi-encoder** | An architecture that embeds queries and documents independently, into the same vector space |
| **Cosine similarity** | Angle-based similarity between two vectors, magnitude-invariant |
| **Dot product** | Raw multiply-and-sum of two vectors — magnitude-sensitive unless vectors are normalized |
| **Euclidean (L2) distance** | Straight-line distance between two vector "tips" in space |
| **MTEB** | Massive Text Embedding Benchmark — the standard leaderboard for comparing embedding models across tasks |
| **Embedding drift** | The problem where changing embedding models invalidates your existing vector index, since different models' vector spaces aren't compatible |
| **Quantization** | Compressing embedding vectors (e.g., float32 → int8 or binary) to save memory/storage at some accuracy cost |

---

# PHASE 1 — Intuition & Visual Map

## How text actually becomes a vector

```
  "How do I reset my Apple ID password?"
                  │
                  ▼
         ┌─────────────────┐
         │   TOKENIZATION    │   splits into sub-word units:
         │                   │   ["How", "do", "I", "reset", "my",
         │                   │    "Apple", "ID", "password", "?"]
         └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  ENCODER MODEL    │   each token gets its own vector,
         │ (transformer)     │   informed by surrounding context
         └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │     POOLING       │   collapse all per-token vectors
         │ (mean / CLS token)│   into ONE fixed-length vector
         └─────────────────┘
                  │
                  ▼
      [0.12, -0.44, 0.08, ..., 0.31]   ← the final embedding
      (e.g. 768 numbers, one point in 768-dimensional space)
```

**Why pooling matters (a frequently skipped detail):** A transformer encoder naturally produces *one vector per token*, not one vector for the whole sentence. To get a single sentence/chunk embedding, you need a pooling strategy:
- **Mean pooling** — average all the token vectors together. Generally the more robust default for sentence-embedding models (e.g., Sentence-BERT-style models are explicitly trained with mean pooling in mind).
- **CLS token pooling** — use only the vector at a special `[CLS]` token position (common in BERT-style classification setups). Works well when the model was specifically *trained* to compress meaning into that position, but is a poor choice if you just grab it from a model that wasn't trained with that objective — the CLS vector wasn't optimized to summarize the sentence in that case.
- **Gotcha:** using the wrong pooling strategy for a given model (e.g., CLS-pooling a model trained for mean-pooling) can silently produce mediocre embeddings that still "sort of work" — this is a common, hard-to-detect production bug because nothing crashes, similarity search just quietly gets worse.

---

# PHASE 2 — Math & Formulas

## Notation table

| Symbol | Meaning |
|---|---|
| `A`, `B` | Two embedding vectors |
| `A · B` | Dot product |
| `‖A‖`, `‖B‖` | Vector magnitudes (lengths) |
| `d` | Embedding dimensionality |
| `a_i`, `b_i` | The i-th coordinate of vectors A and B |

---

### 1. Dot Product

```
A · B = Σ_{i=1}^{d} a_i × b_i
```

**Plain English:** Multiply each corresponding pair of coordinates and sum the results. This single number blends *both* direction alignment and magnitude — two vectors pointing the same way but one being "longer" will produce a bigger dot product than two shorter aligned vectors.

**Worked example:**
```
A = [2, 1]
B = [3, 4]

A · B = (2×3) + (1×4) = 6 + 4 = 10
```

**What happens if magnitude changes:** If we scale A to `[4, 2]` (same direction, doubled length):
```
A · B = (4×3) + (2×4) = 12 + 8 = 20
```
Dot product doubled even though the *direction* of A didn't change at all — this is exactly why raw dot product can be misleading for text similarity unless vectors are normalized first.

---

### 2. Cosine Similarity

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

**Plain English:** Dot product, but divided by both vectors' lengths — this cancels out magnitude entirely and leaves purely the angle between them. Range: **-1** (opposite directions) to **1** (identical direction), with **0** meaning unrelated/perpendicular.

**Worked example (using the same A, B from above):**
```
‖A‖ = √(2² + 1²) = √5 ≈ 2.236
‖B‖ = √(3² + 4²) = √25 = 5.0

cosine_similarity = 10 / (2.236 × 5.0) = 10 / 11.18 ≈ 0.894
```

Now scale A to `[4, 2]` again and recompute:
```
‖A_scaled‖ = √(4² + 2²) = √20 ≈ 4.472
A_scaled · B = 20 (computed above)

cosine_similarity = 20 / (4.472 × 5.0) = 20 / 22.36 ≈ 0.894
```
**Identical result (0.894) even though the raw dot product doubled from 10 to 20.** This is the concrete proof of why cosine similarity is magnitude-invariant and dot product isn't — same direction, same cosine similarity, regardless of scale.

**Why it matters in practice:** This is *the* default similarity metric for text embedding comparison in essentially every vector database, because embedding magnitude often reflects incidental factors (text length, model confidence, numerical artifacts of training) rather than meaning — you usually want to compare *what the text is about*, not *how "big" its vector happens to be*.

> **Critical gotcha (asked constantly):** If your embeddings are pre-normalized to unit length (`‖A‖ = ‖B‖ = 1`), then `cosine_similarity(A,B) = A·B` exactly — dot product and cosine similarity become mathematically identical. Many production systems normalize embeddings once at indexing time specifically so they can use raw (faster) dot product at query time and get cosine-equivalent results without paying the normalization cost on every single comparison.

---

### 3. Euclidean (L2) Distance

```
L2(A, B) = √( Σ_{i=1}^{d} (a_i - b_i)² )
```

**Plain English:** The straight-line distance between the two vectors' tips, as points in space — literally the Pythagorean theorem generalized to *d* dimensions.

**Worked example:**
```
A = [2, 1]
B = [3, 4]

L2(A,B) = √( (2-3)² + (1-4)² )
        = √( 1 + 9 )
        = √10
        ≈ 3.162
```

**What happens if each term changes:** Unlike cosine similarity, L2 distance *is* sensitive to magnitude — two vectors pointing in the exact same direction but with very different lengths will have a large L2 distance despite being "similar" in a cosine sense. This makes L2 the wrong default for most text embedding comparisons (magnitude usually isn't meaningful for text), but it can be the right choice for embeddings where absolute position/scale genuinely matters (some image embeddings, or normalized embeddings where L2 and cosine-based ranking become monotonically related anyway).

### Similarity metric comparison table

| Metric | Sensitive to magnitude? | Range | Typical RAG use |
|---|---|---|---|
| **Cosine similarity** | No | -1 to 1 | Default for text embeddings |
| **Dot product** | Yes (unless pre-normalized) | -∞ to ∞ | Fast alternative when embeddings are pre-normalized |
| **Euclidean (L2) distance** | Yes | 0 to ∞ | Sometimes used for images; less common for text |

---

## Choosing an Embedding Model

This is a decision with real downstream consequences, and interviewers like probing the trade-offs behind it rather than just "which model is best."

### Dimensions to reason about

| Factor | Trade-off |
|---|---|
| **Dimensionality** | Higher dimensions (e.g. 3072 vs. 384) generally capture more nuance, but cost more storage, more compute per similarity comparison, and more memory in the vector index — this is a direct multiplier on your infrastructure cost at scale |
| **General-purpose vs. domain-specific** | A general model (trained on broad web text) is a safe default and works reasonably everywhere; a domain-fine-tuned model (medical, legal, code) often significantly outperforms general models *within* that domain, because domain vocabulary and semantic relationships differ from general web text |
| **Model size / latency** | Larger embedding models produce (often) better embeddings but are slower to run at indexing time and, more importantly, slower at *query* time if you're embedding queries live — this directly affects end-to-end latency budgets |
| **Benchmark performance (MTEB)** | The Massive Text Embedding Benchmark is the standard leaderboard comparing models across retrieval, clustering, classification, and other tasks — useful as a starting filter, but benchmark rank doesn't guarantee it'll be the best fit for *your specific domain and query style*, so validate on your own eval set (tie-in to Module 7's golden eval set) before committing |
| **Matryoshka embeddings** | Some newer embedding models are trained so that *truncating* the vector (e.g., using only the first 256 of 1536 dimensions) still produces a usable, if slightly less accurate, embedding — this lets you dynamically trade off storage/speed vs. accuracy without needing a separate smaller model |

> **Why This Matters callout:** A very realistic interview scenario is "your retrieval quality is mediocre — is it the embedding model, or something else?" A strong answer knows to check: is this a general-purpose model applied to a highly specialized domain (legal/medical/code) where a domain-specific embedding model would likely help substantially? This is often a bigger lever than tuning k, chunk size, or reranking — but it's frequently overlooked because switching embedding models means re-embedding the entire corpus (see "embedding drift" below), which people are reluctant to do.

---

## Embedding Drift & Versioning (a commonly missed practical gotcha)

**The problem:** Two different embedding models do **not** produce vectors in the same space — a vector from Model A and a vector from Model B are not comparable to each other via cosine similarity, even if they're the same dimensionality. This means:
- If you upgrade your embedding model, you must **re-embed your entire corpus**, not just newly added documents — mixing old and new embeddings in the same index silently produces nonsensical similarity scores.
- Query embeddings and document embeddings must always come from the *same* model (or, for asymmetric setups, from a matched query/document encoder pair — some models are explicitly trained as asymmetric bi-encoders with separate query and passage encoders).

**Why it matters in practice:** This is an infrastructure/versioning problem as much as a modeling one — production systems need a plan for how to do a "hot" re-embedding migration (e.g., dual-write to old and new indexes during a transition period, or accept a maintenance window) rather than assuming a model upgrade is a drop-in change.

---

## Quantization (storage/speed optimization)

**Plain English:** Embeddings are normally stored as `float32` (4 bytes per dimension). At large scale (hundreds of millions of vectors), this gets expensive in memory and storage. Quantization compresses each number to a smaller representation — e.g., `int8` (1 byte, a ~4x reduction) or even binary (1 bit per dimension, a ~32x reduction) — at some cost to precision/accuracy.

**Worked numerical example:**
```
1 million vectors, 768 dimensions, float32 (4 bytes/dim):
  storage = 1,000,000 × 768 × 4 bytes = 3,072,000,000 bytes ≈ 2.86 GB

Same vectors, int8 quantized (1 byte/dim):
  storage = 1,000,000 × 768 × 1 byte = 768,000,000 bytes ≈ 0.72 GB
```
Roughly a **4x storage reduction**, at the cost of some retrieval accuracy — a common production trade-off at scale, often paired with a re-ranking step on the un-quantized top candidates to recover most of the lost precision (quantize for the fast first-pass search, use full precision for the final re-scoring of a small candidate set).

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** What does "pooling" mean in the context of generating a sentence embedding, and why is it necessary?

<details>
<summary>Show answer</summary>

A transformer encoder produces one vector per input token, not a single vector for the whole sentence. Pooling collapses those per-token vectors into one fixed-length vector representing the entire input — commonly via mean pooling (averaging all token vectors) or CLS-token pooling (using a single designated token's vector). It's necessary because downstream similarity search needs one comparable vector per chunk/query, not a variable-length set of token vectors.
</details>

---

**Q2 (Easy — calculation).** Compute the cosine similarity between `A = [1, 2]` and `B = [2, 4]`. What do you notice, and why?

<details>
<summary>Show answer</summary>

```
A · B = (1×2) + (2×4) = 2 + 8 = 10
‖A‖ = √(1+4) = √5 ≈ 2.236
‖B‖ = √(4+16) = √20 ≈ 4.472

cosine_similarity = 10 / (2.236 × 4.472) = 10 / 10.0 = 1.0
```
B is exactly A scaled by 2 — same direction, different magnitude. Cosine similarity is a perfect 1.0 because it's magnitude-invariant; only the direction matters.
</details>

---

**Q3 (Medium — conceptual).** Why would you choose dot product over cosine similarity in a production system, and what has to be true for that to be safe?

<details>
<summary>Show answer</summary>

Dot product is cheaper to compute than cosine similarity, since it skips the normalization (dividing by both magnitudes) step. It's safe to use as a drop-in replacement for cosine similarity if and only if the embeddings have already been normalized to unit length at indexing/query time — in that case dot product and cosine similarity are mathematically identical, so you get the speed benefit with no change in ranking behavior. If embeddings aren't normalized, dot product will be skewed by magnitude and won't reflect pure semantic similarity.
</details>

---

**Q4 (Medium — conceptual).** Your retrieval quality is mediocre on a legal-document RAG system using a general-purpose embedding model. What would you investigate, and why might switching embedding models be a bigger lever than tuning chunk size or k?

<details>
<summary>Show answer</summary>

I'd investigate whether a domain-specific (legal) embedding model would outperform the general-purpose one — legal text has specialized vocabulary and semantic relationships (e.g., specific statute references, precedent relationships) that a general web-trained embedding model may not represent as accurately in its vector space. Because embedding quality determines whether semantically relevant documents even land close together in vector space in the first place, a poor embedding model creates a ceiling that no amount of downstream tuning (chunk size, k, reranking) can fully overcome — those levers optimize search *within* an already-drawn map, but if the map itself is drawn poorly for this domain, better search algorithms can't fix that. The trade-off to flag: switching embedding models requires re-embedding the entire corpus (embedding drift), which is more disruptive than a chunk-size or k change, but can be the higher-leverage fix.
</details>

---

**Q5 (Hard — calculation + reasoning).** You have 50 million vectors at 1536 dimensions, stored as float32. Compute the storage size, then compute the storage size if quantized to int8, and explain the accuracy trade-off involved.

<details>
<summary>Show answer</summary>

```
float32: 50,000,000 × 1536 × 4 bytes = 307,200,000,000 bytes ≈ 286.1 GB
int8:    50,000,000 × 1536 × 1 byte  =  76,800,000,000 bytes ≈  71.5 GB
```
Roughly a 4x reduction (286 GB → 71.5 GB). The trade-off: int8 quantization reduces the numerical precision of each dimension, which can slightly blur fine-grained similarity distinctions and reduce recall/ranking accuracy compared to full float32 precision. A common mitigation is a two-stage approach: use the cheap quantized vectors for a fast first-pass candidate search over the full 50M vectors, then re-score only the small candidate set (e.g., top 200) using full-precision vectors for the final ranking — recovering most of the accuracy while still getting the bulk of the storage/speed win.
</details>

---

**Q6 (Hard — "spot the bug" scenario).** A team upgrades their embedding model mid-quarter, re-embeds only newly ingested documents going forward, and leaves the existing 6 months of indexed documents on the old embeddings to "save time." Similarity search quality degrades unpredictably. What happened?

<details>
<summary>Show answer</summary>

This is an **embedding drift** bug: different embedding models produce vectors in incompatible vector spaces, even at the same dimensionality — a vector from the old model and a vector from the new model are not meaningfully comparable via cosine similarity. By mixing old-model embeddings (6 months of history) with new-model embeddings (new documents) in the same index, similarity scores become essentially meaningless whenever a query embedding (produced with the new model) is compared against an old-model document embedding — sometimes it'll accidentally still look "close enough," sometimes it won't, producing exactly the unpredictable degradation described. The fix is a full re-embedding of the entire existing corpus with the new model (via a migration strategy such as dual-writing to old and new indexes during a transition window), not a partial/incremental switch.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Using the wrong pooling strategy for a given embedding model (e.g., CLS-pooling a mean-pooling-trained model) — a silent quality bug, nothing crashes, search just quietly gets worse.
- ❌ Using raw (non-normalized) dot product and being surprised results skew toward longer/higher-magnitude embeddings.
- ❌ Picking an embedding model purely off MTEB leaderboard rank without validating on your own domain-specific eval set.
- ❌ Mixing embeddings from two different model versions in the same index ("embedding drift") after a partial migration.
- ❌ Assuming a bigger/higher-dimensional embedding model is automatically better — it's a real cost/latency/storage trade-off, not a free upgrade.
- ❌ Treating embedding model choice as a "set once, forget it" decision, rather than as a lever worth revisiting when retrieval quality plateaus.

---

# 📌 Cheat Sheet (Day 2)

**Pipeline:** text → tokenize → transformer encoder (per-token vectors) → pooling (mean or CLS) → one fixed-length embedding vector.

**Similarity metrics:** Cosine similarity = `(A·B)/(‖A‖‖B‖)`, magnitude-invariant, the default for text. Dot product = `Σ a_i·b_i`, magnitude-sensitive unless pre-normalized (then it equals cosine similarity, and is cheaper to compute). Euclidean/L2 = straight-line distance, magnitude-sensitive, less common for text.

**Model choice levers:** dimensionality (accuracy vs. cost), general vs. domain-specific (often a bigger lever than people expect), MTEB as a starting filter (not a final answer — validate on your own data), matryoshka embeddings for flexible truncation.

**Operational gotchas:** embedding drift (never mix vector spaces from different models — full re-embed on upgrade), quantization (float32 → int8 ≈ 4x storage savings, pair with a full-precision re-scoring pass on the shortlist to recover accuracy).

---

*End of Day 2. Next up — Day 3: Chunking Strategies.*
