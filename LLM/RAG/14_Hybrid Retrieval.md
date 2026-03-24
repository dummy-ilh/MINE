# 🚀 RAG Mastery – Day 14

# 🧬 Hybrid Retrieval (Dense + Sparse + Filters)


Single-vector search is not enough for production systems.

Modern RAG systems combine:

```
Dense Retrieval  (semantic)
+ Sparse Retrieval (keyword/BM25)
+ Metadata Filters
+ Optional Reranker
```

This is called **Hybrid Retrieval**.

---

# 1️⃣ Why Dense Alone Fails

Dense embeddings are good at:

✔ Semantic similarity
✔ Concept matching
✔ Synonyms

But bad at:

✘ Exact numbers
✘ Rare keywords
✘ IDs, codes, SKUs
✘ Legal references
✘ Short keyword queries

Example:

Query:

> “SOC2 Type II policy 2024 update”

Dense retrieval may ignore:

* "SOC2"
* "Type II"
* "2024"

But sparse retrieval (BM25) nails it.

---

# 2️⃣ Sparse Retrieval (BM25)

BM25 is a classic lexical ranking algorithm used in:

* Google Search
* Elasticsearch
* Apache Lucene

It scores documents using:

* Term frequency
* Inverse document frequency
* Length normalization

It doesn’t understand meaning — but it understands **keywords extremely well**.

---

# 3️⃣ Dense Retrieval (Embeddings)

Used in:

* OpenAI embedding models
* Cohere
* Hugging Face

Captures:

* Semantic similarity
* Context
* Meaning

Fails on:

* Exact term importance
* Domain-specific tokens

---

# 🧠 Hybrid = Best of Both

## Formula (Simple Version)

```
Final Score = α * Dense Score + β * BM25 Score
```

Where:

* α, β tuned via validation
* Normalize both scores first!

---

# 4️⃣ Hybrid Architecture

```
            ┌───────────┐
User Query →│ Embedding │→ Dense Vector Search
            └───────────┘

User Query → BM25 Search (Keyword)

        ↓
Score Normalization
        ↓
Weighted Merge
        ↓
Top-k Results
```

---

# 5️⃣ Practical Implementation

### Option A – Use Elasticsearch Hybrid

Elasticsearch supports:

* BM25
* Vector search
* Hybrid scoring
* Filters

Example query (conceptual):

```json
{
  "query": {
    "bool": {
      "must": $[
        {"match": {"text": "SOC2 policy update"}},
        {"knn": {"embedding": {"vector": $[...]$, "k": 10}}}
      ]$
    }
  }
}
```

---

### Option B – Manual Merge (FAISS + BM25)

```python
dense_results = dense_retriever(query)
sparse_results = bm25_retriever(query)

# normalize scores
# merge weighted
# deduplicate
```

---

# 6️⃣ Metadata Filtering (Underrated Power Move)

Before retrieval, filter by:

* Document type
* Date
* Customer
* Region
* Access permissions

Example:

> “Show Q4 revenue for Europe enterprise customers”

Filter first:

```
region = Europe
segment = enterprise
quarter = Q4
```

Then retrieve.

This dramatically improves precision.

---

# 7️⃣ Score Normalization (Critical)

Dense scores range differently than BM25.

Common methods:

* Min-Max normalization
* Z-score normalization
* Rank-based fusion (RRF)

---

## 🏆 Reciprocal Rank Fusion (RRF)

Used widely in IR systems.

Formula:

```
Score = 1 / (k + rank)
```

Very stable, no tuning needed.

Often beats weighted sums.

---

# 8️⃣ Real Production Pattern

Most serious RAG systems look like:

```
Query Rewrite
↓
Metadata Filter
↓
Hybrid Retrieve (Dense + BM25)
↓
Rerank (Cross-Encoder)
↓
Top 5 → LLM
```

This is how search-quality systems are built.

---

# 9️⃣ When to Use Hybrid

Use Hybrid if:

* Legal docs
* Financial reports
* Codebases
* Technical manuals
* Enterprise search
* FAQs with IDs

If you're building toy chatbot → dense is fine.

If you're building enterprise system → hybrid is mandatory.

---

# 🔬 Experimental Insight

In benchmarks:

Dense only → good semantic recall
Sparse only → good keyword precision
Hybrid → 10–25% retrieval boost

Especially for domain-specific corpora.

---

# 🧪 Exercise for Today

Implement:

1. BM25 (use rank-bm25 or Elastic)
2. Dense retriever
3. RRF merge
4. Compare against dense-only

Evaluate:

* Recall@5
* MRR
* Latency
* Failure cases

---


# PART 1 — What is Hybrid Retrieval?

Hybrid retrieval = combining:

* **Sparse search (BM25 / keyword / lexical)**
* **Dense search (vector similarity)**

Because each solves different failure modes.

---

## Why Not Just Use Vectors?

Dense retrieval:

* Understands semantics
* Good for paraphrases
* Bad at exact keywords, rare terms, numbers

Sparse retrieval:

* Exact matching
* Good for:

  * IDs
  * Dates
  * Product names
  * Legal clauses
* Bad for semantic paraphrasing

---

## Example

Query:

> “What are penalties under Section 498A?”

Vector search might return:

* Domestic violence related documents

BM25 will correctly hit:

* Exact legal section references

Best result?
👉 Combine both.

---

# PART 2 — How Hybrid Retrieval Works

Architecture:

```
User Query
     ↓
Sparse Retriever (BM25)
Dense Retriever (ANN)
     ↓
Score Normalization
     ↓
Fusion / Reranking
     ↓
Top-K Final Results
```

---

# PART 3 — Score Normalization (Critical)

Here’s the issue:

BM25 score range:

```
0 → 20+
```

Cosine similarity:

```
-1 → 1 (usually 0.3 → 0.9)
```

You cannot directly add them.

So we normalize.

---

## 🔹 Method 1: Min-Max Normalization

For each retriever:

$[
normalized = \frac{score - min}{max - min}
]$

Now both become:

```
0 → 1
```

Then combine:

$[
final = \alpha \cdot dense + (1-\alpha) \cdot sparse
]$

Example:

| Doc | BM25 | Dense | Norm BM25 | Norm Dense | Final |
| --- | ---- | ----- | --------- | ---------- | ----- |
| A   | 10   | 0.8   | 0.7       | 0.85       | 0.79  |
| B   | 15   | 0.6   | 1.0       | 0.6        | 0.76  |

If α = 0.6 → favor dense

---

### When to Use?

* Small top-K lists
* Same query batch
* Quick fusion

---

## 🔹 Method 2: Z-score Normalization

$[
z = \frac{score - \mu}{\sigma}
]$

Better when:

* Score distributions vary per query
* You want statistical scaling

Used in:

* Large production systems

---

## 🔹 Method 3: Reciprocal Rank Fusion (RRF)

Instead of scores, use ranks:

$[
RRF = \sum \frac{1}{k + rank}
]$

If document ranks:

| Doc | Sparse Rank | Dense Rank |
| --- | ----------- | ---------- |
| A   | 1           | 5          |
| B   | 3           | 2          |

Then:

$[
score = 1/(k+rank1) + 1/(k+rank2)
]$

Advantages:

* No need for normalization
* Robust to score distribution
* Very popular in hybrid systems

---

### When to Use RRF?

* When sparse and dense scoring scales are very different
* When using heterogeneous models
* When simplicity > tuning

---

# PART 4 — When to Use Which Strategy?

| Scenario               | Best Strategy          |
| ---------------------- | ---------------------- |
| Legal / Compliance     | RRF                    |
| E-commerce search      | Weighted normalization |
| Research search        | Z-score                |
| Low engineering effort | RRF                    |

---

# PART 5 — Metadata Filtering (Very Important)

Now let’s discuss filtering.

Example:

```
Find AI articles
WHERE country = 'India'
AND date > 2024
```

How is this done internally?

---

## Strategy 1 — Filter First, Then Vector Search

```
Filter → Reduce candidate set → ANN search
```

Works when:

* Filter reduces dataset significantly
* Few documents per partition

Used in:

* Qdrant
* Weaviate

Pros:

* Faster
* Efficient

Cons:

* If filter too broad → no benefit

---

## Strategy 2 — Search First, Then Filter

```
ANN top-100 → Apply filter → Return top-K
```

Used when:

* Filter not selective
* ANN search is cheap

Problem:
If filtering removes many results, you may get fewer than K outputs.

Solution:

* Over-fetch (top-200)

---

## Strategy 3 — Hybrid Filtering (Index-Level Filtering)

Advanced systems build:

* Inverted index for metadata
* HNSW graph aware of payload

This allows:

* Search constrained within filtered subset

Used in:

* Pinecone
* Qdrant

---

# Example End-to-End Flow

User query:

> “AI startups in India after 2023”

System:

1. Metadata filter:

   * country=India
   * year>2023
2. Dense retrieval within subset
3. Sparse retrieval within subset
4. Normalize scores
5. RRF fusion
6. Optional cross-encoder reranker

---

# PART 6 — Common Pitfalls

❌ Not normalizing scores
❌ Using fixed α without validation
❌ Filtering after top-10 (too small candidate set)
❌ Not over-fetching
❌ Ignoring evaluation metrics (Recall@k, MRR)

---

# PART 7 — When Hybrid Is Actually Necessary

You should use hybrid if:

* Queries contain numbers, IDs, product names
* Domain is legal / medical / financial
* You need high recall
* You see lexical mismatch failures

If your dataset is:

* Conversational knowledge base
* FAQ style

Dense-only might be enough.

---

# PART 8 — Production-Grade Hybrid Stack

Typical enterprise RAG:

```
Elasticsearch (BM25)
+
Vector DB (HNSW)
+
Fusion Layer (RRF)
+
Cross-Encoder Re-ranker
+
LLM
```

---

# Final Mental Model

Sparse → precision on exact tokens
Dense → semantic recall
Fusion → robustness
Metadata filter → relevance constraints
Reranker → final quality

---

Excellent. Now we design this like a **search infra engineer** building for 100M documents in production.

We’ll split this into two major parts:

1. **System Architecture for 100M hybrid retrieval**
2. **Evaluation + tuning methodology**

No fluff — real infra thinking.

---

# PART 1 — Designing Hybrid Retrieval for 100M Documents

## 🎯 Constraints (Assume)

* 100M documents
* 768-dim embeddings
* Metadata filtering required
* P95 latency target < 150ms
* Continuous updates
* Multi-tenant
* High recall requirement (RAG use case)

---

# 1️⃣ High-Level Architecture

```
                ┌───────────────┐
                │   API Layer   │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │ Query Router  │
                └───────┬───────┘
          ┌─────────────┼─────────────┐
          ↓                             ↓
  Sparse Retrieval                Dense Retrieval
 (BM25 / inverted index)          (ANN - HNSW/IVF)
          ↓                             ↓
          └──────────┬──────────────────┘
                     ↓
             Fusion Layer (RRF / Weighted)
                     ↓
             Cross-Encoder Re-ranker
                     ↓
                   Top-K
                     ↓
                    LLM
```

---

# 2️⃣ Storage Layer Design (100M Scale)

### Document Storage

* Raw docs in object storage (S3)
* Metadata in distributed DB
* Embeddings stored in vector DB shards

---

## Dense Index Choice

At 100M scale:

| Index    | Use?                        | Why         |
| -------- | --------------------------- | ----------- |
| HNSW     | Yes (if enough RAM)         | High recall |
| IVF + PQ | Yes (if memory constrained) | Compression |
| Flat     | No                          | Too slow    |

Most likely production choice:

* **IVF + PQ for base**
* Optional HNSW refinement layer

Used in:

* Milvus
* Pinecone

---

# 3️⃣ Sharding Strategy

100M × 768 dims ≈ 300GB raw float32

So we must shard.

### Shard by:

Option A — Hash of document ID
Option B — Semantic clustering (better pruning)

At 100M, use:

* 16–64 shards
* Replication factor 2–3

---

# 4️⃣ Metadata Filtering at Scale

You cannot filter naïvely.

### Correct Approach:

Build:

* Inverted index for metadata
* Vector index per shard

Execution:

```
Step 1: Apply metadata filter → candidate docIDs
Step 2: Restrict ANN search to those IDs
```

Advanced systems integrate filter constraints inside ANN graph traversal.

Supported by:

* Qdrant
* Weaviate

---

# 5️⃣ Query Execution Strategy

We do **parallel retrieval**:

Each shard:

* Compute sparse top-200
* Compute dense top-200

Coordinator:

* Normalize scores
* Apply RRF
* Merge
* Send top-100 to reranker
* Return top-10

---

# 6️⃣ Why Over-Fetch?

Because:

Filtering + fusion reduces recall.

Typical values:

* Retrieve 5× or 10× final k
* If final k = 20 → fetch 200 candidates

---

# 7️⃣ Reranking Layer

ANN ≠ final answer.

Add:

* Cross-encoder (BERT-like)
* Rerank top 100
* Improves MRR drastically

Latency tradeoff:

* Add 20–50ms

---

# PART 2 — Evaluation Metrics for Hybrid Systems

Now the real science begins.

You cannot tune hybrid systems without proper evaluation.

---

# 1️⃣ Core Retrieval Metrics

### Recall@k

$[
Recall@k = \frac{\text{relevant docs in top-k}}{\text{total relevant docs}}
]$

Most important for RAG.

Why?

If retrieval fails, LLM fails.

---

### Precision@k

$[
Precision@k = \frac{\text{relevant in top-k}}{k}
]$

Important for search systems.

---

### MRR (Mean Reciprocal Rank)

$[
MRR = \frac{1}{rank\ of\ first\ relevant}
]$

Critical when:

* Only one correct answer
* QA systems

---

### NDCG (Normalized Discounted Cumulative Gain)

Accounts for ranking quality:

$[
DCG = \sum \frac{rel_i}{\log_2(i+1)}
]$

Best metric for:

* Graded relevance
* Enterprise search

---

# 2️⃣ Hybrid-Specific Evaluation

Now things get interesting.

We must measure:

* Dense-only performance
* Sparse-only performance
* Hybrid performance

You want:

Hybrid > max(Dense, Sparse)

If not, fusion is broken.

---

# 3️⃣ Offline Evaluation Pipeline

Steps:

1. Build query set (1K–10K queries)
2. Label relevance
3. Compute metrics for:

   * BM25 only
   * Dense only
   * Hybrid α=0.1 … 0.9
   * RRF
4. Plot curves

---

# 4️⃣ Online Metrics

Once deployed:

* Click-through rate (CTR)
* Dwell time
* Success rate
* Query reformulation rate

For RAG:

* LLM answer correctness
* Hallucination rate
* Answer grounding score

---

# 5️⃣ Tuning Strategy

### Step 1 — Tune Dense Retriever

Adjust:

* HNSW ef_search
* IVF nprobe
* Top-k fetch size

Optimize Recall@100

---

### Step 2 — Tune Sparse Retriever

Adjust:

* BM25 k1
* b parameter

Optimize Recall@100

---

### Step 3 — Tune Fusion

Test:

* Weighted sum (α sweep)
* RRF
* Z-score fusion

Pick best NDCG@10

---

### Step 4 — Tune Reranker

Measure:

* MRR before reranker
* MRR after reranker

If reranker gives <5% gain → something wrong.

---

# 6️⃣ Failure Mode Analysis

Check:

* Are long queries failing?
* Are numeric queries failing?
* Are entity queries failing?
* Is metadata filtering killing recall?

Manually inspect 100 failures.

This is where elite systems improve.

---

# 7️⃣ Scaling Concerns at 100M

Watch for:

* Memory pressure
* Shard imbalance
* Filter selectivity skew
* Hot partitions
* Embedding drift over time

You may need:

* Periodic re-indexing
* Embedding model versioning
* Shadow evaluation pipeline

---

# Final System Blueprint

For 100M documents:

* Distributed sparse index (Elasticsearch)
* Distributed vector index (HNSW/IVF)
* RRF fusion
* Cross-encoder reranker
* Metadata-aware ANN
* Evaluation pipeline with NDCG + Recall@k
* A/B testing infra

---

# Advanced Question for You

If:

* Dense Recall@100 = 82%
* Sparse Recall@100 = 75%
* Hybrid Recall@100 = 84%

Is your fusion working properly?

You gave:

* Dense Recall@100 = **82%**
* Sparse Recall@100 = **75%**
* Hybrid Recall@100 = **84%**

Now let’s reason like an infra engineer.

---

# Short Answer

Yes… but **barely**.

It is technically working —
but it is **not working optimally**.

Let’s break this down carefully.

---

# 1️⃣ What Should Hybrid Ideally Do?

In theory, hybrid retrieval should:

$[
Hybrid \ge \max(Dense, Sparse)
]$

That condition is satisfied:

$[
84% > 82%
]$

So it is improving recall.

But here’s the real question:

👉 Is +2% meaningful?

---

# 2️⃣ How Much Gain Should We Expect?

At 100M scale:

* Dense and sparse often retrieve different failure cases.
* Proper hybrid systems often improve **3–8% recall@100**.
* In some domains (legal, finance), even **10%+**.

A +2% gain suggests one of these:

1. Dense and sparse are highly correlated (retrieving same docs)
2. Fusion weights not tuned properly
3. Candidate pool too small (under-fetching)
4. Metadata filtering limiting diversity
5. Sparse not strong enough (BM25 poorly tuned)

---

# 3️⃣ Let’s Think Mathematically

Let:

* D = dense retrieved set
* S = sparse retrieved set

Hybrid recall depends on:

$[
|D ∪ S|
]$

If overlap between D and S is very high:

$[
|D ∩ S| \text{ is large}
]$

Then hybrid cannot improve much.

A +2% gain implies:

Dense and sparse results overlap heavily.

---

# 4️⃣ Diagnostic Questions

To know if it's “really” working, check:

### A. What is the overlap ratio?

$[
Overlap = \frac{|D ∩ S|}{|D|}
]$

If overlap > 70%, sparse adds little diversity.

---

### B. What happens at higher K?

Check:

* Recall@200
* Recall@500

If hybrid gain increases at higher K → fusion is helping but truncated early.

---

### C. What is NDCG@10?

Sometimes recall improves slightly,
but ranking improves significantly.

Hybrid might improve top ranking quality even if recall gain is small.

---

# 5️⃣ When 2% Is Actually Huge

At 100M scale:

* 2% recall gain could mean thousands of queries fixed.
* For RAG, 2% more grounding can reduce hallucinations noticeably.

In enterprise systems, even +1% recall is valuable.

So:

Technically working ✅
But probably under-optimized ⚠️

---

# 6️⃣ How To Improve It

Here’s what I would test:

### 1️⃣ Increase candidate pool

If you fetch:

* Dense top 100
* Sparse top 100

Try 300–500.

Hybrid needs diversity.

---

### 2️⃣ Try RRF instead of weighted sum

If using α-weighted scoring,
switch to Reciprocal Rank Fusion.

RRF often gives more robust gains.

---

### 3️⃣ Tune BM25 parameters

Adjust:

* k1 (term frequency saturation)
* b (length normalization)

Sparse might be underperforming.

---

### 4️⃣ Improve Dense Recall

Increase:

* HNSW ef_search
* IVF nprobe

Better base recall → better hybrid.

---

# 7️⃣ The Real Answer

Is it working?

Yes.

Is it optimal?

Probably not.

Hybrid systems that are truly tuned usually:

* Show consistent gains across Recall@50, @100
* Improve NDCG@10 noticeably
* Reduce failure cases in numeric/entity queries

---

Now let me test your systems thinking.

If:

Dense = 82%
Sparse = 75%
Hybrid = 81%

What does that imply?

Think carefully before answering.
.

You gave:

* Dense Recall@100 = **82%**
* Sparse Recall@100 = **75%**
* Hybrid Recall@100 = **84%**

Now let’s reason like an infra engineer.

---

# Short Answer

Yes… but **barely**.

It is technically working —
but it is **not working optimally**.

Let’s break this down carefully.

---

# 1️⃣ What Should Hybrid Ideally Do?

In theory, hybrid retrieval should:

$[
Hybrid \ge \max(Dense, Sparse)
]$

That condition is satisfied:

$[
84% > 82%
]$

So it is improving recall.

But here’s the real question:

👉 Is +2% meaningful?

---

# 2️⃣ How Much Gain Should We Expect?

At 100M scale:

* Dense and sparse often retrieve different failure cases.
* Proper hybrid systems often improve **3–8% recall@100**.
* In some domains (legal, finance), even **10%+**.

A +2% gain suggests one of these:

1. Dense and sparse are highly correlated (retrieving same docs)
2. Fusion weights not tuned properly
3. Candidate pool too small (under-fetching)
4. Metadata filtering limiting diversity
5. Sparse not strong enough (BM25 poorly tuned)

---

# 3️⃣ Let’s Think Mathematically

Let:

* D = dense retrieved set
* S = sparse retrieved set

Hybrid recall depends on:

$[
|D ∪ S|
]$

If overlap between D and S is very high:

$[
|D ∩ S| \text{ is large}
]$

Then hybrid cannot improve much.

A +2% gain implies:

Dense and sparse results overlap heavily.

---

# 4️⃣ Diagnostic Questions

To know if it's “really” working, check:

### A. What is the overlap ratio?

$[
Overlap = \frac{|D ∩ S|}{|D|}
]$

If overlap > 70%, sparse adds little diversity.

---

### B. What happens at higher K?

Check:

* Recall@200
* Recall@500

If hybrid gain increases at higher K → fusion is helping but truncated early.

---

### C. What is NDCG@10?

Sometimes recall improves slightly,
but ranking improves significantly.

Hybrid might improve top ranking quality even if recall gain is small.

---

# 5️⃣ When 2% Is Actually Huge

At 100M scale:

* 2% recall gain could mean thousands of queries fixed.
* For RAG, 2% more grounding can reduce hallucinations noticeably.

In enterprise systems, even +1% recall is valuable.

So:

Technically working ✅
But probably under-optimized ⚠️

---

# 6️⃣ How To Improve It

Here’s what I would test:

### 1️⃣ Increase candidate pool

If you fetch:

* Dense top 100
* Sparse top 100

Try 300–500.

Hybrid needs diversity.

---

### 2️⃣ Try RRF instead of weighted sum

If using α-weighted scoring,
switch to Reciprocal Rank Fusion.

RRF often gives more robust gains.

---

### 3️⃣ Tune BM25 parameters

Adjust:

* k1 (term frequency saturation)
* b (length normalization)

Sparse might be underperforming.

---

### 4️⃣ Improve Dense Recall

Increase:

* HNSW ef_search
* IVF nprobe

Better base recall → better hybrid.

---

# 7️⃣ The Real Answer

Is it working?

Yes.

Is it optimal?

Probably not.

Hybrid systems that are truly tuned usually:

* Show consistent gains across Recall@50, @100
* Improve NDCG@10 noticeably
* Reduce failure cases in numeric/entity queries

---

Now let me test your systems thinking.

If:

Dense = 82%
Sparse = 75%
Hybrid = 81%

What does that imply?

Think carefully before answering.


