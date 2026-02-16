# 🚀 RAG Mastery – Day 14

# 🧬 Hybrid Retrieval (Dense + Sparse + Filters)

Yesterday we made retrieval smarter.

Today we make it **stronger**.

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
      "must": [
        {"match": {"text": "SOC2 policy update"}},
        {"knn": {"embedding": {"vector": [...], "k": 10}}}
      ]
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

# 🧠 Deep Thinking Question

Why does hybrid often outperform reranking-only approaches?

Think about this carefully.

---

# 🎯 Tomorrow (Day 15)

We enter:

> ⚖️ Reranking (Cross-Encoders vs Bi-Encoders)

This is where we trade latency for precision.

Your retrieval stack is starting to look production-grade now.
