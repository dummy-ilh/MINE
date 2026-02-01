
# 📘 RAG Daily Tutorial

## **Day 5 — Retrieval Strategies: From Basic to Advanced**

---

# 1️⃣ The Naïve Retrieval Strategy (Baseline)

```
query_embedding → top-k similarity → send to LLM
```

Usually:

* k = 3 or 5
* cosine similarity
* no filtering

This works for demos.

It fails in production.

Why?

* Too little context → incomplete answer
* Too much context → hallucination
* Wrong chunk ranked high → misleading answer

---

# 2️⃣ Choosing k (Underrated Design Decision)

### If k is too small:

* Miss critical info
* Multi-hop queries fail

### If k is too large:

* Context overflow
* Noise injection
* Increased hallucination risk

### Better approach:

Instead of fixed k:

### 🔹 Threshold-based retrieval

Retrieve all chunks where:

$[
\text{similarity} > \tau
]$

Or:

### 🔹 Dynamic k

Keep retrieving until:

* similarity drops sharply
* token budget reached

This is more robust than fixed k=5.

---

# 3️⃣ Hybrid Retrieval (Dense + Sparse)

Dense retrieval:

* Good for paraphrasing

Sparse retrieval (BM25):

* Good for exact matches
* Good for numbers
* Good for rare terms

Real systems combine both.

### Scoring method:

$[
Score = \alpha \cdot Dense + (1-\alpha) \cdot Sparse
]$

or

Two independent retrievals → merge results.

Hybrid is almost always better than pure dense.

---

# 4️⃣ Re-Ranking (This Is Where Quality Jumps)

Dense retrieval retrieves candidates.

But similarity search is approximate.

So we apply:

## Cross-Encoder Re-Ranker

Instead of embedding query and doc separately:

Feed both together into a model:

```
$[Query]$ + $[Document]$
→ relevance score
```

This allows:

* Deep semantic comparison
* Token-level interaction

Pipeline:

```
Query
 ↓
Dense retrieve (top 20)
 ↓
Re-rank with cross-encoder
 ↓
Top 5 to LLM
```

This dramatically improves answer grounding.

Cost:

* More compute
* Higher latency

Worth it in high-accuracy systems.

---

# 5️⃣ Query Expansion (Often Ignored)

User query:

> “Late refund rules”

Better retrieval query:

> “Refunds requested after policy deadline manual review process”

How to generate expansions?

* LLM rewriting
* Synonym expansion
* Multi-query retrieval (generate 3 variations)

Then merge results.

This improves recall significantly.

---

# 6️⃣ Multi-Query Retrieval (Advanced but Powerful)

Instead of:

```
Retrieve once
```

Do:

```
Generate 3 reformulations
Retrieve for each
Merge unique chunks
Re-rank
```

Why?
Because embeddings sometimes anchor on different aspects of meaning.

This is used in:

* Enterprise QA
* Legal RAG
* Research retrieval

---

# 7️⃣ Iterative Retrieval (Multi-Hop Questions)

Question:

> “What penalties apply if a refund exceeds the 30-day window in Germany?”

You need:

1. Refund policy
2. Germany-specific clause
3. Penalty subsection

Single retrieval may miss this.

Better approach:

Step 1 → retrieve
Step 2 → refine query using retrieved context
Step 3 → retrieve again

This is primitive multi-hop RAG.

---

# 8️⃣ Retrieval Failure Modes

## ❌ High similarity but irrelevant

Cause:

* Generic chunk
* Large chunk
* Embedding bias

Fix:

* Better chunking
* Re-ranking

---

## ❌ Relevant doc not retrieved

Cause:

* Embedding model mismatch
* Poor query phrasing
* Low ANN recall

Fix:

* Hybrid search
* Query rewriting
* Increase search depth

---

## ❌ Over-retrieval causing hallucination

Cause:

* Too many chunks
* Conflicting info

Fix:

* Aggressive re-ranking
* Context compression

---

# 9️⃣ Retrieval Strategy Ladder (Maturity Model)

Level 1:

* Top-k dense

Level 2:

* Hybrid search

Level 3:

* Dense + re-ranking

Level 4:

* Multi-query + re-ranking

Level 5:

* Iterative retrieval + verification

Most real systems stop at Level 2.

High-performing systems reach Level 4+.

---

# 🔟 Interview-Level Answer

If asked:

> “How would you improve retrieval quality in a RAG system?”

Strong answer:

> “I would move beyond naive top-k dense retrieval by introducing hybrid search for exact term recall, cross-encoder re-ranking for semantic precision, and multi-query expansion to improve recall coverage while maintaining a strict token budget.”

That shows system depth.

---

# 🧠 Mental Model

Retrieval has two goals:

* Maximize recall (don’t miss relevant info)
* Maximize precision (don’t send noise to LLM)

RAG quality depends more on retrieval strategy than model size.

---

Perfect.
Retrieval strategy is where most senior interviews separate “RAG user” from “RAG architect.”

I’ll give you:

1. 🧠 Concept Build-Up
2. 🎯 Retrieval Strategy Interview Questions (with strong answers)
3. 📌 Retrieval Strategy Cheatsheet (condensed high-impact reference)

This will cover hybrid search, query rewriting, multi-stage retrieval, routing, fusion, and evaluation.

---

# 🧠 PART 1: Concept Build-Up — What Is a Retrieval Strategy?

A retrieval strategy answers:

> Given a query and a corpus, how do I efficiently retrieve the most relevant, diverse, grounded information?

It is not just:

* “Use vector search.”

It includes:

* Query rewriting
* Metadata filtering
* Shard routing
* ANN search
* Hybrid search
* Reranking
* Context assembly
* Failure handling

Retrieval strategy = **decision logic over retrieval components**.

---

# 🎯 PART 2: Retrieval Strategy — Interview Q&A

---

## Q1️⃣ What is a retrieval strategy in a RAG system?

**Answer:**

A retrieval strategy is the structured pipeline that determines:

* How queries are processed
* Which retrieval mechanisms are used
* How candidates are filtered, ranked, and assembled
* How relevance vs latency tradeoffs are managed

It optimizes for:

* Recall
* Precision
* Latency
* Cost
* Faithfulness

---

## Q2️⃣ When would you use hybrid search?

Hybrid search combines:

* Sparse retrieval (BM25)
* Dense retrieval (vector similarity)

Use hybrid when:

* Exact keyword matching matters
* Rare terminology exists
* Legal or technical terms must match exactly
* Embeddings underperform for specific jargon

Example:
Query: “Section 409A compliance clause”

Vector may miss exact clause reference.
BM25 retrieves exact phrase.

Hybrid = best of both worlds.

---

## Q3️⃣ How do you combine sparse and dense scores?

Common methods:

1. Weighted sum
2. Reciprocal Rank Fusion (RRF)
3. Two-stage reranking

RRF is popular:

Score = Σ (1 / (k + rank_i))

It’s robust and simple.

---

## Q4️⃣ How would you design retrieval for multi-hop reasoning?

Problem:
Query requires combining multiple documents.

Example:
“What changed in policy between 2022 and 2023?”

Strategy:

* Retrieve docs for 2022
* Retrieve docs for 2023
* Compare sections
* Expand graph neighbors

Techniques:

* Graph-based expansion
* Iterative retrieval
* Self-ask prompting

---

## Q5️⃣ How do you handle ambiguous queries?

Example:
“Vacation policy”

Strategy:

* Query rewriting:

  * Expand with synonyms
  * Add likely entity types
* Intent classification
* Ask clarifying question
* Retrieve top clusters and disambiguate

---

## Q6️⃣ What is query rewriting and why use it?

Query rewriting transforms user queries into:

* More searchable form
* Expanded terms
* Clarified intent

Techniques:

* LLM-based expansion
* Synonym injection
* Entity normalization
* Time constraint inference

Example:
User: “WFH rules”
Rewrite:
“Work from home policy eligibility rules”

Improves recall.

---

## Q7️⃣ When would you use multi-stage retrieval?

When:

* Corpus > 1M documents
* Latency constraints exist
* Precision is critical

Pipeline:

```
Metadata filter
 → ANN top 100
 → Reranker top 10
 → Diversity filter top 5
 → LLM
```

This reduces hallucination.

---

## Q8️⃣ How do you reduce retrieval noise?

* Strong metadata filtering
* Smaller chunk sizes
* Better embeddings
* Reranking
* Remove near-duplicates
* Namespace isolation

Noise reduction improves:

* Faithfulness
* Token efficiency

---

## Q9️⃣ How do you handle time-sensitive queries?

Add time-awareness:

* Index by timestamp
* Boost recent documents
* Filter by date
* Decay older content

This prevents outdated answers.

---

## Q🔟 How do you evaluate retrieval strategy?

Metrics:

Retrieval:

* Recall@K
* Precision@K
* NDCG

Generation:

* Faithfulness
* Citation correctness

System:

* P95 latency
* QPS

You must measure:
Recall vs latency tradeoff.

---

# 📌 PART 3: RETRIEVAL STRATEGY CHEATSHEET

This is your high-impact condensed reference.

---

## 🔷 Retrieval Types

| Strategy     | When to Use                 |
| ------------ | --------------------------- |
| Pure Dense   | Semantic similarity queries |
| Pure Sparse  | Exact term queries          |
| Hybrid       | Technical/legal queries     |
| Graph-based  | Multi-hop reasoning         |
| Hierarchical | Structured documents        |
| Query-aware  | Mixed query types           |

---

## 🔷 Query Processing Techniques

* Intent classification
* Query rewriting
* Expansion
* Normalization
* Time inference
* Entity extraction

---

## 🔷 Retrieval Pipeline Template

Basic:

```
Query → Embedding → ANN → LLM
```

Intermediate:

```
Query → Rewrite → Metadata Filter → ANN → Rerank → LLM
```

Advanced:

```
Query → Intent Classifier
       → Rewrite
       → Shard Routing
       → Hybrid Search
       → ANN Top 100
       → Cross-Encoder Rerank
       → Diversity Filter
       → Context Assembly
       → LLM
```

---

## 🔷 Optimization Levers

| Lever              | Improves            |
| ------------------ | ------------------- |
| Smaller chunk size | Recall              |
| Reranker           | Precision           |
| Metadata filtering | Latency + Precision |
| Query rewriting    | Recall              |
| Diversity filter   | Context quality     |
| Sharding           | Scalability         |
| Caching            | Latency             |

---

## 🔷 Common Failure Modes

| Problem              | Likely Cause      |
| -------------------- | ----------------- |
| Wrong doc retrieved  | Poor embedding    |
| Right doc ignored    | Bad chunking      |
| Too slow             | efSearch too high |
| Inconsistent answers | ANN randomness    |
| Outdated answer      | No time filter    |

---

## 🔷 Staff-Level Insight

Retrieval strategy is not static.

It should adapt to:

* Query intent
* Corpus size
* Latency budget
* Risk tolerance
* Compliance level

Elite systems dynamically change strategy per query.

---

# Final Interview-Level Summary

Retrieval strategy is the art of balancing:

Recall
Precision
Latency
Cost
Scalability
Faithfulness

If you optimize only one, you degrade others.

---

If you want next:

* ⚔️ Hybrid Search Deep Dive
* 🧠 Query Routing Architectures
* 🏗️ Billion-scale Retrieval System Design
* 📉 Retrieval-Induced Hallucination Analysis
* 🔬 End-to-End RAG Evaluation Framework

Choose the next depth.

