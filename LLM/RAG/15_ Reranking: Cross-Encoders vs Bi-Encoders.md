# 🚀 RAG Mastery – Day 15

# ⚖️ Reranking: Cross-Encoders vs Bi-Encoders

Up to now, your pipeline looks like:

```
Query → Hybrid Retrieval → Top-K Documents
```

But here’s the reality:

> Retrieval optimizes **recall**.
> Reranking optimizes **precision**.



---

# 🧠 Why Reranking Is Needed

Vector search is approximate:

* Embedding similarity ≠ true relevance
* Hybrid search still noisy
* Long documents dilute meaning
* Multi-topic chunks get high scores

So instead of sending noisy context to the LLM, we re-score more precisely.

---

# 1️⃣ Bi-Encoder (What You’ve Been Using)

Dense retrievers use a **bi-encoder** architecture:

```
Query → Encoder A → Vector
Doc   → Encoder B → Vector

Similarity = cosine(query_vec, doc_vec)
```

Fast because:

* Documents are pre-embedded
* Only one query encoding needed
* Scales to millions of docs

But limitation:

* Query and doc encoded independently
* No deep token-level interaction

---

# 2️⃣ Cross-Encoder (Precision Machine)

Cross-encoders take:

```
$[Query + Document]$ → Transformer → Relevance Score
```

Now the model sees full interaction:

* Token-by-token attention
* Phrase matching
* Negations
* Subtle context shifts

This dramatically improves ranking.

---

# 🔬 Example

Query:

> Does rate limiting reduce API latency?

Document A:

> Rate limiting protects systems from overload.

Document B:

> Rate limiting can increase latency under heavy load.

Dense retrieval may score both similarly.

Cross-encoder understands:

* “reduce latency” vs “increase latency”
* Directional meaning

---

# 3️⃣ Architecture with Reranker

```
Query
   ↓
Hybrid Retrieve (top 50)
   ↓
Cross-Encoder Rerank
   ↓
Top 5
   ↓
LLM
```

This is modern production RAG.

Used by:

* Perplexity AI
* Google
* Microsoft search stacks

---

# 4️⃣ Performance Tradeoff

| Model Type    | Speed       | Accuracy | Use Case          |
| ------------- | ----------- | -------- | ----------------- |
| Bi-Encoder    | ⚡ Very Fast | Medium   | Initial retrieval |
| Cross-Encoder | 🐢 Slower   | High     | Final rerank      |

Cross-encoder complexity:

If you retrieve 50 docs → you must run 50 forward passes.

So latency scales linearly.

---

# 5️⃣ Popular Rerank Models

* Cohere rerank models
* SentenceTransformers cross-encoders
* BERT-based cross-attention models
* OpenAI reranking APIs

Many production systems use lightweight cross-encoders fine-tuned on relevance data.

---

# 6️⃣ Code Example (SentenceTransformers)

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = $[(query, doc) for doc in retrieved_docs]$

scores = model.predict(pairs)

reranked = sorted(zip(retrieved_docs, scores),
                  key=lambda x: x$[1]$,
                  reverse=True)

top_docs = $[doc for doc, score in reranked$[:5]$]$
```

---

# 7️⃣ Latency Optimization Strategies

## ✅ Strategy 1: Limit candidate pool

Retrieve 20–50 docs only.

## ✅ Strategy 2: Use smaller cross-encoders

MiniLM-based models are efficient.

## ✅ Strategy 3: Distillation

Train a smaller reranker on outputs of a large reranker.

## ✅ Strategy 4: GPU batching

Batch query-doc pairs.

---

# 8️⃣ When Reranking Helps Most

Reranking gives massive gains when:

* Documents are long
* Corpus is noisy
* Legal/financial domain
* Multi-hop questions
* Overlapping topics

Less useful when:

* Small clean dataset
* FAQ-style corpus
* Very short chunks

---

# 9️⃣ Reranking vs Increasing K

You might think:

> Why not just send top 20 docs to LLM?

Because:

* Context window cost explodes
* LLM attention gets diluted
* Hallucination risk increases
* Token cost ↑

Better to send:

* Top 3–7 high precision chunks

---

# 🧠 Deep Insight

Think in IR terms:

Dense Retrieval → High Recall
Hybrid Retrieval → Balanced Recall
Cross-Encoder → High Precision

Together:

```
Recall first
Precision second
Generation last
```

This ordering is fundamental.

---

# 🔥 Advanced Insight

Many top-tier systems now use:

```
Hybrid Retrieve (k=100)
↓
Cross-Encoder Rerank
↓
LLM-based Rerank (optional)
↓
Top 5
```

Yes — sometimes LLMs themselves are used as relevance judges.

---


Now we’re moving into the **real quality engine** of retrieval systems.

ANN retrieval finds *candidates*.
**Reranking decides what actually matters.**

Let’s go deep — architect level.

---

# 🎯 Why Reranking Exists

Dense retrieval (HNSW / IVF) optimizes:

$[
\text{Vector similarity}
]$

But vector similarity ≠ true relevance.

ANN is:

* Approximate
* Bi-encoder based (query & doc encoded separately)
* Fast but shallow

Reranking is:

* Exact
* Cross-attention based
* Slow but powerful

---

# 🧠 Core Idea

Instead of:

```
score = cosine(query_emb, doc_emb)
```

We compute:

```
score = relevance(query, doc) using joint encoding
```

The key difference:

| Bi-encoder        | Cross-encoder   |
| ----------------- | --------------- |
| Encode separately | Encode together |
| Fast              | Expensive       |
| Approximate       | Precise         |

---

# 🔬 Under the Hood: Cross-Encoder Reranking

Let’s say:

Query:

> “penalties under section 498A”

Document:

> “Section 498A IPC describes cruelty by husband and related punishment…”

---

## Step 1 — Concatenation

Model input:

```
$[CLS]$ query tokens $[SEP]$ document tokens $[SEP]$
```

Now the transformer sees BOTH together.

---

## Step 2 — Cross-Attention

Unlike bi-encoder:

* Query tokens attend to document tokens
* Document tokens attend to query tokens

This enables:

* Exact term matching
* Negation understanding
* Context sensitivity
* Numerical reasoning

---

## Step 3 — Output Score

Model outputs:

$[
P(relevant | query, document)
]$

Often a single scalar.

---

# ⚙️ Pipeline in Production

For 100M documents:

```
ANN → top 200 candidates
        ↓
Cross-encoder rerank
        ↓
Top 10 returned
```

We never rerank all 100M.
Only small candidate set.

---

# 🔥 Why It Works So Well

Dense embeddings compress meaning into fixed vector.

But compression loses:

* Fine-grained word interactions
* Rare entity specificity
* Logical structure

Cross-encoder restores those interactions.

---

# 📊 Measurable Effect

Typical improvements:

| Metric  | Before | After |
| ------- | ------ | ----- |
| MRR@10  | 0.62   | 0.71  |
| NDCG@10 | 0.68   | 0.76  |

Huge gain at top ranks.

---

# 🏗️ Architecture Variants

## 1️⃣ MonoT5

Treat reranking as text-to-text:

```
Input: Query + Doc
Output: "relevant" or "not relevant"
```

---

## 2️⃣ BERT Cross-Encoder

Output scalar relevance score.

Most common approach.

---

## 3️⃣ Late Interaction Models (ColBERT-style)

Hybrid approach:

* Token-level embeddings
* MaxSim aggregation
* Faster than full cross-encoder

Used when:

* Want better accuracy than bi-encoder
* But cheaper than full cross-encoder

---

# ⏱ Latency Consideration

If:

* 200 candidates
* Each inference 3ms on GPU

Total ≈ 600ms ❌ too slow

So we:

* Batch process
* Use smaller model (MiniLM)
* Reduce candidates to 50–100

Target:

* 20–50ms reranking latency

---

# 🎯 What Reranker Actually Learns

It learns:

* Query-document semantic alignment
* Entity match importance
* Field weighting
* Phrase importance
* Negation patterns
* Answer-bearing signals

It is trained on:

* Click logs
* Relevance labels
* Pairwise ranking loss
* Listwise ranking loss

---

# 🧪 Training Objective (Behind the Scenes)

Common loss:

### Pairwise Loss

For relevant doc ( d^+ ) and irrelevant ( d^- ):

$[
L = \max(0, 1 - s(q,d^+) + s(q,d^-))
]$

Force relevant > irrelevant.

---

# 💡 Why Not Use Cross-Encoder Directly on All Docs?

Complexity:

$[
O(N × Transformer)
]$

Impossible at 100M scale.

ANN reduces to:

$[
O(200 × Transformer)
]$

---

# 🧠 Mental Model

Think of retrieval as:

Stage 1 — Candidate Generator
Stage 2 — Precision Refiner

ANN maximizes recall.
Reranker maximizes precision.

---

# 🚨 Common Mistakes

❌ Reranking only top 10
❌ Using too small candidate pool
❌ Not batching GPU inference
❌ Using reranker without evaluation
❌ Training reranker on biased click data

---

# 🔍 Reranking in Hybrid Systems

Hybrid gives broader candidate pool.

Reranker then:

* Resolves sparse vs dense conflicts
* Picks best ordering
* Fixes lexical noise

Hybrid + reranker together are multiplicative.

---

# 🧠 Deep Insight

Dense retrieval approximates:

$[
argmax_d f(q)^T f(d)
]$

Cross-encoder approximates:

$[
argmax_d g(q, d)
]$

Where:

* f = compressed representation
* g = full interaction function

g is strictly more expressive than f.

---

# 🧪 How to Measure Reranker Quality

Evaluate:

* MRR@10
* NDCG@10
* Delta improvement over base retriever

If improvement < 3%:

* Candidate pool weak
* Model too small
* Domain mismatch

---

# 🔥 Production Blueprint

For 100M docs:

* ANN (Recall@200 ≥ 90%)
* Hybrid fusion
* Rerank top 100
* Return top 10
* Feed to LLM

---

Correct ✅ — it’s **good**.

Now let’s explain *why*, at a systems level.

---

# 🎯 If MRR Improves but Recall@100 Stays the Same

That means:

* The correct documents were **already in the top 100**
* The reranker just moved them **higher**

So:

Recall measures:

> *Did we retrieve the relevant doc anywhere in top K?*

MRR measures:

> *How early did we retrieve it?*

Reranking **does not increase recall**
It improves **ordering quality**.

---

# 🔍 Why That’s Excellent

For:

* Search systems
* QA systems
* RAG pipelines

Users care about:

* Top 1–10 results
* Not top 100

If the relevant doc moves:

From rank 42 → rank 2

That’s a massive improvement in user experience.

---

# 🧠 In RAG Context

If relevant doc is:

* Rank 75 → LLM might not see it
* Rank 3 → LLM definitely sees it

Even though Recall@100 unchanged.

So reranker improves:

* Grounding
* Faithfulness
* Answer accuracy
* Hallucination reduction

---

# 📊 Concrete Example

Before reranking:

| Rank | Relevant? |
| ---- | --------- |
| 1    | ❌         |
| 2    | ❌         |
| 3    | ❌         |
| 4    | ❌         |
| 5    | ❌         |
| 42   | ✅         |

MRR = 1/42 ≈ 0.023

After reranking:

| Rank | Relevant? |
| ---- | --------- |
| 1    | ❌         |
| 2    | ✅         |

MRR = 1/2 = 0.5

Huge gain.

Recall@100 = same.

---

# 🚀 What This Tells You About Your System

It means:

1. Retriever has decent recall
2. Candidate generation working
3. Reranker doing its job

If reranker improves MRR significantly,
your stage-1 retrieval is good enough.

---

# 🔥 When It Would Be Bad

If:

Recall@100 = low (say 60%)
Reranker improves MRR

That means:
You are polishing bad candidates.

No reranker can fix missing recall.

---

# 🧠 The Core Principle

Stage 1 → maximize recall
Stage 2 → maximize precision

If reranker improves MRR but recall unchanged,
your pipeline architecture is correct.

---



