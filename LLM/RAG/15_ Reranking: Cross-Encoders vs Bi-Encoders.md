# 🚀 RAG Mastery – Day 15

# ⚖️ Reranking: Cross-Encoders vs Bi-Encoders

Up to now, your pipeline looks like:

```
Query → Hybrid Retrieval → Top-K Documents
```

But here’s the reality:

> Retrieval optimizes **recall**.
> Reranking optimizes **precision**.

Day 15 is about making your top 20 results become the **correct top 5**.

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
[Query + Document] → Transformer → Relevance Score
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

pairs = [(query, doc) for doc in retrieved_docs]

scores = model.predict(pairs)

reranked = sorted(zip(retrieved_docs, scores),
                  key=lambda x: x[1],
                  reverse=True)

top_docs = [doc for doc, score in reranked[:5]]
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

# 🧪 Exercise for Today

1. Add cross-encoder reranking.
2. Compare:

   * Dense only
   * Hybrid only
   * Hybrid + rerank
3. Measure:

   * Recall@5
   * MRR
   * Latency
4. Analyze failure cases.

---

# 🧠 Critical Thinking

Why not replace dense retrieval entirely with cross-encoders?

Answer carefully — this question separates beginners from system designers.

---

# 🎯 Tomorrow (Day 16)

We go into:

> 📏 RAG Evaluation — Measuring Groundedness, Faithfulness, and Retrieval Quality

This is where engineers become researchers.

You’re now building serious search systems.
