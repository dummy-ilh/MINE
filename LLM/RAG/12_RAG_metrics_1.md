# 📘 Day 12 — Evaluating Retrieval in RAG (Metrics That Actually Matter)

---

# 1️⃣ First Principle

RAG has two distinct components:

1. Retrieval
2. Generation

If retrieval fails, generation cannot fix it.

So we evaluate them separately.

Today = **retrieval evaluation** only.

---

# 2️⃣ What Does “Good Retrieval” Mean?

Given a query:

> Did we retrieve the documents that contain the answer?

Not:

* Did the final answer sound good?
* Did the LLM phrase it well?

We isolate retrieval.

---

# 3️⃣ You Need Ground Truth

Evaluation requires:

```
Query → Relevant Document(s)
```

You must build a small labeled dataset:

Example:

| Query                  | Relevant Doc IDs |
| ---------------------- | ---------------- |
| Late refund rules      | doc_3            |
| Germany VAT penalty    | doc_8            |
| Refund processing time | doc_1            |

Even 50–200 labeled queries is enough to benchmark models.

Without this:
You cannot evaluate embeddings properly.

---

# 4️⃣ Core Retrieval Metrics

---

## 🔹 1. Recall@k (Most Important)

$[
Recall@k = \frac{\text{Relevant docs in top-k}}{\text{Total relevant docs}}
]$

If correct document appears in top 5 → success.

This answers:

> Did we retrieve the answer at all?

For RAG:
Recall@k matters more than precision.

Because:
LLM cannot use documents it never sees.

---

## 🔹 2. Precision@k

$[
Precision@k = \frac{\text{Relevant docs in top-k}}{k}
]$

If you retrieve 5 docs:

* 4 relevant → good precision
* 1 relevant → noisy retrieval

Important when:

* Token budget is tight
* Noise causes hallucination

---

## 🔹 3. MRR (Mean Reciprocal Rank)

Measures ranking quality.

If correct doc is:

* Rank 1 → score = 1
* Rank 2 → score = 1/2
* Rank 5 → score = 1/5

MRR averages this across queries.

Why it matters:
LLMs focus more on top-ranked chunks.

---

# 5️⃣ Example (Concrete)

Query:

> What happens after 30 days for refunds?

Top 5 retrieved:

1. doc_2 ❌
2. doc_7 ❌
3. doc_3 ✅ (correct)
4. doc_9 ❌
5. doc_1 ❌

Metrics:

* Recall@5 = 1 (correct doc retrieved)
* Precision@5 = 1/5 = 0.2
* MRR = 1/3 ≈ 0.33

Interpretation:

* Retrieval works (recall ok)
* Ranking is weak
* Noise high

---

# 6️⃣ Why Recall@k Is King in RAG

Because generation is conditional.

If recall@5 = 60%:
40% of your answers are doomed before LLM even runs.

Good RAG systems aim for:

* Recall@5 ≥ 85–95%
* Then optimize precision

---

# 7️⃣ Evaluating Hybrid vs Dense

You can compare:

* Dense-only retrieval
* BM25-only retrieval
* Hybrid retrieval

If hybrid increases recall@5 by 10–15%,
that’s massive in production.

This is how you justify architectural changes.

---

# 8️⃣ Retrieval Evaluation Pipeline (Code Sketch)

Pseudo:

```python
correct = 0

for query, relevant_docs in eval_dataset:
    retrieved = retrieve(query, k=5)

    if any(doc in retrieved for doc in relevant_docs):
        correct += 1

recall_at_5 = correct / len(eval_dataset)
```

Then compare across:

* Embedding models
* Chunk sizes
* ANN parameters
* Hybrid strategies

Evaluation drives engineering decisions.

---

# 9️⃣ Common Evaluation Mistakes

❌ Evaluating generation instead of retrieval
❌ Using only 5 queries
❌ No ground-truth labels
❌ Changing embedding model without re-benchmarking
❌ Ignoring ranking position

Most teams skip formal retrieval metrics.
Then they can’t explain why RAG behaves inconsistently.

---

# 🔟 Retrieval vs Generation Evaluation

Important distinction:

Retrieval evaluation:

> Did we fetch the right context?

Generation evaluation:

> Did we use it faithfully?

Both must be measured separately.

Tomorrow we go deeper into generation evaluation (faithfulness, groundedness, RAGAS).

---

# 🧠 Mental Model

Retrieval defines the **upper bound** of RAG accuracy.

If retrieval recall is 80%,
your system accuracy cannot exceed 80%.

Improving generation won’t fix retrieval failure.

---

# 🎯 Interview-Level Answer

If asked:

> “How do you evaluate a RAG system?”

Strong answer:

> “I separate retrieval and generation evaluation. For retrieval, I measure Recall@k, Precision@k, and MRR on a labeled query set. Retrieval recall defines the upper bound of system performance.”

That’s senior-level clarity.

---
