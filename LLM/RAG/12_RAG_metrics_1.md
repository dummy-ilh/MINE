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



# 📏 RAG Evaluation: Measuring What Actually Matters

Until now, we’ve engineered retrieval pipelines.

Today we answer the hardest question:

> **How do you know your RAG system is actually good?**

If you cannot measure it, you cannot improve it.

Modern RAG evaluation has **three layers**:

```
1️⃣ Retrieval Quality
2️⃣ Generation Quality
3️⃣ End-to-End Faithfulness
```

Let’s go deep.

---

# 🧱 Layer 1: Retrieval Evaluation

Before judging answers, judge retrieval.

Because:

> If the right document is not retrieved, the LLM never had a chance.

---

## Core Retrieval Metrics

### 1️⃣ Recall@K

**Definition:**
Did we retrieve at least one relevant document in top-K?

```
Recall@5 = % of queries where at least one correct doc is in top 5
```

This is the most important retrieval metric.

---

### 2️⃣ Precision@K

How many of top-K are relevant?

More useful in reranking evaluation.

---

### 3️⃣ MRR (Mean Reciprocal Rank)

Measures how high the first relevant doc appears.

```
MRR = average(1 / rank_of_first_relevant_doc)
```

Higher = better ranking quality.

---

## ⚠️ Important Insight

For RAG systems:

> High Recall@K matters more than high Precision@K
> because rerankers + LLM can fix precision later.

---

# 🧠 Layer 2: Generation Evaluation

Now assume correct docs were retrieved.

We evaluate:

* Is the answer correct?
* Is it grounded in context?
* Is it hallucinated?

---

## 1️⃣ Exact Match (EM)

Used in QA datasets.

Very strict — not ideal for generative systems.

---

## 2️⃣ F1 Score

Measures token overlap between generated answer and reference.

Still brittle for long-form answers.

---

## 3️⃣ LLM-as-Judge (Modern Approach)

Use an LLM to evaluate:

* Correctness
* Completeness
* Faithfulness
* Relevance

This is what:

* OpenAI eval pipelines use
* Anthropic research teams use
* Google Gemini eval pipelines use

---

# 🧬 Layer 3: Faithfulness / Groundedness

This is RAG-specific.

We care about:

> Did the model use only retrieved context?

This is where hallucination detection comes in.

---

## 1️⃣ Context Precision

Does the answer contain unsupported claims?

Approach:

* Break answer into atomic statements
* Check if each statement is supported in retrieved text

---

## 2️⃣ Attribution Score

How well does answer cite evidence?

Used heavily in:

* Perplexity AI

---

## 3️⃣ RAGAS Framework

Ragas provides:

* Faithfulness
* Answer relevance
* Context precision
* Context recall

It uses LLM-based scoring.

---

# 🏗 Full Evaluation Architecture

```
Dataset (Query, Gold Answer, Gold Docs)
        ↓
Run RAG Pipeline
        ↓
Compute:

Retrieval:
- Recall@K
- MRR

Generation:
- LLM correctness score

Faithfulness:
- Statement grounding check
```

---

# 🔬 Building a Proper Eval Dataset

You need:

For each query:

* Ground truth answer
* Ground truth supporting documents

Without this, you are guessing.

In enterprise systems, teams manually annotate:

* 100–500 queries
* High-quality ground truth labels

This is your benchmark suite.

---

# ⚖️ Offline vs Online Evaluation

### Offline

* Controlled dataset
* Reproducible
* Used for model comparison

### Online (A/B testing)

* Real users
* Measure:

  * Click-through rate
  * User satisfaction
  * Resolution rate

Large companies rely heavily on online eval.

---

# 🧠 Failure Taxonomy (Extremely Important)

When RAG fails, classify:

### 1️⃣ Retrieval Failure

Correct doc not retrieved.

Fix:

* Hybrid retrieval
* Better chunking
* Query rewriting

---

### 2️⃣ Ranking Failure

Correct doc retrieved but ranked low.

Fix:

* Reranker
* Better scoring

---

### 3️⃣ Generation Failure

Correct doc retrieved but LLM ignored it.

Fix:

* Better prompting
* Stronger grounding instructions

---

### 4️⃣ Hallucination

Answer not supported by context.

Fix:

* Smaller context
* Faithfulness constraints
* Citation enforcement

---

# 🧪 Example Eval Loop (Pseudo-Code)

```python
for query in eval_dataset:
    retrieved_docs = retrieve(query)
    answer = generate(query, retrieved_docs)

    retrieval_score = compute_recall(query, retrieved_docs)
    faithfulness_score = llm_judge(answer, retrieved_docs)
    correctness_score = llm_compare(answer, ground_truth)
```

Aggregate metrics at end.

---

# 📊 What Production Teams Track

Serious RAG systems track:

* Recall@5
* Recall@20
* MRR
* Faithfulness score
* Hallucination rate
* Latency
* Cost per query

Evaluation is multi-dimensional.

---

# 🧠 Deep Insight

In RAG:

> Retrieval quality sets the ceiling.
> Generation quality determines how close you get to that ceiling.

You cannot prompt-engineer your way out of bad retrieval.

---

# 🔥 Research Direction

Recent work explores:

* LLM-based retrieval evaluation
* Synthetic test set generation
* Adversarial queries
* Robustness to query reformulation

This is active research.

---

# 🧪 Today’s Exercise

1. Build small evaluation dataset (20–50 queries).
2. Implement:

   * Recall@5
   * MRR
3. Add LLM-as-judge scoring.
4. Classify failures into:

   * Retrieval
   * Ranking
   * Generation

You now move from engineer → researcher.

