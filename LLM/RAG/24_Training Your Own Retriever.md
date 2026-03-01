

# 🧠 Training Your Own Retriever (From Theory to Practice)

Today we move from **using embeddings** to **training retrieval models properly**.

This is where serious RAG systems differentiate.

---

# 1️⃣ Why Fine-Tune a Retriever?

Pretrained embedding models (e.g., OpenAI, open-source sentence transformers) are trained on:

* General web data
* QA datasets
* Mixed domains

But your RAG system might operate on:

* Legal contracts
* Medical reports
* Financial filings
* Internal company emails
* Research PDFs

Domain mismatch = poor retrieval.

Even if Recall@20 looks decent, relevance quality suffers.

---

# 2️⃣ Retrieval Training Objective

Dense retrievers are usually trained with **contrastive learning**.

Given:

* Query ( q )
* Positive document ( d^+ )
* Negative documents ( d^- )

Goal:

[
\text{sim}(q, d^+) > \text{sim}(q, d^-)
]

Loss commonly used:

### InfoNCE Loss

[
L = -\log \frac{e^{sim(q,d^+)}}{e^{sim(q,d^+)} + \sum e^{sim(q,d^-)}}
]

Interpretation:

Push positives closer
Push negatives farther

---

# 3️⃣ Types of Negatives (CRUCIAL)

This is where mastery lies.

### 🟢 Easy Negatives

Random documents
→ Too easy, low learning signal

### 🟡 In-Batch Negatives

Other positives in same batch
→ Cheap and effective

### 🔴 Hard Negatives

Documents semantically similar but wrong
→ Massive improvement in retrieval quality

Hard negatives are the key to state-of-the-art retrievers.

---

# 4️⃣ Hard Negative Mining Strategies

### 1️⃣ BM25 Hard Negatives

Use sparse retrieval to get near-miss docs.

Often effective.

### 2️⃣ Dense Hard Negatives

Use pretrained retriever → take top incorrect hits.

Better signal.

### 3️⃣ Cross-Encoder Mining

Use cross-encoder to find misleading high-score negatives.

Very strong training signal.

---

# 5️⃣ Architecture Choices

## Option A: Bi-Encoder (Standard Dense)

```text
Query → Encoder
Doc → Encoder
Cosine similarity
```

Fast inference
Works well in production

---

## Option B: Dual-Encoder with Shared Weights

Query encoder = Doc encoder

Memory efficient
Common in sentence-transformers

---

## Option C: Asymmetric Encoder

Different encoders for:

* Short queries
* Long documents

Better when distributions differ significantly.

---

# 6️⃣ Training Pipeline (Practical)

Step-by-step production-ready pipeline:

1. Collect query-document pairs
2. Generate hard negatives
3. Format into triplets:

   ```
   (query, positive, negative)
   ```
4. Train using contrastive loss
5. Periodically evaluate on:

   * Recall@k
   * MRR
   * nDCG
6. Freeze model
7. Re-index full corpus

---

# 7️⃣ Evaluation Metrics (Research-Level)

### Recall@k

Did we retrieve relevant document in top k?

### MRR (Mean Reciprocal Rank)

Rewards earlier ranking.

### nDCG

Accounts for graded relevance.

---

# 8️⃣ When Fine-Tuning Helps Most

Fine-tuning gives biggest lift when:

* Domain terminology is specialized
* Queries are short and ambiguous
* Documents are long
* Retrieval confusion is common

In general web corpora → small gains
In specialized corpora → huge gains

---

# 9️⃣ Domain Adaptation vs Full Fine-Tuning

Two approaches:

### Light Adaptation

* Continue training with small learning rate
* Preserve general semantic space

### Full Fine-Tuning

* Retrain heavily on domain data
* Risk catastrophic forgetting

Best practice:
Start light.

---

# 🔥 Real Production Insight

Many teams:

* Use strong pretrained model
* Add hybrid search
* Add reranker
* Stop there

Why?

Because retriever training requires:

* Labeled data
* Negative mining pipeline
* Evaluation framework
* Infra support

But if you want research-level impact → you train your own.

---

# 🧠 Deep Insight

Retrieval quality depends more on:

> Negative sampling strategy

than on model architecture.

People obsess over model size.
Experts obsess over negatives.

---

# 🧪 Interview-Level Questions

Answer carefully:

1. Why do easy negatives fail to improve retriever quality?
2. Why can hard negatives sometimes hurt if overused?
3. What happens if positives are noisy?
4. How does retrieval training differ from classification training?

Think like a retrieval researcher.

---

