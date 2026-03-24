# 📘 Day 16 — Re-Ranking & Compression

**Theme:** Precision > Quantity

By Day 15, your system retrieves *more intelligently*.
Today, we make it retrieve *more precisely*.

Most RAG systems fail not because they retrieve too little —
but because they pass too much noisy context to the LLM.

We fix that with:

1. Cross-encoder re-ranking
2. Contextual compression
3. Passage selection

---

# 🧠 The Core Problem

Dense retrieval (bi-encoder embeddings):

```
embed(query)
embed(document_chunk)
cosine_similarity
```

This is fast but approximate.

It cannot deeply evaluate semantic alignment between full query and full passage.

So we introduce a second stage.

---

# 1️⃣ Cross-Encoder Re-Ranking

## 🧩 Bi-Encoder vs Cross-Encoder

### Bi-Encoder (Embedding Model)

* Encode query and doc separately
* Compare vectors
* Fast
* Approximate

### Cross-Encoder

* Concatenate query + passage
* Feed into transformer
* Directly score relevance

Much slower.
Much more accurate.

---

## 🧠 Why It Works

Instead of:

```
sim(E(q), E(d))
```

We compute:

```
Score(q, d) using full attention
```

The model can attend:

* Query term → exact phrase in passage
* Negations
* Subtle relationships
* Context alignment

---

## 🔧 Minimal Example (HuggingFace Style)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, passages):
    scores = []

    for passage in passages:
        inputs = tokenizer(query, passage, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model(**inputs)
        score = output.logits.item()
        scores.append((passage, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

## ⚠️ Tradeoff

| Property    | Bi-Encoder | Cross-Encoder |
| ----------- | ---------- | ------------- |
| Speed       | Fast       | Slow          |
| Cost        | Low        | Higher        |
| Accuracy    | Medium     | High          |
| Scalability | Massive    | Limited       |

So production pattern:

```
Retrieve top 20 (bi-encoder)
Re-rank top 20
Select top 5
```

Two-stage retrieval.

This is industry standard.

---

# 2️⃣ Contextual Compression

Even after reranking, chunks may contain irrelevant sections.

Instead of passing full chunk → compress it relative to query.

---

## 🧠 Concept

Given:

Query:

> “How does RAG handle token budget overflow?”

Chunk:

```
RAG systems often face latency issues...
...
Token budgets must be managed carefully...
...
Vector databases scale horizontally...
```

We only need the middle section.

---

## 🔧 LLM-Based Compression

```python
def compress_chunk(client, query, chunk):
    prompt = f"""
Given the query and passage, extract ONLY the relevant portion.

Query:
{query}

Passage:
{chunk}

Relevant Extract:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

Now you reduce:

* Token usage
* Noise
* Hallucination risk

---

## ⚠️ Risk

Compression must preserve factual accuracy.

You must measure:

* Answer correctness
* Context faithfulness

---

# 3️⃣ Passage Selection (Fine-Grained Retrieval)

Instead of chunk-level retrieval:

* Retrieve large chunk
* Split into sentences
* Score sentences
* Keep best ones

---

## Minimal Sentence-Level Selection

```python
import nltk

def select_passages(query, chunk, reranker):
    sentences = nltk.sent_tokenize(chunk)
    ranked = rerank(query, sentences)
    return [s for s, _ in ranked[:3]]
```

Now context is:

* Dense
* Precise
* Minimal

This dramatically improves grounding.

---

# 🔄 Updated Week 3 Pipeline

```
User Query
   ↓
Intent Detection
   ↓
Query Rewrite
   ↓
Multi-Query Retrieval
   ↓
Dense Retrieval (top 20)
   ↓
Cross-Encoder Re-rank
   ↓
Passage Selection
   ↓
Contextual Compression
   ↓
Token-Aware Context Builder
   ↓
LLM Answer
```

This is no longer beginner RAG.

This is robust RAG.

---

# ⚡ Where Systems Break Here

1. Latency explodes (cross-encoder too slow)
2. Cost increases (extra LLM calls)
3. Token budget miscalculated
4. Over-compression loses key facts
5. No observability for ranking stage

This is why Day 14 logging was critical.

---

# 🧠 Performance Engineering Insight

Latency profile often becomes:

* 40% embedding
* 30% reranking
* 25% LLM answer
* 5% glue logic

Most people assume LLM is slowest.

Not always true.

---

# 🎯 Interview-Level Answers

## Why use cross-encoder if embeddings exist?

Because embeddings approximate similarity in vector space.
Cross-encoders evaluate full semantic interaction.

They reduce false positives in top-k.

---

## When would you skip reranking?

* Small corpus
* Latency-critical systems
* Low-stakes retrieval
* When embedding model is already very strong

---

## How do you evaluate reranking impact?

* Measure Recall@k
* Measure MRR (Mean Reciprocal Rank)
* Compare answer correctness
* Measure token reduction %

---

## What is contextual compression’s main benefit?

It reduces irrelevant tokens while preserving relevance, improving grounding and reducing hallucination risk.

---

# 🧠 Architect-Level Takeaway

There are 3 precision levers in RAG:

1. Query quality
2. Retrieval ranking
3. Context density

Week 1 systems optimize none.
Week 2 systems optimize 1.
Week 3 systems optimize all 3.

---

You now understand:

* Two-stage retrieval
* Precision optimization
* Context minimization
* Latency tradeoffs
* Ranking evaluation

This is where most engineers stop.

From here, we enter:

* Hybrid search (BM25 + dense)
* Adaptive top-k
* Dynamic routing
* Hierarchical retrieval
* Agentic retrieval




