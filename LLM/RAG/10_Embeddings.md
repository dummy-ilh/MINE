Excellent.
Today we move from *“how embeddings work”* to **“which embeddings should you use and why.”**

This is where cost, performance, and domain expertise collide.

---

# 📘 Day 10 — Embedding Models: Selection, Tradeoffs, and Evaluation

You already know:

* Embeddings map text → vectors
* Similar vectors → similar meaning

Today we answer:

> Which embedding model should I use in a real RAG system?

---

# 1️⃣ Embedding Model Landscape

There are three main categories:

---

## 🔹 1. API-Based Embeddings (Hosted)

Examples:

* OpenAI `text-embedding-3-small`
* OpenAI `text-embedding-3-large`
* Cohere embeddings

### Pros:

* High quality
* No infra to manage
* Well-optimized

### Cons:

* Ongoing cost
* Data leaves your environment
* Less customization

Best for:

* Startups
* MVPs
* Fast iteration

---

## 🔹 2. Open-Source Embeddings (Self-Hosted)

Examples:

* `all-MiniLM-L6-v2`
* `bge-large`
* `e5-large`
* `Instructor-xl`

Used via:

* SentenceTransformers
* HuggingFace

### Pros:

* No per-call cost
* Full control
* On-prem deployment

### Cons:

* Need GPU infra
* Must manage scaling
* Quality varies by domain

Best for:

* Enterprise
* Privacy-sensitive use cases
* High-volume systems

---

## 🔹 3. Domain-Specific Embeddings

Examples:

* BioBERT (medical)
* FinBERT (finance)
* Code embeddings
* Legal embeddings

These outperform generic embeddings in specialized corpora.

But:

* May underperform on general queries
* Require domain knowledge to evaluate

---

# 2️⃣ Dimensionality Matters (But Not How You Think)

Typical sizes:

* 384
* 768
* 1024
* 1536
* 3072

Higher dimension ≠ automatically better.

What matters:

* Training objective
* Dataset diversity
* Contrastive learning quality

Higher dimension:

* Increases memory
* Increases ANN compute
* May improve fine-grained separation

But:

> Retrieval quality is more sensitive to training data than dimension count.

---

# 3️⃣ Embedding Model Objectives (Critical Concept)

Modern embedding models are trained using:

### Contrastive Learning

They learn:

```
Query → Positive Doc → Close
Query → Negative Doc → Far
```

So embeddings are optimized for:

* Search
* Retrieval
* Semantic similarity

Models trained only for classification often perform worse for RAG.

---

# 4️⃣ OpenAI vs Open-Source (Practical Comparison)

| Aspect        | OpenAI       | Open-Source        |
| ------------- | ------------ | ------------------ |
| Setup         | Instant      | Requires infra     |
| Cost          | Per call     | GPU + hosting      |
| Scaling       | Easy         | You manage         |
| Privacy       | External API | Fully private      |
| Customization | Limited      | Fine-tune possible |

For most production RAG:

* Start with OpenAI
* Migrate to open-source at scale

---

# 5️⃣ Domain Mismatch Problem

Suppose:

* You use general embedding model
* Your data = legal contracts

Result:

* Semantic drift
* Poor retrieval precision

Solution:

* Try domain-specific embeddings
* Or fine-tune embedding model

Never assume one model fits all.

---

# 6️⃣ How to Evaluate Embedding Quality

Do NOT evaluate by “feels right.”

Use retrieval metrics.

### Step 1:

Create test queries with ground-truth relevant documents.

### Step 2:

Compute:

* Precision@k
* Recall@k
* MRR (Mean Reciprocal Rank)

Example:

If correct document is in top 3:

* Good recall
* Decent ranking

If correct document is rank 15:

* Embedding weak or chunking flawed

---

# 7️⃣ Latency & Cost Considerations

Embedding cost scales with:

```
#chunks × embedding_dim × update frequency
```

Large corpora:

* Embedding generation cost becomes real
* Re-indexing expensive

You must consider:

* Update frequency
* Real-time vs batch indexing
* Cold-start performance

---

# 8️⃣ When to Fine-Tune Embeddings

Fine-tune only if:

* Domain extremely specialized
* Large training dataset available
* Clear evaluation metric exists

Fine-tuning without evaluation = waste.

Often:
Better chunking + hybrid search > fine-tuning.

---

# 9️⃣ Common Mistakes

❌ Switching embedding models without re-indexing
❌ Mixing embeddings from different models
❌ Ignoring normalization
❌ Choosing model by dimension size alone
❌ No evaluation dataset

---

# 🔟 Interview-Level Answer

If asked:

> “How do you choose an embedding model for RAG?”

Strong answer:

> “I benchmark multiple embedding models using retrieval metrics like Recall@k and MRR on a labeled evaluation set. I consider domain alignment, dimensionality, cost, latency, and whether hybrid retrieval reduces model sensitivity before considering fine-tuning.”

That shows maturity.

---

# 🧠 Mental Model

Embeddings define the geometry of your knowledge space.

Wrong geometry → perfect index won’t help.
Right geometry → even simple retrieval performs well.

---


