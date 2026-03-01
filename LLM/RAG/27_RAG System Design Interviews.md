# 🔥 Day 27 — RAG System Design Interviews

**Whiteboard Architecture • Tradeoffs • Failure Analysis**

This is where most strong candidates fail.

They know embeddings.
They know vector DBs.

But they **cannot design a production system under constraints.**

Today we fix that.

---

# 1️⃣ The Interview Framing

Most interview prompts look like:

* “Design a document QA system over internal company data.”
* “Build a legal assistant using RAG.”
* “Design ChatGPT over enterprise docs.”
* “How would you scale RAG to 10M documents?”

The interviewer is evaluating:

1. Architectural clarity
2. Tradeoff reasoning
3. Failure awareness
4. Evaluation strategy
5. Safety awareness

---

# 2️⃣ Canonical Whiteboard Architecture

You should always start with high-level blocks.

```text
User
 ↓
API Gateway
 ↓
Query Processor
 ↓
Retriever
 ↓
Ranker
 ↓
Context Builder
 ↓
LLM Generator
 ↓
Post-Processor (Verification / Guardrails)
 ↓
Response
```

Then expand each block.

---

# 3️⃣ Deep Dive Block-by-Block

---

## A. Document Ingestion Pipeline

```text
Raw Docs
 ↓
Cleaning
 ↓
Chunking
 ↓
Embedding
 ↓
Metadata tagging
 ↓
Vector Store
```

### Interview Gold:

Mention:

* Chunk size tradeoff
* Overlap strategy
* Metadata filters
* Batch embedding vs streaming

---

## Chunking Tradeoff

| Small chunks         | Large chunks              |
| -------------------- | ------------------------- |
| High recall          | Better context coherence  |
| Noisy                | Expensive                 |
| Fragmented reasoning | Lower retrieval precision |

A strong answer:

> “I’d tune chunk size based on document structure and average query length.”

---

## B. Retrieval Layer

Types you must mention:

* Dense retrieval
* Hybrid retrieval (BM25 + dense)
* Metadata filtering
* Reranking (cross-encoder)

If you don’t mention reranking, you look junior.

---

## C. Context Builder

Key constraints:

* Token budget
* Ordering strategy
* Deduplication
* Citation tracking

This is where many hallucinations begin.

---

## Token Budget Formula

If:

[
Window = 8k
]

You must allocate:

* 1k for system + instructions
* 1k for user
* 5k for retrieved docs
* 1k buffer for generation

Interviewers love hearing token accounting.

---

# 4️⃣ Latency Budget Thinking

Suppose SLA = 2 seconds.

Breakdown:

| Component    | Budget |
| ------------ | ------ |
| Retrieval    | 200ms  |
| Reranking    | 200ms  |
| LLM          | 1200ms |
| Verification | 300ms  |

If you don’t budget, you don’t pass.

---

# 5️⃣ Failure Analysis (Critical Section)

Now we move to the hardest part.

---

## Failure Mode 1: Wrong Answer

You must isolate:

* Retrieval issue?
* Context assembly issue?
* Generation hallucination?

Debug strategy:

1. Log retrieved documents.
2. Compare similarity scores.
3. Evaluate ground truth retrieval recall.
4. Run LLM with gold context.

This separation shows senior thinking.

---

## Failure Mode 2: Empty Retrieval

Causes:

* Bad embedding model
* Query phrasing mismatch
* Metadata filter too strict

Fix:

* Add query expansion
* Add HyDE-style reformulation
* Relax filters

---

## Failure Mode 3: Latency Spike

Causes:

* Large context
* Too many rerank calls
* Slow model

Fix:

* Cache embeddings
* Use smaller reranker
* Use streaming responses

---

## Failure Mode 4: Hallucination Despite Good Retrieval

Cause:
LLM overweights parametric memory.

Fix:

* Citation-constrained prompting
* Claim verification
* Refusal policy

---

# 6️⃣ Scaling to 10M+ Documents

This is where system design depth shows.

Mention:

### 1. Sharded Vector DB

Partition by:

* Tenant
* Domain
* Time

### 2. ANN Index (HNSW / IVF)

Approximate nearest neighbors for speed.

### 3. Async Retrieval

Parallel:

* Dense
* Sparse
* Metadata query

### 4. Embedding Refresh Strategy

If embedding model updates:

* Re-embed entire corpus?
* Dual-index migration strategy?

This shows production awareness.

---

# 7️⃣ Evaluation in Production

Mention metrics:

### Offline:

* Recall@k
* MRR
* Faithfulness
* Groundedness

### Online:

* CTR
* User correction rate
* Refusal rate
* Latency

Senior candidates talk about monitoring dashboards.

---

# 8️⃣ Safety Considerations

Mention:

* Prompt injection detection
* Access control per document
* Sensitive data filtering
* Rate limiting

If you skip safety, interviewers mark you down.

---

# 9️⃣ Tradeoff Questions You Must Handle

---

### Q: Long context model vs RAG?

Answer:

* Long context → expensive, retrieval implicit
* RAG → modular, scalable, controllable

Use RAG when:

* Knowledge updates frequently
* Corpus large
* Need explainability

---

### Q: When would you fine-tune instead?

When:

* Domain language stable
* Retrieval overhead too high
* Need style control
* Closed corpus

---

### Q: What is the biggest bottleneck in RAG?

Answer depends:

* Retrieval recall ceiling
* Context window limit
* LLM hallucination
* Latency under scale

Good answer:

> “Retrieval recall is usually the upper bound. If relevant doc never retrieved, generation cannot fix it.”

---

# 1️⃣0️⃣ Whiteboard Strategy Template

When asked to design RAG:

Follow this order:

1. Clarify requirements
2. Define constraints (latency, corpus size, safety)
3. Draw high-level blocks
4. Deep dive ingestion
5. Deep dive retrieval
6. Discuss scaling
7. Discuss failure modes
8. Discuss evaluation
9. Discuss safety

That flow wins interviews.

---

# 1️⃣1️⃣ Mock Interview Questions (Hard Mode)

---

### Q1: How would you detect if retrieval quality degraded over time?

Answer:

* Track recall@k on fixed evaluation set.
* Monitor similarity score distribution drift.
* Track drop in groundedness.

---

### Q2: How do you handle conflicting documents?

Answer:

* Retrieve multiple sources.
* Present disagreement explicitly.
* Add timestamp metadata.
* Possibly rank by authority.

---

### Q3: What’s your biggest concern deploying RAG in healthcare?

Answer:

* Hallucination under incomplete context.
* Outdated medical guidance.
* Overconfidence in answers.
* Refusal calibration.

---

### Q4: How would you reduce cost by 40%?

Options:

* Smaller embedding model
* Cache frequent queries
* Reduce top-k
* Use distilled reranker
* Compress context

---

# 1️⃣2️⃣ What Separates Junior vs Senior Answer

Junior:

> “Use embeddings and vector DB.”

Senior:

> Talks about chunking, ranking, latency, evaluation, drift detection, safety, refusal calibration, scaling strategy.

Staff-level:

> Mentions cost modeling, migration strategy, A/B testing, multi-tenant isolation.

---

# Mastery Check

You should now:

* Be able to whiteboard a full RAG system.
* Discuss tradeoffs without hesitation.
* Isolate failures cleanly.
* Defend architectural decisions.

---


