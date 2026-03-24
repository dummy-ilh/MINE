# 🔥 Day 29 — Capstone Project

**Real Dataset • Measured Improvements • Written Design Doc**

Today you move from theory → execution.

You will design, build, evaluate, and defend a **production-grade RAG system**.

This is portfolio-level.
This is interview-level.
This is “I actually built it” level.

---

# 0️⃣ Project Goal

Build a **measurable RAG system** over a real-world dataset and demonstrate:

1. Baseline performance
2. Iterative improvements
3. Quantified gains
4. Clear system design

No vibes. Only metrics.

---

# 1️⃣ Choosing the Dataset

You want:

* Medium scale (5k–100k docs)
* Domain-specific
* Non-trivial queries

Here are strong options:

---

## 📚 Legal Dataset

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Joseph_Shine_vs_Union_of_India_%28Adultery_Judgement%29.pdf/page1-424px-Joseph_Shine_vs_Union_of_India_%28Adultery_Judgement%29.pdf.jpg)

![Image](https://www.flowlu.com/site/assets/files/28578/legal-case-1.712x443.png)

![Image](https://m.media-amazon.com/images/I/61p0C0D9m2L._AC_UF1000%2C1000_QL80_.jpg)

![Image](https://static.cambridge.org/content/id/urn%3Acambridge.org%3Aid%3Aarticle%3AS0008197300134154/resource/name/firstPage-S0008197300134154a.jpg)

Examples:

* US court opinions
* Contract clauses
* Regulatory filings

---

## 🏥 Medical Dataset

![Image](https://i1.rgstatic.net/publication/354240290_Clinical_Research_An_Overview_of_Study_Types_Designs_and_Their_Implications_in_the_Public_Health_Perspective/links/6131e30c0360302a0076f6e4/largepreview.png)

![Image](https://www.nlm.nih.gov/bsd/policy/graphics/structured_abs_fig1a.gif)

![Image](https://imgv2-2-f.scribdassets.com/img/document/506184974/original/e0524484cf/1?v=1)

![Image](https://nhsrcindia.org/sites/default/files/stg.jpg)

Examples:

* Clinical guidelines
* Research abstracts
* Medical Q&A corpus

---

## 💻 Code Dataset

![Image](https://camo.githubusercontent.com/ccd0f0debb449778034c17b06817ce8139f381524aee0882d925778fd3f92341/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6c756379706f7274666f6c696f2f70726f6a656374732f6769742d747265652f747265652e706e67)

![Image](https://us1.discourse-cdn.com/flex002/uploads/python1/original/2X/9/93a24780ad41b431929cf4a50982516873dea566.png)

![Image](https://blog.bitbox.swiss/en/content/images/2021/09/pexels-pixabay-270348-1.jpg)

![Image](https://medevel.com/content/images/2024/02/434443.png)

Examples:

* GitHub repo documentation
* API reference manuals
* StackOverflow-style Q&A

---

## 💼 Finance Dataset

![Image](https://www.investopedia.com/thmb/7VhbIdLJZqkNiUzYAjeVQRrjjDQ%3D/1500x0/filters%3Ano_upscale%28%29%3Amax_bytes%28150000%29%3Astrip_icc%28%29/Microsoft202310-KFinancialStatements-2401d7d8a55e4d5c9c2a3283fe65fdf2.png)

![Image](https://img.yumpu.com/30649058/1/500x640/earnings-call-transcript-pdf-interactive-brokers.jpg)

![Image](https://www.sec.gov/files/styles/embed_full_width_large/public/images/edgar-filing-website.png?itok=rruKcrB7)

![Image](https://www.sec.gov/files/styles/component_block_image_1x/public/images/scrsht-online-forms-mgmt-144-ssp3m.png?itok=lc7M6evY)

Examples:

* Earnings transcripts
* 10-K reports
* Investment research reports

---

# 2️⃣ Define the Evaluation Problem

Before writing code, define:

[
What\ does\ good\ look\ like?
]

You need:

* Query set (100–500 questions)
* Ground-truth answers
* Gold source documents

Without ground truth → no scientific improvement.

---

# 3️⃣ Phase 1 — Baseline RAG

Architecture:

```text
Chunk (fixed 512 tokens)
↓
Embed
↓
Vector search (top-k=5)
↓
Single-pass generation
```

Measure:

* Recall@k
* Answer relevance
* Faithfulness
* Latency
* Token cost

This is your control group.

---

# 4️⃣ Phase 2 — Improve Retrieval

Now apply controlled experiments.

### Experiment A: Hybrid Retrieval

Combine dense + BM25.

Measure improvement in recall@k.

---

### Experiment B: Better Chunking

Compare:

* 512 fixed tokens
* Section-aware chunking
* Overlap vs no-overlap

Measure:

[
\Delta recall@k
]

---

### Experiment C: Re-Ranking

Add cross-encoder ranker.

Measure:

* MRR
* Precision@k
* Downstream answer quality

---

# 5️⃣ Phase 3 — Hallucination Mitigation

Add:

* Citation-constrained prompting
* Claim verification pass
* Refusal threshold

Measure:

* Faithfulness score
* Unsupported claim ratio
* Refusal rate

---

# 6️⃣ Phase 4 — Latency Optimization

Measure stage timing:

| Stage        | Time |
| ------------ | ---- |
| Retrieval    | ?    |
| Rerank       | ?    |
| LLM          | ?    |
| Verification | ?    |

Then optimize:

* Reduce top-k
* Cache embeddings
* Compress context

Measure cost reduction.

---

# 7️⃣ What Your Final Results Should Look Like

Example (hypothetical):

| Metric             | Baseline | Final |
| ------------------ | -------- | ----- |
| Recall@5           | 0.68     | 0.84  |
| Faithfulness       | 0.72     | 0.91  |
| Latency            | 2.4s     | 1.8s  |
| Hallucination Rate | 18%      | 6%    |

If you can show this table in an interview, you win.

---

# 8️⃣ Write the Design Document

Structure:

---

## 1. Problem Statement

What system are we building?

---

## 2. Constraints

* Corpus size
* Latency SLA
* Risk tolerance
* Update frequency

---

## 3. Architecture Diagram

Include ingestion + retrieval + ranking + generation + verification.

---

## 4. Retrieval Strategy

* Embedding model
* Chunking strategy
* Metadata usage
* Reranking

---

## 5. Evaluation Framework

* Dataset
* Metrics
* Offline vs online
* Ablation experiments

---

## 6. Results

Show quantitative improvements.

---

## 7. Failure Analysis

Describe:

* Where system fails
* Example bad cases
* Root cause analysis

This shows maturity.

---

# 9️⃣ Advanced Additions (If You Want To Impress)

* Add HyDE query expansion
* Add RAPTOR-style hierarchical retrieval
* Add observability dashboard
* Add multi-tenant isolation
* Add document-level access control

---

# 🔟 Capstone Interview Questions

---

### Q1: What was your biggest bottleneck?

Expected answer:
Retrieval recall ceiling or hallucination control.

---

### Q2: What experiment failed?

Strong candidates mention:

* Chunk size too small hurt coherence.
* Large top-k increased noise.
* Verification increased latency too much.

Failures show depth.

---

### Q3: How would you scale to 100M documents?

Answer:

* ANN indexing
* Sharding by tenant/domain
* Async ingestion
* Embedding refresh pipeline

---

# 1️⃣1️⃣ What Separates Strong vs Elite Candidates

Strong:

> “I built RAG and improved recall.”

Elite:

> “I ran ablations, measured faithfulness, reduced hallucination by 12%, optimized cost by 35%, and documented tradeoffs.”

---

# 1️⃣2️⃣ What This Capstone Proves

You now understand:

* Retrieval recall ceiling
* Ranking precision impact
* Context budgeting
* Hallucination mitigation
* Evaluation methodology
* Latency-cost tradeoffs

That is end-to-end RAG mastery.

---

Next:

🔥 Day 30 — Mastery Check
(Explain RAG to CEO, ML Engineer, Researcher + Mock Interview + Final System Review)

Say continue when ready.
