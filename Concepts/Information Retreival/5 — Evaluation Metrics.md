# Chapter 5 — Evaluation Metrics

## What is it?

Evaluation metrics answer one question: **how good is your retrieval system?** You've built a pipeline — now you need a number that tells you whether it's working, whether a change improved it, and how it compares to a baseline.

This is one of the most tested topics in FAANG ML interviews because it sits at the intersection of IR, ML, and product thinking. The interviewer isn't just checking if you know the formula — they want to know **which metric you'd pick for a given product situation and why.**

Every metric in this chapter measures something slightly different. Knowing when to use which one is the real skill.

---

## The intuition

Imagine you search for "machine learning tutorials" and the system returns 10 results. How do you measure quality?

- Did it find the good ones at all? → **Recall**
- Were the ones it returned actually good? → **Precision**
- Did the good ones appear near the top? → **MAP, NDCG**
- Did the first good result appear quickly? → **MRR**
- Were some results better than others (not just good/bad)? → **NDCG**

Each metric captures a different aspect of what "good retrieval" means. A system can score perfectly on one and terribly on another — which is why picking the right metric matters as much as optimizing it.

---

## Setup — what you need before computing any metric

Every metric requires:

```
1. A set of queries Q = {q1, q2, ..., qn}
2. For each query: a ranked list of retrieved documents
3. For each query: ground truth relevance labels

Binary labels (most metrics):   relevant = 1,  not relevant = 0
Graded labels (NDCG):           highly relevant = 3, relevant = 2,
                                 somewhat relevant = 1, not relevant = 0
```

We'll use this running example throughout:

```
query: "deep learning optimization"

system returns (in ranked order):
  rank 1: D1 — "Adam optimizer for deep neural networks"      → relevant    (1)
  rank 2: D2 — "Introduction to Python programming"           → not relevant (0)
  rank 3: D3 — "SGD and momentum in deep learning"            → relevant    (1)
  rank 4: D4 — "Deep learning hardware accelerators"          → not relevant (0)
  rank 5: D5 — "Gradient descent variants and learning rates" → relevant    (1)

total relevant documents in corpus: 4  (D1, D3, D5, D6 — but D6 was never retrieved)
```

---

## Metric 1 — Precision@k

### What is it?

Of the top-k results returned, what fraction are relevant?

```
P@k = (relevant documents in top-k) / k
```

### Worked example

```
P@1 = 1/1 = 1.00   (D1 is relevant)
P@2 = 1/2 = 0.50   (D1 relevant, D2 not)
P@3 = 2/3 = 0.67   (D1, D3 relevant, D2 not)
P@4 = 2/4 = 0.50
P@5 = 3/5 = 0.60
```

### Why it works / why it fails

**Good for:** Measuring how much junk is in the top results. A user who only looks at the first page cares about P@10 — they don't care if you have 10,000 relevant documents buried on page 100.

**Bad for:** It ignores where within the top-k the relevant results appear. P@3 = 0.67 whether the two relevant docs are at ranks 1,2 or at ranks 2,3. A system that puts the best result at rank 1 looks identical to one that buries it at rank 3. That's what MAP fixes.

---

## Metric 2 — Recall@k

### What is it?

Of all relevant documents in the corpus, what fraction did you retrieve in the top-k?

```
R@k = (relevant documents in top-k) / (total relevant documents in corpus)
```

### Worked example

```
total relevant = 4 (D1, D3, D5, D6)

R@1 = 1/4 = 0.25   (found D1 only)
R@3 = 2/4 = 0.50   (found D1, D3)
R@5 = 3/4 = 0.75   (found D1, D3, D5 — D6 never retrieved)
```

### Why it works / why it fails

**Good for:** Tasks where missing a relevant document is costly. Medical search, legal discovery, patent search — you need to find everything, not just a few good results.

**Bad for:** Recall can always be increased by returning more documents. Return every document in the corpus → R@N = 1.0. It says nothing about how many irrelevant results you also returned. Recall without Precision is meaningless.

---

## Precision vs Recall — the fundamental tradeoff

```
high threshold (return few docs) → high precision, low recall
low threshold (return many docs) → low precision, high recall

          ↑ Precision
      1.0 |●
          |  ●
          |    ●
          |       ●
          |           ●
          |                  ●
      0.0 +--------------------→ Recall
          0.0                1.0
```

A perfect system has precision = 1.0 at all recall levels. Real systems trace a curve — the Precision-Recall curve. Area Under this curve (AUC-PR) summarizes performance across all thresholds, but MAP (below) is the IR-specific version of this idea.

---

## Metric 3 — Average Precision (AP) and MAP

### What is it?

AP rewards systems that **rank relevant documents higher**. It computes precision every time a relevant document is found in the ranked list, then averages those precisions.

```
AP = (1 / R) × Σ P@k × rel(k)
```

Where:
- `R` = total number of relevant documents
- `P@k` = precision at rank k
- `rel(k)` = 1 if document at rank k is relevant, 0 otherwise
- Sum is over all ranks where a relevant document appears

MAP (Mean Average Precision) = mean of AP over all queries.

```
MAP = (1/|Q|) × Σ AP(q)   for each query q in Q
```

### Worked example

```
ranked list: D1(1), D2(0), D3(1), D4(0), D5(1)
total relevant R = 4 (including D6, never retrieved)

relevant document found at rank 1 (D1): P@1 = 1/1 = 1.000
relevant document found at rank 3 (D3): P@3 = 2/3 = 0.667
relevant document found at rank 5 (D5): P@5 = 3/5 = 0.600

AP = (1/4) × (1.000 + 0.667 + 0.600)
   = (1/4) × 2.267
   = 0.567
```

Note: D6 was never retrieved so it contributes 0 to the sum but still divides by R=4. Missing a relevant document always hurts AP.

### Why it works / why it fails

**Good for:** Captures both precision (not too much junk) and recall (don't miss relevant docs) and ranking quality (relevant docs should be near the top) in a single number. The standard metric for academic IR benchmarks (TREC).

**Bad for:** Binary relevance only — a document is relevant or not. If some relevant documents are much better than others (a perfect answer vs a tangentially related one), AP treats them identically. NDCG handles this.

---

## Metric 4 — NDCG (Normalized Discounted Cumulative Gain)

### What is it?

NDCG extends AP in two ways:
1. **Graded relevance** — documents get scores of 0, 1, 2, 3 instead of just 0 or 1
2. **Position discount** — a highly relevant doc at rank 1 is worth much more than the same doc at rank 5

```
DCG@k = Σ (2^relᵢ - 1) / log₂(i + 1)    summed from i=1 to k
```

Where:
- `relᵢ` = relevance score of document at rank i
- `2^relᵢ - 1` = exponential gain (heavily rewards high relevance scores)
- `log₂(i + 1)` = position discount (rank 1 has no discount, rank 5 is discounted by log₂(6) ≈ 2.58)

```
NDCG@k = DCG@k / IDCG@k
```

Where `IDCG@k` = DCG of the ideal ranking (best possible ordering). This normalizes the score to [0, 1].

### Worked example

```
query: "deep learning optimization"

graded relevance labels:
  rank 1: D1 — "Adam optimizer"               → rel = 3  (highly relevant)
  rank 2: D2 — "Python programming"           → rel = 0  (not relevant)
  rank 3: D3 — "SGD and momentum"             → rel = 2  (relevant)
  rank 4: D4 — "Hardware accelerators"        → rel = 1  (somewhat relevant)
  rank 5: D5 — "Gradient descent variants"    → rel = 3  (highly relevant)
```

### Step 1 — compute DCG@5

```
rank 1: (2³ - 1) / log₂(2) = 7     / 1.000 = 7.000
rank 2: (2⁰ - 1) / log₂(3) = 0     / 1.585 = 0.000
rank 3: (2² - 1) / log₂(4) = 3     / 2.000 = 1.500
rank 4: (2¹ - 1) / log₂(5) = 1     / 2.322 = 0.431
rank 5: (2³ - 1) / log₂(6) = 7     / 2.585 = 2.708

DCG@5 = 7.000 + 0.000 + 1.500 + 0.431 + 2.708 = 11.639
```

### Step 2 — compute IDCG@5 (ideal ranking: 3, 3, 2, 1, 0)

```
rank 1: (2³ - 1) / log₂(2) = 7 / 1.000 = 7.000
rank 2: (2³ - 1) / log₂(3) = 7 / 1.585 = 4.416
rank 3: (2² - 1) / log₂(4) = 3 / 2.000 = 1.500
rank 4: (2¹ - 1) / log₂(5) = 1 / 2.322 = 0.431
rank 5: (2⁰ - 1) / log₂(6) = 0 / 2.585 = 0.000

IDCG@5 = 7.000 + 4.416 + 1.500 + 0.431 + 0.000 = 13.347
```

### Step 3 — compute NDCG@5

```
NDCG@5 = DCG@5 / IDCG@5 = 11.639 / 13.347 = 0.872
```

0.872 out of 1.0. Lost points mainly because the second highly relevant document (D5, rel=3) was at rank 5 instead of rank 2. Moving D5 to rank 2 would significantly improve NDCG.

### Why it works / why it fails

**Good for:** Any situation where relevance is graded — e-commerce (perfect match vs. related product vs. loosely related), web search (exact answer vs. useful page vs. tangentially relevant). Penalizes putting highly relevant docs deep in the list more harshly than putting mediocre docs deep.

**Bad for:** Requires graded relevance labels, which are expensive to collect. Binary labels (0/1) collapse NDCG to something similar to MAP. Also, the choice of gain function (2^rel - 1) is somewhat arbitrary — linear gain (rel directly) is sometimes used instead.

---

## Metric 5 — MRR (Mean Reciprocal Rank)

### What is it?

MRR measures how quickly the system returns **the first relevant document.** Used when only the first good result matters — question answering, voice search, featured snippets.

```
RR(q) = 1 / rank_of_first_relevant_document

MRR = (1/|Q|) × Σ RR(q)   for each query q in Q
```

### Worked example

```
query 1: ranked list D2(0), D1(1), D3(1) → first relevant at rank 2 → RR = 1/2 = 0.500
query 2: ranked list D1(1), D2(0), D3(1) → first relevant at rank 1 → RR = 1/1 = 1.000
query 3: ranked list D2(0), D3(0), D1(1) → first relevant at rank 3 → RR = 1/3 = 0.333

MRR = (1/3) × (0.500 + 1.000 + 0.333) = (1/3) × 1.833 = 0.611
```

### Why it works / why it fails

**Good for:** Question answering systems, voice assistants, "I'm Feeling Lucky" style search. Any scenario where the user wants exactly one answer and doesn't care about results beyond the first relevant one.

**Bad for:** MRR completely ignores everything after the first relevant document. A system that returns one relevant doc followed by 99 irrelevant ones scores identically to a system that returns 100 relevant docs — as long as the first relevant doc is at the same rank. For most search use cases (where users browse several results), MAP or NDCG are more appropriate.

---

## Metric comparison — when to use which

| Metric | Relevance | Cares about rank? | Use when |
|--------|-----------|-------------------|----------|
| P@k | Binary | No (within top-k) | User only looks at top-k, no ranking preference |
| R@k | Binary | No | Missing a result is costly (legal, medical) |
| MAP | Binary | Yes | Academic benchmarks, general ranked retrieval |
| NDCG | Graded | Yes | Results have quality levels, web search, e-commerce |
| MRR | Binary | Yes (first only) | QA systems, voice search, one-answer tasks |

---

## The one thing to remember

All metrics measure the same underlying idea — **relevant documents should appear near the top** — but they differ in whether relevance is binary or graded, whether they care about all relevant docs or just the first, and how harshly they penalize relevant docs buried deep in the list. Pick the metric that matches how your users actually consume results.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `P@k = relevant_in_top_k / k` | Precision at cutoff k |
| `R@k = relevant_in_top_k / total_relevant` | Recall at cutoff k |
| `AP = (1/R) × Σ P@k × rel(k)` | Average precision for one query |
| `MAP = (1/\|Q\|) × Σ AP(q)` | Mean AP over all queries |
| `DCG@k = Σ (2^relᵢ - 1) / log₂(i+1)` | Discounted cumulative gain |
| `NDCG@k = DCG@k / IDCG@k` | Normalized DCG, range [0,1] |
| `RR = 1 / rank_of_first_relevant` | Reciprocal rank for one query |
| `MRR = (1/\|Q\|) × Σ RR(q)` | Mean reciprocal rank over all queries |

---

## Interview Q&A

**Q1. A product manager asks: "our NDCG went up but our MRR went down — is that good or bad?" How do you answer?**

It depends on the product. NDCG going up means the overall quality of the ranked list improved — highly relevant documents are appearing earlier on average across all positions. MRR going down means the very first relevant result is appearing slightly later. If your users typically scan several results (e-commerce browsing, research queries), the NDCG improvement matters more and this is likely a win. If your users want exactly one answer (voice assistant, question answering), the MRR drop is a problem regardless of NDCG. The right answer is: look at the product context, check whether the first result matters more than the overall list quality, and A/B test to see if user engagement metrics (click-through rate, dwell time) confirm the direction the offline metrics are pointing.

**Q2. Why does MAP penalize you for never retrieving a relevant document?**

Because AP divides by R — the total number of relevant documents in the corpus — not by the number retrieved. If 4 documents are relevant and you only retrieved 3, you divide the sum of precisions by 4, not 3. The missing document contributes 0 to the numerator but still inflates the denominator, pulling AP down. This is intentional — a system that misses relevant documents should score lower even if the ones it did retrieve were perfectly ranked.

**Q3. What is the difference between DCG and NDCG? Why do we need the normalization?**

DCG is the raw discounted cumulative gain for a specific ranked list. The problem with raw DCG is that it's not comparable across queries — a query with 10 highly relevant documents will naturally have a higher DCG ceiling than a query with only 2 relevant documents, regardless of ranking quality. NDCG divides by IDCG (the DCG of the perfect ranking for that query), normalizing to [0,1] for every query. This makes NDCG comparable across queries with very different numbers of relevant documents, which is essential when averaging over a query set to get a single system-level score.

**Q4. You have two systems. System A has MAP=0.72, System B has MAP=0.68. Can you conclude A is better?**

Not without statistical testing. MAP is averaged over a query set — the difference could be due to a few outlier queries rather than a systematic improvement. You should run a paired statistical significance test (paired t-test or Wilcoxon signed-rank test) over per-query AP scores. If p < 0.05, the difference is statistically significant. In academic IR (TREC evaluations), a difference of 0.04 MAP is sometimes significant and sometimes not depending on query set size. Additionally, offline MAP improvement doesn't always translate to online improvements — always validate with A/B tests on real user behavior metrics (CTR, session abandonment, dwell time).

**Q5. When would you use graded relevance labels over binary labels?**

When the difference between a perfect result and a merely relevant result matters to the user. In e-commerce: a search for "running shoes size 10" where the exact shoe in size 10 is rel=3, the same shoe in size 9 is rel=1, and a different shoe entirely is rel=0 — binary labels would treat both matches as equally good, missing the quality distinction. In web search: the Wikipedia article directly answering a question is rel=3, a blog post that partially answers it is rel=1, a tangentially related result is rel=0. Graded labels are more expensive to collect (annotators must agree on a scale, not just a binary) but produce a much more informative NDCG signal. In practice, implicit feedback (dwell time, clicks, purchases) is often used to infer graded relevance at scale without manual annotation.

---

Ready for your comments — what stays, what changes, what's missing?
