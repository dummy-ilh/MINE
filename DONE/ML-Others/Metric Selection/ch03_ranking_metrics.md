# Chapter 3: Ranking Metrics — NDCG, MAP, MRR

> *"In search and recommendation, not all positions are equal. Showing the right result at rank 10 is not the same as showing it at rank 1. Your metric must know that."*

---

## 3.1 Why Classification Metrics Break for Ranking

Imagine you build a search engine. Your model retrieves 10 documents for a query. You want to know: how good is this ranking?

You cannot use accuracy — there's no single "correct" answer. You cannot use AUC — it ignores position. You need metrics that:

1. **Reward relevance** — did we return good results?
2. **Reward position** — did the good results appear early?
3. **Handle graded relevance** — some results are "perfect", some are "good", some are "irrelevant"

This is the domain of **ranking metrics**.

---

## 3.2 Foundational Concepts

### Relevance Labels

Before computing any ranking metric, you need a relevance judgment for each (query, document) pair.

**Binary relevance:**
```
0 = irrelevant
1 = relevant
```

**Graded relevance** (more expressive):
```
0 = irrelevant
1 = somewhat relevant
2 = relevant
3 = highly relevant
4 = perfectly relevant (navigational query answered exactly)
```

Graded relevance is standard in modern search (used by Google, Bing, Amazon). It lets you distinguish between "good enough" and "exactly right."

### The Position Assumption

All ranking metrics share a core assumption: **users scan from top to bottom and are less likely to examine lower positions.**

This is empirically validated by eye-tracking studies. The click probability at position k roughly follows:

```
P(examined at rank k) ≈ 1 / log₂(k + 1)
```

This discounting function is the foundation of NDCG.

---

## 3.3 Precision@K and Recall@K

The simplest ranking metrics. Start here.

### Precision@K

*Of the top-K results returned, what fraction are relevant?*

```
Precision@K = (# relevant items in top K) / K
```

**Example:** Query returns [Rel, Irrel, Rel, Irrel, Rel] for K=5

```
Precision@5 = 3/5 = 0.60
```

**Limitation:** Treats all positions equally. A relevant result at rank 1 and rank 5 contribute equally. That's wrong.

### Recall@K

*Of all relevant items in the corpus, what fraction appear in the top K?*

```
Recall@K = (# relevant items in top K) / (total # relevant items)
```

**Limitation:** Doesn't care about order within the top K.

### When to use Precision@K

- You only show K results (e.g., a 10-result search page)
- You care about the quality of what's shown, not what's missed
- Binary relevance is fine

---

## 3.4 Mean Reciprocal Rank (MRR)

*How quickly does the first relevant result appear?*

### Formula

For a set of queries Q:

```
MRR = (1/|Q|) × Σ (1 / rank_of_first_relevant_result)
```

For a single query, the reciprocal rank (RR) is:

```
RR = 1/1 = 1.0    if first result is relevant
RR = 1/2 = 0.5    if second result is first relevant
RR = 1/3 = 0.33   if third result is first relevant
RR = 0             if no relevant result in top K
```

### Worked Example

Three queries:

| Query | Ranking | First Relevant At | RR |
|---|---|---|---|
| Q1 | [Rel, Irrel, Irrel] | Rank 1 | 1.00 |
| Q2 | [Irrel, Rel, Irrel] | Rank 2 | 0.50 |
| Q3 | [Irrel, Irrel, Rel] | Rank 3 | 0.33 |

```
MRR = (1.00 + 0.50 + 0.33) / 3 = 0.61
```

### When to use MRR

- **Question answering** — there's exactly one correct answer
- **Known-item search** — user looks for a specific document
- **Voice assistants** — you read out the first result only
- **Head queries** with one dominant relevant result

### MRR Limitation

MRR only cares about the **first** relevant result. If you have multiple relevant results (e.g., product search where many products could satisfy the user), MRR ignores all but the first. Use MAP or NDCG instead.

---

## 3.5 Average Precision and MAP

*How well is the ranking ordered overall, considering all relevant items?*

### Average Precision (AP)

For a single query, AP rewards models that place relevant items **early** and penalizes models that place them **late**.

```
AP = (1 / R) × Σₖ [Precision@k × Relevance(k)]
```

Where:
- R = total number of relevant items for this query
- k = position in the ranking
- Relevance(k) = 1 if item at position k is relevant, 0 otherwise

In plain English: compute Precision@k at every position where a relevant item appears, then average those values.

### Worked Example

Relevant items: {Doc A, Doc C, Doc E}. Ranking returned: [A, B, C, D, E]

| Position | Item | Relevant? | Precision@k | Included? |
|---|---|---|---|---|
| 1 | A | ✓ | 1/1 = 1.00 | Yes |
| 2 | B | ✗ | — | No |
| 3 | C | ✓ | 2/3 = 0.67 | Yes |
| 4 | D | ✗ | — | No |
| 5 | E | ✓ | 3/5 = 0.60 | Yes |

```
AP = (1/3) × (1.00 + 0.67 + 0.60) = (1/3) × 2.27 = 0.756
```

Compare to a bad ranking [B, D, A, C, E]:

| Position | Item | Relevant? | Precision@k |
|---|---|---|---|
| 1 | B | ✗ | — |
| 2 | D | ✗ | — |
| 3 | A | ✓ | 1/3 = 0.33 |
| 4 | C | ✓ | 2/4 = 0.50 |
| 5 | E | ✓ | 3/5 = 0.60 |

```
AP = (1/3) × (0.33 + 0.50 + 0.60) = (1/3) × 1.43 = 0.477
```

Same documents retrieved, but worse order → AP penalizes it correctly.

### Mean Average Precision (MAP)

Average AP over all queries:

```
MAP = (1/|Q|) × Σ AP(q)
```

### When to use MAP

- Multiple relevant items per query
- Binary relevance labels
- Information retrieval benchmarks (TREC, MS-MARCO)
- Document search, patent search, legal search

### MAP Limitation

MAP uses binary relevance — a document is either relevant or not. It can't express that one result is *better* than another relevant result. For that, you need NDCG.

---

## 3.6 Normalized Discounted Cumulative Gain (NDCG)

The most widely used ranking metric in industry. Handles graded relevance and position discounting.

### Building Up to NDCG

**Step 1: Gain**

The relevance score of each item.

```
Gain at position k = rel(k)
e.g., [3, 0, 2, 1, 3] for a 5-item ranking
```

**Step 2: Cumulative Gain (CG)**

Sum of gains — but ignores position.

```
CG@5 = 3 + 0 + 2 + 1 + 3 = 9
```

**Step 3: Discounted Cumulative Gain (DCG)**

Weight each gain by a position discount. Higher positions count more.

```
DCG@K = Σₖ rel(k) / log₂(k + 1)
```

The discount factor:

| Position k | log₂(k+1) | Discount |
|---|---|---|
| 1 | 1.00 | 1.000 |
| 2 | 1.58 | 0.631 |
| 3 | 2.00 | 0.500 |
| 4 | 2.32 | 0.431 |
| 5 | 2.58 | 0.387 |
| 10 | 3.46 | 0.289 |

**Alternate DCG formula** (used by many industry systems, stronger emphasis on highly relevant items):

```
DCG@K = Σₖ (2^rel(k) - 1) / log₂(k + 1)
```

This exponential gain formula amplifies the difference between relevance grades.

**Step 4: Ideal DCG (IDCG)**

The DCG of the perfect ranking — all items sorted by relevance descending. This is the ceiling.

```
Perfect ranking: [3, 3, 2, 1, 0]
IDCG@5 = 3/1 + 3/1.58 + 2/2 + 1/2.32 + 0/2.58
       = 3.00 + 1.90 + 1.00 + 0.43 + 0
       = 6.33
```

**Step 5: NDCG**

Normalize DCG by IDCG to get a score in [0, 1]:

```
NDCG@K = DCG@K / IDCG@K
```

### Full Worked Example

Query with 5 results. Relevance labels: [3, 0, 2, 1, 3]

```
DCG@5 = 3/log₂(2) + 0/log₂(3) + 2/log₂(4) + 1/log₂(5) + 3/log₂(6)
      = 3/1 + 0/1.58 + 2/2 + 1/2.32 + 3/2.58
      = 3.00 + 0 + 1.00 + 0.43 + 1.16
      = 5.59

Perfect ranking: [3, 3, 2, 1, 0]
IDCG@5 = 3/1 + 3/1.58 + 2/2 + 1/2.32 + 0/2.58
       = 3.00 + 1.90 + 1.00 + 0.43 + 0
       = 6.33

NDCG@5 = 5.59 / 6.33 = 0.883
```

An NDCG of 0.883 means we captured 88.3% of the possible discounted gain.

### NDCG@K: Choosing K

| K | Use case |
|---|---|
| NDCG@1 | Voice assistant, answer cards |
| NDCG@5 | Mobile search (5 visible results) |
| NDCG@10 | Standard web search |
| NDCG@100 | Retrieval stage of a two-stage system |

---

## 3.7 Comparing the Metrics

| Metric | Relevance | Position-aware | Multiple relevant items | Graded relevance |
|---|---|---|---|---|
| Precision@K | ✓ | ✗ | ✓ | ✗ |
| MRR | ✓ | ✓ | ✗ (only first) | ✗ |
| MAP | ✓ | ✓ | ✓ | ✗ |
| NDCG | ✓ | ✓ | ✓ | ✓ |

**Choosing:**
- One correct answer, find it fast → **MRR**
- Multiple relevant items, binary labels → **MAP**
- Graded relevance, position matters → **NDCG**
- Simple, fast sanity check → **Precision@K**

---

## 3.8 Practical Pitfalls

### Incomplete Relevance Judgments

In practice, you don't have relevance labels for every document. You only have labels for documents that were previously shown (position bias) or explicitly judged. This means:

- Your DCG denominator (IDCG) may be wrong
- Models that retrieve documents with *no* labels get unfairly penalized
- Solution: **Pooling** (judge all documents retrieved by any competing system) or **counterfactual correction**

### Position Bias in Logged Data

When collecting relevance from click data, clicks are biased toward position 1 even if a lower result is better. A document at position 1 gets more clicks than one at position 3 even if equally relevant.

Solutions:
- **Inverse Propensity Scoring (IPS):** Weight clicks by 1/P(examined at position k)
- **Randomized experiments:** Occasionally shuffle rankings to collect unbiased data

### NDCG Is Not Decomposable

You cannot compute NDCG incrementally over batches like you can accuracy. You need the full ranked list per query. Keep this in mind for distributed evaluation pipelines.

### Metric Gaming

If teams are measured purely on NDCG, they may:
- Stuff high-relevance items into position 1 even if they don't fit the query well
- Inflate relevance judgments
- Optimize for head queries (which drive metric) and ignore tail queries (which drive user satisfaction)

Always pair NDCG with **coverage metrics** (% of queries answered well) and **tail query analysis**.

---

## 3.9 Beyond the Standard Metrics

### Expected Reciprocal Rank (ERR)

Models user behavior more realistically: a user stops scanning after finding a satisfying result. Each position has a probability of satisfying the user, given they've reached it.

```
ERR = Σₖ (1/k) × P(stop at k)
P(stop at k) = R(k) × Π_{i<k} (1 - R(i))
```

Where R(k) is the graded relevance at position k, normalized to [0,1].

### Diversity Metrics

A ranking with 10 identical results scores well on NDCG but is useless. Metrics like **α-NDCG** and **D#-NDCG** penalize redundancy by reducing the gain of a result if it's similar to something already seen higher in the list.

### Novelty and Serendipity

In recommendation:
- **Novelty**: Are we recommending items the user hasn't seen before?
- **Serendipity**: Are we recommending items the user wouldn't have found themselves but will love?

These are covered in depth in Chapter 15 (Recommender System Metrics).

---

## 3.10 Implementation Notes

### Computing NDCG in Python

```python
import numpy as np

def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k])
    positions = np.arange(1, len(relevances) + 1)
    return np.sum(relevances / np.log2(positions + 1))

def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

# Example
relevances = [3, 0, 2, 1, 3]
print(f"NDCG@5: {ndcg_at_k(relevances, 5):.4f}")  # 0.8826
```

### Libraries

- `sklearn.metrics.ndcg_score` — standard NDCG
- `pytrec_eval` — full TREC evaluation suite (MAP, MRR, NDCG, P@K, ERR)
- `ranx` — fast, modern ranking evaluation library

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Precision@K | Fraction of top-K results that are relevant; ignores order |
| MRR | How fast does the first relevant result appear |
| MAP | AP averaged over queries; handles multiple relevant items |
| DCG | Cumulative gain discounted by position |
| NDCG | DCG normalized by the ideal; handles graded relevance |
| Position bias | Clicks are biased; correct with IPS or randomization |
| Diversity | NDCG doesn't penalize redundancy; need extra metrics |

---

## Further Reading

- Järvelin & Kekäläinen — *Cumulated Gain-Based Evaluation of IR Techniques* (original NDCG paper, 2002)
- Burges et al. — *Learning to Rank Using Gradient Descent* (LambdaRank, Microsoft)
- Chapelle & Chang — *Yahoo! Learning to Rank Challenge Overview*
- `pytrec_eval` documentation — battle-tested TREC evaluation tooling

---

*Next: Chapter 4 — Calibration (Platt Scaling, Isotonic Regression)*
