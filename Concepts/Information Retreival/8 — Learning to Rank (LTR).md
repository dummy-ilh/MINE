# Chapter 8 — Learning to Rank (LTR) — Boosted Master Notes
### 🔥 Boosted Edition: Master Notes + Full Interview Q&A Bank

> **How to use this document:** Nothing from the original chapter has been removed. Every original section, formula, worked example, and Q&A is intact below. On top of that, this edition adds: (1) a rapid-review cheat sheet at the top, (2) an expanded interview Q&A bank, (3) "rapid-fire flashcards," and (4) a combined formula/pitfall sheet at the end. Look for the 🆕 marker to spot everything that's new.

---

## 🆕 MASTER CHEAT SHEET — Chapter 8 at a glance

| Paradigm | Training unit | Loss target | Canonical algorithm | Key weakness |
|---|---|---|---|---|
| Pointwise | Single (query, doc) | Absolute relevance score | Regression/classification | Ignores ordering entirely |
| Pairwise | Pair of docs | Correct relative order | RankNet | Weights all pairs equally (rank 1↔2 swap = rank 99↔100 swap) |
| Listwise | Full ranked list | Ranking metric (NDCG) directly | LambdaRank / LambdaMART | Uses a heuristic pseudo-gradient, not a true NDCG gradient |

| Key fact | Detail |
|---|---|
| Why NDCG can't be optimized directly | Depends on rank (from sorting) — sorting is non-differentiable, no ∂NDCG/∂score |
| LambdaRank's fix | Multiply RankNet's gradient by \|ΔNDCG\| from swapping that pair |
| LambdaMART = | LambdaRank gradients + MART (gradient boosted trees) |
| Pairwise training cost | n(n-1)/2 pairs per query — quadratic in candidates per query |
| Position bias | Users click rank-1 more often regardless of true relevance → feedback loop |
| Fix for position bias | Inverse Propensity Scoring (IPS): weight click by 1/P(examination at that position) |
| Why trees dominated 2008–2018 | Handle mixed feature types, fast serving, interpretable, work with less data |
| Why neural re-rankers rose after 2018 | Learn feature interactions + text features automatically |
| Typical production hybrid | LambdaMART fast first-stage + neural cross-encoder re-ranker on top-k |

---

## What is it?

Learning to Rank is a family of supervised machine learning approaches that train a model to produce an optimal ranking of documents for a given query. Instead of hand-crafting a scoring formula like BM25, you let the model **learn the ranking function from labeled data.**

In classical IR (Chapters 2–3), the scoring function is fixed — BM25 with k₁ and b. In dense retrieval (Chapter 4), the scoring function is a dot product between learned embeddings. LTR is different: it takes **many signals simultaneously** — BM25 score, embedding similarity, document freshness, click-through rate, PageRank, title match, URL quality — and learns how to combine them optimally for your specific corpus and user base.

This is the dominant approach at Google, Bing, Baidu, and Meta for the final ranking stage. When an interviewer at these companies asks "how does ranking work at scale?" — LTR is the expected answer.

---

## The intuition

Imagine you're a hiring manager reviewing 100 CVs. You have several signals for each candidate: years of experience, university ranking, previous company prestige, interview score, referral strength. No single signal is perfect. Someone with 10 years of experience at a bad company might be worse than someone with 3 years at Google. You need to **combine signals intelligently.**

LTR does exactly this for documents. Each document gets a feature vector — a list of numerical signals — and the model learns how to combine those features into a single relevance score that best matches human relevance judgments.

The key insight: **the ranking problem is not the same as the classification or regression problem.** You don't just want to predict relevance accurately — you want the ordering to be correct. A model that predicts relevance scores of [0.9, 0.8, 0.1] for documents [A, B, C] produces the same ranking as one that predicts [100, 50, 1] — even though the absolute values are completely different. LTR loss functions are designed around this ordering property.

---

## The three paradigms

LTR approaches are divided by what they treat as the training instance:

```
pointwise:  one document at a time  → predict relevance score for (query, doc)
pairwise:   two documents at a time → predict which of (doc_i, doc_j) is more relevant
listwise:   full ranked list        → optimize ranking quality of the entire list directly
```

Each is a different answer to the question: **what should the loss function measure?**

---

## Paradigm 1 — Pointwise

### What it does

Treat ranking as regression or classification. For each (query, document) pair, predict a relevance score (0/1 for binary, 0–3 for graded).

```
input:   feature vector f(q, d)
output:  relevance score ŷ ∈ [0, 3]
loss:    MSE = (ŷ - y)²   or   cross-entropy
```

### Worked numeric example

```
training data:
  (q1, D1): features=[0.8, 0.6, 0.9], label=3  (highly relevant)
  (q1, D2): features=[0.4, 0.2, 0.3], label=1  (somewhat relevant)
  (q1, D3): features=[0.1, 0.8, 0.2], label=0  (not relevant)

model predicts:
  D1: ŷ=2.7, true=3  → loss = (2.7-3)² = 0.09
  D2: ŷ=1.2, true=1  → loss = (1.2-1)² = 0.04
  D3: ŷ=0.4, true=0  → loss = (0.4-0)² = 0.16

total MSE loss = (0.09 + 0.04 + 0.16) / 3 = 0.097
```

### Why it fails

Pointwise loss doesn't care about ordering — it cares about absolute score accuracy. Consider:

```
true labels:    D1=3,  D2=2,  D3=1
model A scores: D1=2.9, D2=1.9, D3=0.9  → MSE=0.01, ranking: D1>D2>D3 ✓
model B scores: D1=0.5, D2=1.5, D3=0.3  → MSE=2.17, ranking: D2>D1>D3 ✗
```

Model B has much higher loss but its ranking is wrong in a way that matters (D2 ranked above D1). Model A has low loss AND correct ranking. Pointwise loss doesn't directly optimize for ranking correctness — it's an indirect proxy.

---

## Paradigm 2 — Pairwise

### What it does

For each pair of documents (Dᵢ, Dⱼ) with different relevance labels, train the model to score the more relevant document higher.

```
input:   feature vectors f(q, Dᵢ) and f(q, Dⱼ)
output:  P(Dᵢ ≻ Dⱼ) = probability that Dᵢ should rank above Dⱼ
loss:    binary cross-entropy on the ordering
```

### The RankNet loss

RankNet (Burges et al., Microsoft 2005) is the foundational pairwise model:

```
sᵢ = model score for Dᵢ
sⱼ = model score for Dⱼ

P(Dᵢ ≻ Dⱼ) = σ(sᵢ - sⱼ) = 1 / (1 + e^(-(sᵢ-sⱼ)))

loss = -y_ij × log(P) - (1 - y_ij) × log(1 - P)

where y_ij = 1 if Dᵢ more relevant than Dⱼ, else 0
```

The model learns to push the score of the more relevant document above the less relevant one.

### Worked numeric example

```
query q1, documents D1 (label=3) and D2 (label=1)
model scores: s1=2.1, s2=1.4

P(D1 ≻ D2) = σ(2.1 - 1.4) = σ(0.7) = 1 / (1 + e^(-0.7))
           = 1 / (1 + 0.497)
           = 1 / 1.497
           = 0.668

true label: y_12 = 1 (D1 should rank above D2)

loss = -1 × log(0.668) - 0 × log(0.332)
     = -log(0.668)
     = 0.404
```

Now suppose the model had it backwards — s1=1.4, s2=2.1:

```
P(D1 ≻ D2) = σ(1.4 - 2.1) = σ(-0.7) = 0.332

loss = -log(0.332) = 1.102   ← much higher loss, gradient pushes s1 up
```

The loss is higher when the ordering is wrong, so gradients push the model toward correct orderings.

### Number of pairs

```
if a query has n documents:
  number of pairs = n(n-1)/2

n=10   → 45 pairs
n=100  → 4,950 pairs
n=1000 → 499,500 pairs
```

Pairwise training scales quadratically with documents per query — expensive for large candidate sets.

### Why pairwise is better than pointwise but still imperfect

Pairwise directly optimizes ordering between pairs. But it weights all pairs equally — a swap between rank 1 and rank 2 (which users notice) counts the same as a swap between rank 99 and rank 100 (which nobody notices). Listwise approaches fix this.

---

## Paradigm 3 — Listwise

### What it does

Treat the entire ranked list as a single training instance. Optimize a ranking metric (NDCG, MAP) directly — or a smooth approximation of it.

```
input:   all feature vectors {f(q, D1), f(q, D2), ..., f(q, Dn)}
output:  full ranked list
loss:    directly related to NDCG or a smooth surrogate of it
```

### Why not optimize NDCG directly?

NDCG is not differentiable — it depends on ranks, which are integers. You can't compute a gradient through a sorting operation. Listwise methods use smooth surrogate losses that approximate NDCG.

### LambdaRank and LambdaMART

LambdaRank (also Microsoft, Burges 2006) is the most important listwise method. It builds on RankNet but **weights each pair by how much swapping them would change NDCG:**

```
λᵢⱼ = |ΔNDCG| × ∂L_RankNet / ∂(sᵢ - sⱼ)
```

Where |ΔNDCG| = how much would NDCG change if we swapped the ranks of Dᵢ and Dⱼ?

This means:
- Swapping rank 1 and rank 2 → large |ΔNDCG| → large gradient → model strongly penalized
- Swapping rank 98 and rank 99 → tiny |ΔNDCG| → tiny gradient → model barely penalized

### Worked numeric example — LambdaRank weighting

```
query with 5 documents, current ranking: D1, D2, D3, D4, D5
true relevance:                          3,  1,  2,  0,  1

current NDCG@5 = 0.856

consider swapping D1 (rank 1, rel=3) with D3 (rank 3, rel=2):
  new ranking: D3, D2, D1, D4, D5
  new NDCG@5  = 0.821
  |ΔNDCG| = |0.856 - 0.821| = 0.035  ← large, gradient amplified

consider swapping D4 (rank 4, rel=0) with D5 (rank 5, rel=1):
  new ranking: D1, D2, D3, D5, D4
  new NDCG@5  = 0.863
  |ΔNDCG| = |0.856 - 0.863| = 0.007  ← small, gradient barely matters
```

LambdaRank multiplies the RankNet gradient by |ΔNDCG| for each pair — so the model focuses training effort on the swaps that matter most for the metric you care about.

**LambdaMART** = LambdaRank gradients + MART (Multiple Additive Regression Trees = gradient boosted trees). This is the specific algorithm used in production at Bing and historically at Yahoo. Gradient boosted trees handle mixed feature types (continuous BM25 scores, categorical language flags, integer click counts) naturally without normalization.

---

## Feature engineering for LTR

This is where domain knowledge matters. Common feature categories:

### Query-document match features
```
BM25_body        → BM25 score on document body
BM25_title       → BM25 score on document title (usually boosted)
BM25_url         → BM25 score on URL tokens
dense_sim        → cosine similarity from bi-encoder
exact_match      → 1 if query appears verbatim in doc, 0 otherwise
query_coverage   → fraction of query terms appearing in doc
```

### Document quality features
```
PageRank         → graph-based authority score
inlink_count     → number of pages linking to this document
domain_authority → quality score of the domain
content_length   → number of words
spam_score       → classifier output for spam/low-quality content
```

### Freshness features
```
doc_age_days     → how old is the document
last_modified    → days since last update
query_freshness  → is this a trending/time-sensitive query?
```

### User engagement features (very powerful, requires logging)
```
CTR_at_rank_k    → historical click-through rate at this rank position
dwell_time       → average time users spend on this doc after clicking
pogo_sticking    → rate at which users immediately return to results
```

### Worked feature vector example

```
query: "python sorting algorithms"
document: "Sorting in Python: sort() vs sorted() — complete guide"

feature vector:
  BM25_body:        8.43
  BM25_title:      12.71   ← title match is strong
  dense_sim:        0.89
  exact_match:      1       ← "python" and "sorting" in title
  query_coverage:   0.75    ← 3 of 4 query terms matched
  PageRank:         0.0034
  domain_authority: 72
  doc_age_days:     180
  CTR_at_rank_1:    0.42
  dwell_time:       145s

label: 3 (highly relevant, from human annotation)
```

---

## The full LTR pipeline

```
offline (training):
  1. collect queries + candidate docs from production logs
  2. get relevance labels (human annotators or click-based implicit labels)
  3. compute feature vectors for all (query, doc) pairs
  4. train LambdaMART on (features, labels) with NDCG as target metric
  5. evaluate on held-out query set: NDCG@10, MAP

online (serving):
  retrieval stage:  BM25 + dense ANN → top 200 candidates    ~50ms
  feature compute:  extract all features for 200 candidates  ~20ms
  LTR score:        LambdaMART forward pass on 200 × F       ~5ms
  return top 10                                               ~75ms total
```

---

## Why it works / why it fails

**Why it works:**
- Combines many heterogeneous signals that no single formula can capture
- LambdaMART directly optimizes NDCG — what you measure is what you train for
- Gradient boosted trees handle missing features, mixed types, and non-linear interactions without careful preprocessing
- Interpretable — you can inspect feature importances to understand what the model learned
- Decades of production validation at Bing, Yahoo, and Baidu

**Why it fails:**
- Requires labeled training data — expensive to collect at scale, especially for tail queries
- Features must be computed at serving time — each feature adds latency
- Click-based labels have position bias — documents at rank 1 get clicked more simply because they're at rank 1, not because they're more relevant. Model learns to rank position-1 documents higher, reinforcing the bias. Requires debiasing techniques (Inverse Propensity Scoring)
- LambdaMART doesn't generalize across domains — a model trained on web search performs poorly on e-commerce without retraining
- Feature staleness — CTR features are stale within hours for trending topics

---

## The one thing to remember

LTR learns to combine many relevance signals — BM25, embeddings, PageRank, CTR — into a single score by directly optimizing ranking metrics like NDCG. LambdaMART is the production-proven algorithm: gradient boosted trees trained with LambdaRank gradients weighted by how much each pairwise swap would change NDCG.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `P(Dᵢ ≻ Dⱼ) = σ(sᵢ - sⱼ)` | RankNet: probability that Dᵢ ranks above Dⱼ |
| `σ(x) = 1 / (1 + e^(-x))` | Sigmoid function |
| `L_RankNet = -y_ij × log(P) - (1-y_ij) × log(1-P)` | RankNet pairwise cross-entropy loss |
| `λᵢⱼ = \|ΔNDCG\| × ∂L/∂(sᵢ-sⱼ)` | LambdaRank gradient — weighted by NDCG impact |
| `pairs = n(n-1)/2` | Number of document pairs for pairwise training |
| `MSE = (ŷ - y)²` | Pointwise regression loss |

---

## Interview Q&A

**Q1. What are the three LTR paradigms and what does each optimize?**

Pointwise treats each (query, document) pair independently and optimizes absolute relevance score prediction — essentially regression or classification. It's simple but doesn't directly optimize ordering. Pairwise takes pairs of documents and optimizes the probability that the more relevant document scores higher — RankNet is the canonical example. It directly optimizes ordering between pairs but weights all pairs equally regardless of their position impact. Listwise takes the full ranked list as a single instance and optimizes a ranking metric like NDCG directly — LambdaRank and LambdaMART are the canonical examples. Listwise is the strongest because it weights training signal by how much each swap would change the metric you actually care about.

**Q2. Why can't you optimize NDCG directly as a loss function?**

NDCG depends on ranks — the position of a document in the sorted list. Sorting is a non-differentiable operation: a tiny change in score doesn't continuously change rank, it jumps discretely when two scores cross. Since you can't compute ∂NDCG/∂score, you can't use gradient descent directly. LambdaRank solves this by computing what the NDCG gradient *would be* if it were differentiable — it uses the RankNet gradient multiplied by |ΔNDCG| as a proxy. This pseudo-gradient is not mathematically derived from NDCG but empirically it pushes the model toward better NDCG, which is why it works in practice.

**Q3. What is position bias in LTR and how do you correct for it?**

Position bias occurs because users are more likely to click documents at rank 1 than rank 5 regardless of actual relevance — simply because rank 1 is more visible. If you train on raw click data, the model learns that rank-1 documents are "relevant" and will keep promoting them, creating a feedback loop. Inverse Propensity Scoring (IPS) corrects for this: each click is weighted by 1/P(click at position k), where P is the examination probability at that position. A click at rank 5 (low examination probability) is weighted much higher than a click at rank 1 (high examination probability). The examination probabilities are estimated via randomization experiments — randomly shuffle results for a fraction of queries and observe click patterns across positions.

**Q4. You're joining a company that has no click data and no human labels. How do you bootstrap an LTR system?**

Start with BM25 as the initial ranker — no labels needed. Use it to serve results and log user interactions. From those logs, extract implicit relevance signals: clicks, dwell time, scroll depth, add-to-cart events. These are noisy but scalable. Apply position bias correction (IPS) to the click data to get debiased relevance estimates. Use these as weak labels to train an initial LTR model. Once the LTR model is deployed, its improved rankings generate better click data, which generates better labels — a virtuous cycle. For the head queries (top 1000 most common queries covering ~50% of traffic), invest in human annotation to get high-quality labels for the queries that matter most. This hybrid approach — human labels for head, implicit for tail — is standard at every major search company.

**Q5. LambdaMART uses gradient boosted trees. Why not a neural network?**

In practice, both are used. LambdaMART with gradient boosted trees (XGBoost/LightGBM) dominated from 2008–2018 for several reasons: it handles heterogeneous features (floats, integers, booleans) without normalization, it's fast at serving time (a few milliseconds for a tree forward pass vs. GPU inference for neural nets), it's interpretable via feature importance scores, and it generalizes well from relatively small labeled datasets. Neural LTR (e.g. using BERT features or end-to-end neural rankers) has become more common since 2018 because it can learn feature interactions automatically and handles text features natively. Modern production systems often use a hybrid: LambdaMART for fast first-stage ranking, neural re-ranker (cross-encoder) for the top-k. The cross-encoder from Chapter 4 is essentially a neural LTR model — it's just trained on (query, document) pairs with relevance labels.

---

## 🆕 EXPANDED INTERVIEW Q&A BANK — Chapter 8

**Q6 🆕: "Derive why RankNet's gradient with respect to (sᵢ - sⱼ) simplifies so cleanly, the same way sigmoid+cross-entropy simplifies in binary classification."**

**Answer:** Let `s = sᵢ - sⱼ` and `P = σ(s)`. RankNet's loss is `L = -y·log(P) - (1-y)·log(1-P)` — structurally identical to binary cross-entropy with `P` playing the role of a predicted probability and `y` the true pairwise label. Since `σ'(s) = P(1-P)`, the same chain-rule cancellation that produces `∂BCE/∂z = ŷ-y` in ordinary logistic regression applies here directly: `∂L/∂s = P - y`. This means the gradient pushing `sᵢ` up and `sⱼ` down is simply "how wrong the pairwise probability estimate is" — large when the model confidently has the order backwards, near zero when it's already correct — for exactly the same structural reason sigmoid+cross-entropy is well-behaved in standard binary classification (Chapter 5). RankNet is, in essence, logistic regression applied to score *differences* rather than raw features.

---

**Q7 🆕: "LambdaMART's gradient (λᵢⱼ) is described as a 'pseudo-gradient, not a true derivative of NDCG.' Why is this scientifically defensible rather than just a hack?"**

**Answer:** It's defensible because the construction satisfies the property that actually matters for gradient-based optimization: **it points in a direction that provably decreases NDCG-relevant pairwise errors while producing a valid, well-behaved gradient field** that gradient boosting can consume. Formally, Donmez et al. and later Burges showed that if you integrate the λ-gradients across all pairs, the resulting "implicit loss function" has NDCG-consistent stationary points — meaning wherever the λ-gradients vanish, that ranking is also a local optimum of NDCG itself, even though no explicit closed-form loss function was ever written down whose exact derivative is λ. In other words, LambdaRank works backward: instead of writing a differentiable loss and taking its gradient, it directly specifies what the gradient *should* look like (RankNet gradient scaled by NDCG-sensitivity) and shows that pursuing that gradient reaches NDCG-optimal points — a legitimate technique in optimization known as constructing a gradient without an explicit potential function, not merely an empirical trick.

---

**Q8 🆕: "In the worked LambdaRank example, swapping D1↔D3 changed NDCG by 0.035, much more than swapping D4↔D5 (0.007). Explain why this specific pattern — top-of-list swaps mattering more — falls directly out of NDCG's formula, not just intuition."**

**Answer:** NDCG's gain term is `Σᵢ (2^relᵢ - 1)/log₂(i+1)`, where `i` is the rank position. The denominator `log₂(i+1)` grows slowly, but critically, the *relative* difference in discount between adjacent ranks is far larger near the top: `log₂(2)=1` vs `log₂(3)≈1.585` (a ~37% discount drop from rank 1→2) compared to `log₂(99)≈6.629` vs `log₂(100)≈6.644` (a ~0.2% discount drop from rank 98→99). Since a swap's NDCG impact is proportional to the *difference* in discount weights between the two positions being swapped (multiplied by the difference in relevance grades), swapping high-relevance documents near the top — where discount weights are changing steeply — mechanically produces much larger |ΔNDCG| than swapping anywhere near the tail, where the discount curve has already flattened out. This isn't a heuristic pattern LambdaRank imposes; it's a direct mathematical consequence of the logarithmic discount already baked into NDCG's definition.

---

**Q9 🆕: "The chapter says pairwise training costs n(n-1)/2 pairs per query. In production with n=1000 candidates, that's ~500K pairs per query. How do production systems make LambdaMART training tractable at this scale?"**

**Answer:** Several complementary techniques are used in practice, none of which change the underlying algorithm but which make it computationally feasible: (1) **Sampling pairs** rather than enumerating all of them — since most information content comes from pairs with different relevance grades, many implementations only sample a bounded number of pairs per query (e.g., a few hundred) rather than the full combinatorial set, especially since most pairs in a large candidate set are between two irrelevant documents and carry near-zero training signal. (2) **Restricting to graded-label pairs** — pairs where both documents share the same relevance label contribute exactly zero gradient (since `y_ij` is undefined/irrelevant when they're tied), so those pairs are skipped entirely rather than computed and discarded. (3) **Truncating candidate lists** — since first-stage retrieval already narrows millions of documents down to a few hundred candidates (as shown in the pipeline: "top 200 candidates"), `n` in practice is bounded by the retrieval stage's cutoff, not the full corpus, keeping `n(n-1)/2` in the tens of thousands rather than billions. (4) **Tree-based factorization** — because LambdaMART's trees split on individual features rather than pairs directly, the boosting procedure itself amortizes cost across all pairs sharing a feature value, rather than treating each pair as an independent unit of work.

---

**Q10 🆕: "How would inverse propensity scoring (IPS) actually be estimated in practice — where do the position-wise examination probabilities P(examination at rank k) come from?"**

**Answer:** The chapter mentions "estimated via randomization experiments" — concretely, this typically works via a **result randomization / RandTop-n experiment**: for a small, controlled slice of live traffic, the search engine randomly shuffles the order of the top-n results (or randomly swaps a small number of adjacent positions) rather than serving the model's true ranking. Because relevance is now decoupled from position (a genuinely relevant document is just as likely to land at rank 5 as rank 1 in this randomized slice), the *observed* click-through rate at each position in this randomized data is a direct, unbiased estimate of the pure "examination probability" at that position — independent of document quality. This gives you an empirical curve `P(examine | rank=k)` for `k=1...n`, typically showing a sharp drop-off (rank 1 might have 0.6 examination probability, rank 10 might have 0.05). Once you have this curve from the randomized experiment, you apply it as the IPS denominator on the much larger volume of *normal* (non-randomized) production traffic, correcting every click in the standard logs by dividing by the examination probability at the rank it occurred, without needing to randomize the majority of live traffic (which would hurt user experience if done broadly).

---

## 🆕 RAPID-FIRE FLASHCARDS — Chapter 8

| Prompt | Answer |
|---|---|
| Three LTR paradigms? | Pointwise, pairwise, listwise |
| Pointwise loss? | MSE or cross-entropy on absolute relevance score |
| Pointwise's core flaw? | Doesn't optimize ordering, only score accuracy |
| RankNet formula? | P(Dᵢ≻Dⱼ) = σ(sᵢ-sⱼ) |
| RankNet loss? | Binary cross-entropy on the pairwise ordering label |
| Pairs per query (n docs)? | n(n-1)/2 |
| Pairwise's core flaw? | Weights all pairs equally regardless of position impact |
| Why can't NDCG be optimized directly? | Depends on rank via sorting — non-differentiable |
| LambdaRank gradient? | λᵢⱼ = \|ΔNDCG\| × ∂L_RankNet/∂(sᵢ-sⱼ) |
| LambdaMART = ? | LambdaRank gradients + gradient boosted trees (MART) |
| Position bias cause? | Users click rank-1 more regardless of true relevance |
| Position bias fix? | Inverse Propensity Scoring — weight click by 1/P(examination) |
| How are examination probabilities estimated? | Randomized result-ordering experiments on a small traffic slice |
| Why trees beat neural nets 2008–2018? | Mixed feature types, fast serving, interpretable, less data-hungry |
| Modern production hybrid? | LambdaMART first-stage + neural cross-encoder re-ranker |
| Bootstrap LTR with no labels? | BM25 → log clicks → IPS-debias → weak labels → human labels for head queries |

---

## 🆕 CHAPTER 8 FORMULA SHEET

```
RankNet probability:      P(Dᵢ≻Dⱼ) = σ(sᵢ-sⱼ) = 1/(1+e^(-(sᵢ-sⱼ)))
RankNet loss:             L = -y_ij·log(P) - (1-y_ij)·log(1-P)
RankNet gradient:         ∂L/∂(sᵢ-sⱼ) = P - y_ij      [same cancellation as BCE]
LambdaRank gradient:      λᵢⱼ = |ΔNDCG| · ∂L_RankNet/∂(sᵢ-sⱼ)
Pointwise loss:           MSE = (ŷ-y)²
Number of pairs:          n(n-1)/2
NDCG gain term:           Σᵢ (2^relᵢ - 1)/log₂(i+1)
IPS-corrected click:      weight = 1 / P(examination at rank k)
```

## 🆕 "TOP 5 THINGS THAT TRIP PEOPLE UP" — Chapter 8

1. Saying LambdaRank "optimizes NDCG directly" — it doesn't; it uses a pseudo-gradient constructed to have NDCG-consistent stationary points, since NDCG itself has no true gradient.
2. Forgetting that pairwise loss weights a rank-1↔2 swap identically to a rank-99↔100 swap — this is exactly the gap listwise methods close.
3. Assuming raw click-through rate is an unbiased relevance signal — without IPS correction, click data is contaminated by position bias and creates a self-reinforcing feedback loop.
4. Treating LambdaMART and neural rankers as competitors rather than complementary — production systems typically use trees for fast first-stage ranking and neural cross-encoders for a smaller top-k re-ranking pass.
5. Not connecting RankNet's gradient back to ordinary logistic regression — the `P - y` cancellation is the exact same mechanism as Chapter 5's sigmoid+cross-entropy gradient, just applied to a score difference instead of a raw feature combination.

---

*This document preserves 100% of the original Chapter 8 content and adds interview-focused expansions marked with 🆕.*
