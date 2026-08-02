# Chapter 7 — Learning to Rank (LTR)
### Complete Mastery Edition

---

## The Big Picture First

Every retrieval system so far (BM25, dense, hybrid RRF) produces a ranked list. But these systems rank documents by a single signal — keyword overlap, vector similarity, or a fusion of the two.

**Learning to Rank asks a different question:** what if we treat ranking itself as a machine learning problem, where the model learns from hundreds of signals simultaneously — BM25 score, PageRank, freshness, click-through rate, user dwell time — and from human judgments about what "good ranking" actually means?

```
Before LTR:
  ranked list = f(one signal)
  e.g. rank by BM25 score

After LTR:
  ranked list = f(many signals, learned from labeled data)
  e.g. rank by ML model trained on [BM25, PageRank, freshness, CTR, ...]
```

This is what Google, Bing, and Meta actually do in production. The retrieval stage (BM25 + dense) is just candidate generation. LTR is the brain that decides the final order.

---

## The Three Paradigms — Master This Taxonomy

There are three fundamentally different ways to frame ranking as a learning problem. The difference is in **what the loss function operates on.**

```
Pointwise  →  loss on individual documents
Pairwise   →  loss on pairs of documents
Listwise   →  loss on the entire ranked list
```

Each paradigm trains a different kind of model and has different failure modes. Understanding the progression from pointwise → pairwise → listwise is the story of the field.

---

## Part 1 — Pointwise Ranking

### The idea

Treat ranking as regression or classification on individual documents. Train a model to predict the relevance score of each document independently. Rank by that score.

```
Input:   (query, document) → feature vector x
Output:  predicted relevance score ŷ ∈ {0, 1, 2, 3}  (e.g. 0=irrelevant, 3=perfect)
Model:   standard regression or classification (linear, tree, neural)
Loss:    mean squared error, cross-entropy — any standard supervised loss
```

### How training data looks

```
query: "best running shoes for flat feet"

document D1: "Brooks Adrenaline GTS review — best stability shoe"
  features: [BM25=8.2, PageRank=0.71, freshness=0.9, CTR=0.34]
  label:    3  (highly relevant — human judge rated it)

document D2: "Nike Air Max sale — 40% off"
  features: [BM25=2.1, PageRank=0.85, freshness=0.95, CTR=0.12]
  label:    1  (marginally relevant — mentions shoes but wrong type)

document D3: "Flat feet anatomy and causes"
  features: [BM25=4.3, PageRank=0.55, freshness=0.4, CTR=0.05]
  label:    0  (not relevant — medical, not product)
```

Train a regression model: minimize `(ŷ - y)²` summed over all (query, document) pairs.

### Calculation example

```
model: simple linear regression
features: [BM25_score, PageRank, freshness]
learned weights: w = [0.4, 0.3, 0.3]

scoring D1:
  ŷ = 0.4×8.2 + 0.3×0.71 + 0.3×0.9
    = 3.28 + 0.213 + 0.27
    = 3.763

scoring D2:
  ŷ = 0.4×2.1 + 0.3×0.85 + 0.3×0.95
    = 0.84 + 0.255 + 0.285
    = 1.380

scoring D3:
  ŷ = 0.4×4.3 + 0.3×0.55 + 0.3×0.4
    = 1.72 + 0.165 + 0.12
    = 2.005

ranked list: D1 (3.763) > D3 (2.005) > D2 (1.380)
```

### Why pointwise fails

The core problem: **ranking is relative, but pointwise loss is absolute.**

```
query A:  label D1=3, D2=2  →  D1 should rank above D2
query B:  label D1=1, D2=0  →  D1 should rank above D2

The correct ranking is identical for both queries.
But pointwise MSE treats them differently:
  query A error: predicting 2.9 and 1.9 is fine
  query B error: predicting 0.9 and 0.1 is fine

The model learns absolute score prediction, not relative ordering.
If it predicts D1=1.2 and D2=1.5 for query B, the ranking is wrong,
but the MSE might still be low.
```

The loss function doesn't directly penalize wrong orderings — it penalizes wrong absolute predictions. A model can minimize pointwise loss while producing terrible rankings.

---

## Part 2 — Pairwise Ranking

### The idea

Instead of predicting relevance scores, train a model to predict: **given documents Di and Dj, which should rank higher?**

```
Input:   (query, Di, Dj) → feature difference vector Δx
Output:  P(Di should rank above Dj) ∈ [0, 1]
Loss:    binary cross-entropy on (Di > Dj) vs (Dj > Di)
```

The training set is built from document pairs where we know the correct ordering from human labels.

### Training pairs construction

```
From query labels: D1=3, D2=2, D3=1, D4=0

Valid pairs (where ordering is known):
  (D1, D2): D1 > D2  ✓  label = 1
  (D1, D3): D1 > D3  ✓  label = 1
  (D1, D4): D1 > D4  ✓  label = 1
  (D2, D3): D2 > D3  ✓  label = 1
  (D2, D4): D2 > D4  ✓  label = 1
  (D3, D4): D3 > D4  ✓  label = 1

Total pairs from n documents: n(n-1)/2
  4 documents → 6 pairs
  100 documents → 4,950 pairs  ← explodes quickly
```

### RankNet — the foundational pairwise model

RankNet (Burges et al., Microsoft, 2005) is a neural network trained on pairwise preferences. It's the direct ancestor of LambdaRank and LambdaMART.

**The scoring function:**

```
For document Di, the model predicts a scalar score: sᵢ = f(xᵢ)
(where f is a neural network or any differentiable function)

Predicted probability that Di ranks above Dj:
  Pᵢⱼ = σ(sᵢ - sⱼ) = 1 / (1 + e^(-(sᵢ-sⱼ)))
```

This is just sigmoid applied to the score difference. If sᵢ > sⱼ, then Pᵢⱼ > 0.5, meaning we predict Di ranks above Dj.

**The loss function (cross-entropy over pairs):**

```
True probability that Di should rank above Dj:
  P̄ᵢⱼ = 1   if label(Di) > label(Dj)
  P̄ᵢⱼ = 0   if label(Di) < label(Dj)
  P̄ᵢⱼ = 0.5 if label(Di) = label(Dj)

RankNet loss for one pair:
  L(i,j) = -P̄ᵢⱼ × log(Pᵢⱼ) - (1 - P̄ᵢⱼ) × log(1 - Pᵢⱼ)

When Di is definitely better than Dj (P̄ᵢⱼ = 1):
  L(i,j) = -log(Pᵢⱼ) = -log(σ(sᵢ - sⱼ)) = log(1 + e^(-(sᵢ-sⱼ)))
```

**Worked calculation:**

```
Documents D1 and D2, labels: D1=3 (relevant), D2=1 (not relevant)
Current model scores: s₁=2.1, s₂=1.8

Step 1 — score difference:
  s₁ - s₂ = 2.1 - 1.8 = 0.3

Step 2 — predicted probability:
  P₁₂ = σ(0.3) = 1/(1 + e^(-0.3)) = 1/(1 + 0.741) = 1/1.741 = 0.574

Step 3 — true probability (D1 > D2 definitively):
  P̄₁₂ = 1

Step 4 — loss:
  L = -log(0.574) = -(-0.554) = 0.554

The gradient pushes s₁ up and s₂ down to increase P₁₂ toward 1.

Now suppose model scores were s₁=1.8, s₂=2.1 (wrong order):
  P₁₂ = σ(1.8 - 2.1) = σ(-0.3) = 1/(1 + e^(0.3)) = 1/1.350 = 0.426
  L = -log(0.426) = 0.854  ← higher loss, bigger gradient correction
```

### Why pairwise is better than pointwise

The loss now **directly penalizes wrong orderings.** If D2 ranks above D1 when it shouldn't, the loss is high and gradients correct the model. The absolute score value doesn't matter — only relative differences.

### Why pairwise still fails

Pairwise loss treats all pairs equally. But in practice, swapping rank 1 and rank 2 is a catastrophe (the top result is wrong). Swapping rank 98 and rank 99 is nearly invisible (no user sees it).

```
query: 100 documents, D1 is perfectly relevant

Pair (D2, D1) — D2 wrongly ranked above D1:
  pairwise loss contribution = same as any other pair ✗
  
Pair (D99, D98) — D99 wrongly ranked above D98 (both irrelevant):
  pairwise loss contribution = same as (D2, D1) ✗

Both pairs contribute equally to the loss.
But getting D1 to rank first is worth astronomically more.
```

This is the core failure: **pairwise ranking doesn't know which position in the list matters.**

---

## Part 3 — Listwise Ranking

### The idea

Operate on the entire ranked list at once. Use a loss function that directly measures list quality — typically NDCG or MAP — and optimize for it.

```
Input:   (query, [D1, D2, ..., Dn]) → all documents for this query
Output:  optimal permutation of documents
Loss:    directly based on NDCG, MAP, or other list-quality metrics
```

### Why NDCG — quick review

NDCG (Normalized Discounted Cumulative Gain) is the standard metric for measuring ranking quality:

```
DCG@k = Σᵢ₌₁ᵏ  (2^relevance(i) - 1) / log₂(i + 1)

NDCG@k = DCG@k / IDCG@k
  where IDCG = DCG of the perfect ranking (upper bound)
```

The key property: **positions near the top are weighted exponentially more** (via the `log₂(i+1)` denominator). Rank 1 is worth much more than rank 10.

**Calculation:**

```
query: 5 documents, labels [3, 2, 3, 0, 1]

System ranking: [D1, D3, D2, D5, D4] → relevances [3, 3, 2, 1, 0]

DCG@5:
  pos 1: (2³ - 1) / log₂(2) = 7 / 1.000 = 7.000
  pos 2: (2³ - 1) / log₂(3) = 7 / 1.585 = 4.416
  pos 3: (2² - 1) / log₂(4) = 3 / 2.000 = 1.500
  pos 4: (2¹ - 1) / log₂(5) = 1 / 2.322 = 0.431
  pos 5: (2⁰ - 1) / log₂(6) = 0 / 2.585 = 0.000
  DCG = 7.000 + 4.416 + 1.500 + 0.431 + 0.000 = 13.347

Ideal ranking: [D1, D3, D2, D5, D4] → same as above (already optimal here)
  IDCG = 13.347

NDCG@5 = 13.347 / 13.347 = 1.000  ← perfect

Bad ranking: [D4, D5, D2, D1, D3] → relevances [0, 1, 2, 3, 3]

DCG@5:
  pos 1: (2⁰ - 1) / log₂(2) = 0 / 1.000 = 0.000
  pos 2: (2¹ - 1) / log₂(3) = 1 / 1.585 = 0.631
  pos 3: (2² - 1) / log₂(4) = 3 / 2.000 = 1.500
  pos 4: (2³ - 1) / log₂(5) = 7 / 2.322 = 3.014
  pos 5: (2³ - 1) / log₂(6) = 7 / 2.585 = 2.708
  DCG = 0.000 + 0.631 + 1.500 + 3.014 + 2.708 = 7.853

NDCG@5 = 7.853 / 13.347 = 0.588  ← significant quality loss
```

The difference between the two rankings is 0.412 NDCG — despite returning the same 5 documents. **Position matters enormously.**

### The problem: NDCG is not differentiable

NDCG depends on the rank of each document, which is a step function — it jumps discretely. You can't take a gradient of a step function. Standard backpropagation breaks.

```
NDCG as a function of scores s₁, s₂, ..., sₙ:
  changing s₁ slightly doesn't change rank until it crosses s₂
  at that crossing, NDCG jumps discontinuously
  gradient is 0 everywhere, ∞ at crossings
  → standard gradient descent can't optimize this
```

This is the central challenge of listwise ranking. The models below are the solutions.

---

## Part 4 — LambdaRank — The Key Insight

### The problem it solves

We can't compute ∂NDCG/∂score directly. But LambdaRank asks: **what if we just define the gradient we wish we had, and train with that?**

### LambdaRank's trick

For each pair of documents (Di, Dj) where Di should rank above Dj, RankNet computes a gradient that pushes their scores apart. LambdaRank *scales* that gradient by how much swapping Di and Dj would change NDCG:

```
λᵢⱼ  =  (RankNet gradient for pair i,j)  ×  |ΔNDCG from swapping i and j|
```

Documents whose swap would cause a big NDCG drop get a large gradient. Documents whose swap is irrelevant get a tiny gradient. The model is told: **care more about getting the top results right.**

### Computing ΔNDCG for a swap

```
query: 5 documents
current ranking: [D1, D2, D3, D4, D5]  → relevances [3, 2, 1, 0, 0]

NDCG of current ranking (call it NDCG₀):
  DCG = 7/1.000 + 3/1.585 + 1/2.000 + 0/2.322 + 0/2.585
      = 7.000 + 1.893 + 0.500 + 0 + 0 = 9.393

Swap D1 and D2 (ranks 1 and 2):
  new ranking: [D2, D1, D3, D4, D5] → relevances [2, 3, 1, 0, 0]
  DCG = 3/1.000 + 7/1.585 + 1/2.000 + 0/2.322 + 0/2.585
      = 3.000 + 4.416 + 0.500 + 0 + 0 = 7.916
  ΔNDCG = |9.393 - 7.916| / IDCG  (ignoring IDCG for comparison)
  ΔNDCG ≈ 1.477 / IDCG  ← LARGE — this swap is costly

Swap D3 and D4 (ranks 3 and 4):
  new ranking: [D1, D2, D4, D3, D5] → relevances [3, 2, 0, 1, 0]
  DCG = 7/1.000 + 3/1.585 + 0/2.000 + 1/2.322 + 0/2.585
      = 7.000 + 1.893 + 0 + 0.431 + 0 = 9.324
  ΔNDCG = |9.393 - 9.324| / IDCG ≈ 0.069 / IDCG  ← SMALL — barely matters
```

LambdaRank uses `|ΔNDCG| = 1.477` as the weight for the D1-D2 gradient and `|ΔNDCG| = 0.069` for the D3-D4 gradient. The model is 21× more focused on fixing the D1-D2 ordering.

### The full lambda gradient

```
For each pair (i, j) where Di should rank above Dj:

  λᵢⱼ = -σ̄ᵢⱼ × |ΔNDCGᵢⱼ|

  where σ̄ᵢⱼ = σ(sⱼ - sᵢ) = 1 / (1 + e^(sᵢ-sⱼ))
  (this is the RankNet gradient — it's largest when the ordering is wrong)

Total gradient for document i:
  λᵢ = Σⱼ λᵢⱼ - Σⱼ λⱼᵢ
  (sum over all documents that i should rank above, minus those it shouldn't)
```

**The intuition:** λᵢ is a force vector on document i's score. Documents that are wrongly ranked below less-relevant documents feel a strong upward push, proportional to how much NDCG would improve by fixing the ordering.

---

## Part 5 — LambdaMART — The Production Model

### From LambdaRank to LambdaMART

LambdaRank defined the lambda gradients. But what function should we use to compute scores? A neural network works but is slow to train and prone to overfitting on small datasets.

LambdaMART (Multiple Additive Regression Trees) applies the lambda gradients to **gradient boosted decision trees (GBDT)** instead of a neural network.

```
LambdaMART = LambdaRank gradients + Gradient Boosted Trees
```

GBDT builds an ensemble of decision trees sequentially. Each new tree fits the *residual error* of the previous ensemble. With LambdaRank, "residual error" is replaced by "lambda gradient" — the direction and magnitude each document's score should move to improve NDCG.

### How GBDT works (the core loop)

```
Algorithm:
  Initialize: F₀(x) = constant (e.g. 0)
  
  For t = 1 to T:
    1. Compute lambda gradients: λᵢ = ∂NDCG/∂sᵢ (via LambdaRank)
    2. Fit a decision tree hₜ to the lambdas:
       hₜ(x) ≈ λᵢ  for each document i
    3. Update the model: Fₜ(x) = Fₜ₋₁(x) + η × hₜ(x)
       where η = learning rate (e.g. 0.1)
  
  Final model: F(x) = Σₜ η × hₜ(x)  ← ensemble of trees
```

Each tree fits the current error signal (lambdas). The ensemble incrementally improves NDCG. After T=1000 trees, you have a strong ranking model.

### Decision tree — concrete example

```
Training data for one tree iteration:
  D1: features [BM25=8.2, PageRank=0.71], lambda=+2.3  ← needs score boost
  D2: features [BM25=2.1, PageRank=0.85], lambda=-1.1  ← needs score cut
  D3: features [BM25=5.4, PageRank=0.40], lambda=+0.3  ← small boost
  D4: features [BM25=1.2, PageRank=0.20], lambda=-0.8  ← small cut

Tree split (depth 2):
  if BM25 > 4.0:
    if PageRank > 0.5:  → leaf value = avg(lambda D1) = +2.3
    else:               → leaf value = avg(lambda D3) = +0.3
  else:
    if PageRank > 0.5:  → leaf value = avg(lambda D2) = -1.1
    else:               → leaf value = avg(lambda D4) = -0.8

Score update (η=0.1):
  D1: s₁ += 0.1 × 2.3  = +0.23  ← score rises
  D2: s₂ += 0.1 × (-1.1) = -0.11  ← score drops
  D3: s₃ += 0.1 × 0.3  = +0.03
  D4: s₄ += 0.1 × (-0.8) = -0.08
```

Repeat for 1000 trees. Each tree makes a small correction. By iteration 1000, the model has learned a complex non-linear function over all features that maximizes NDCG.

### Why LambdaMART dominates in practice

```
Property          Neural (LambdaRank)   Tree (LambdaMART)
─────────────────────────────────────────────────────────
Training speed    Slow (backprop)        Fast (GBDT)
Small datasets    Overfits easily        Robust
Feature handling  Learned automatically  Explicit, interpretable
Missing values    Needs imputation       Handles natively
Inference speed   Moderate              Very fast (tree lookup)
Tuning effort     High                  Moderate
Industry use      RAG re-ranking         Web search (Google, Bing)
```

LambdaMART is what runs in production at web-scale search engines. LightGBM and XGBoost both implement it. You can train a production-quality ranking model in hours on a single machine.

---

## Part 6 — Feature Engineering for Ranking

### What features go into an LTR model

The power of LTR is that it can combine *any signals* you can compute. The model learns which signals matter and how to combine them. Here are the major categories:

### Category 1 — Query-Document Relevance Features

These measure how well a document matches the query.

```
BM25_score          — TF-IDF weighted keyword overlap
BM25_title          — BM25 score computed on title only (often stronger signal)
BM25_body           — BM25 score on body text
BM25_anchor         — BM25 on anchor text (what other pages link with)
TF-IDF_score        — simpler variant
dense_similarity    — cosine similarity from bi-encoder
exact_match_title   — binary: does title contain exact query string?
query_coverage      — fraction of query terms appearing in document
term_proximity      — are query terms close together in document?
```

**Why anchor text matters:** If thousands of pages link to a document using the anchor text "best running shoes," that's an external signal that the document is actually about running shoes — even if those words don't appear prominently in the document itself.

### Category 2 — Document Quality Features

These measure the document's inherent quality, independent of the query.

```
PageRank            — probability of landing on this page via random web walk
in_link_count       — number of pages linking to this document
out_link_count      — number of outbound links (too many = spam signal)
document_length     — word count (too short or too long = quality signal)
reading_level       — Flesch-Kincaid or similar
spam_score          — ML-based spam classifier output
domain_authority    — quality of the hosting domain
avg_word_length     — content quality proxy
```

**PageRank calculation intuition:**

```
PageRank is the stationary distribution of a random walk:
  PR(d) = (1-d) / N  +  d × Σ_{v → d}  PR(v) / out_degree(v)

  d = damping factor (typically 0.85)
  N = total pages
  v = pages that link to d

Interpretation: a page is important if many important pages link to it.
This is defined recursively — solved via iterative convergence.

Example (3 pages, all linking to each other):
  Initial: PR(A) = PR(B) = PR(C) = 1/3 = 0.333

  Iteration 1 (d=0.85):
    PR(A) = 0.15/3 + 0.85 × [PR(B)/1 + PR(C)/1]
          = 0.05 + 0.85 × [0.333 + 0.333]
          = 0.05 + 0.85 × 0.666 = 0.616
    (same for B and C by symmetry)
  
  Iteration 2 normalizes, converges to ~0.333 for each in this toy example
  (In real web graphs with asymmetric link structures, values vary widely)
```

### Category 3 — User Behavior Features

These are the most powerful and hardest to fake.

```
CTR (click-through rate)    — fraction of users who clicked this result for this query
dwell_time                  — how long users stayed on the page after clicking
bounce_rate                 — fraction who immediately returned to results
scroll_depth                — how far users scrolled (proxy for engagement)
return_visit_rate           — did users come back to this page later?
abandonment_rate            — did users stop searching after this result? (positive signal)
```

**CTR calculation:**

```
Over 30 days for query "best running shoes":
  D1 shown 1000 times, clicked 340 times → CTR = 340/1000 = 0.340
  D2 shown 1000 times, clicked 120 times → CTR = 120/1000 = 0.120
  D3 shown 1000 times, clicked 280 times → CTR = 280/1000 = 0.280

CTR is a strong LTR feature, but raw CTR has position bias:
  results at rank 1 get clicked more just because they're at rank 1
  (not because they're better)

Position-corrected CTR (via Inverse Propensity Scoring):
  corrected_CTR(d, pos) = raw_CTR(d, pos) / P(click | shown at pos, not relevant)
  
  where propensity P(click | pos) is estimated from randomized experiments
  or EM algorithms on click logs.
```

### Category 4 — Freshness Features

```
document_age_days       — days since publication
last_modified_days      — days since last update
publication_velocity    — how fast is this domain publishing?
query_freshness_need    — does this query need fresh results?
                          (news queries: high; recipe queries: low)
```

**Freshness score example:**

```
freshness(d, q) = query_freshness_weight(q) × time_decay(d)

time_decay(d) = e^(-λ × age_in_days)  where λ controls decay rate

For news query (λ=0.1):
  article from yesterday:  e^(-0.1×1)  = 0.905  (high freshness)
  article from last week:  e^(-0.1×7)  = 0.497  (moderate)
  article from last year:  e^(-0.1×365) ≈ 0.000  (effectively 0)

For evergreen query (λ=0.001):
  article from yesterday:  e^(-0.001×1)  = 0.999
  article from last year:  e^(-0.001×365) = 0.694  (still high)
```

### Full feature vector for LambdaMART

```
document D for query Q — feature vector x:

x = [
  BM25_total(Q, D),        # 8.2
  BM25_title(Q, D),        # 6.1
  BM25_anchor(Q, D),       # 4.3
  dense_sim(Q, D),         # 0.87
  exact_title_match(Q, D), # 1   (binary)
  query_coverage(Q, D),    # 0.75
  PageRank(D),             # 0.71
  in_link_count(D),        # 4200
  document_length(D),      # 1850
  spam_score(D),           # 0.02
  CTR(Q, D),               # 0.34  (position-corrected)
  dwell_time(Q, D),        # 142   (seconds)
  freshness(D, Q),         # 0.83
  domain_authority(D),     # 0.88
]

LambdaMART maps this → scalar score → rank documents by score
```

---

## Part 7 — Training Data: Relevance Judgments

Where do the labels come from?

### Human judgment (editorial labels)

Search engines hire human raters to evaluate (query, document) pairs on a scale:

```
0 = Perfectly Useless
1 = Bad
2 = Fair
3 = Good
4 = Excellent
5 = Perfect

Google's standard: "Perfect" means the document is the ideal answer to this query.
These labels are called "editorial judgments" or "qrels" (query relevance judgments).
```

Cost: expensive and slow. Google has thousands of raters. TREC provides public qrel datasets.

### Implicit feedback (click logs)

At scale, behavior signals are gold:

```
user searches "diabetes treatment"
user sees results [D1, D2, D3, D4, D5]
user clicks D3, reads for 4 minutes, then stops searching

Inference:
  D3: positive signal (clicked + long dwell + satisfied)
  D1, D2 (above D3): probably worse than D3  (skipped despite being shown)
  D4, D5 (below D3): uncertain (may not have been seen)
```

This is called **click-through data** and is used to create millions of pairwise training labels automatically. It's noisy but abundant.

### The position bias problem

```
Rank 1 gets clicked 34% of the time on average.
Rank 10 gets clicked 2% of the time.

If we naively treat "click = relevant," rank 1 documents
appear relevant even when they're not — just because they were ranked there.

Solutions:
  1. Inverse Propensity Scoring (IPS)
  2. Counterfactual LTR (randomize rankings for a sample of queries)
  3. EM-based propensity estimation
```

---

## Part 8 — The Three Paradigms — Side-by-Side Comparison

| Property | Pointwise | Pairwise | Listwise |
|---|---|---|---|
| What loss operates on | Single documents | Document pairs | Entire ranked list |
| Training target | Predict relevance score | Predict which doc ranks higher | Optimize NDCG/MAP directly |
| Key model | Linear regression, SVM | RankNet | LambdaMART, ListNet |
| Handles position importance? | No | No | Yes (NDCG weights top positions) |
| Training data | (query, doc, label) | (query, docA, docB, label) | (query, [docs], [labels]) |
| Training pairs | n per query | O(n²) per query | 1 list per query |
| Practical performance | Weakest | Better | Best |
| Used in production | Rarely (baseline) | Historical | Yes (LambdaMART) |

---

## Part 9 — Full Production LTR Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  OFFLINE (training)                                               │
│                                                                   │
│  1. Collect training data                                         │
│     Human raters → qrels                                         │
│     Click logs → implicit pairwise labels                        │
│                                                                   │
│  2. Feature engineering                                           │
│     For each (query, document) pair: compute feature vector x    │
│     BM25, PageRank, CTR, freshness, dense_sim, ...               │
│                                                                   │
│  3. Train LambdaMART                                              │
│     Optimize NDCG@10 via lambda gradients + GBDT                 │
│     Tune: tree depth, num_leaves, learning_rate, num_trees        │
│     Evaluate on held-out query set: measure NDCG@10 improvement  │
└──────────────────────────────────────────────────────────────────┘
                              ↓  trained model
┌──────────────────────────────────────────────────────────────────┐
│  ONLINE (serving)                                                 │
│                                                                   │
│  1. Retrieval stage (Chapter 6)                                   │
│     BM25 + dense → RRF → top 200 candidates                      │
│                                                                   │
│  2. Feature computation (real-time)                               │
│     For each of 200 candidates: compute BM25, dense_sim, CTR,    │
│     freshness, PageRank (precomputed), ...                        │
│                                                                   │
│  3. LambdaMART scoring                                            │
│     Run 200 feature vectors through tree ensemble                 │
│     Get 200 scores in ~5ms (tree inference is fast)              │
│                                                                   │
│  4. Return top 10 to user                                         │
└──────────────────────────────────────────────────────────────────┘
```

**Total latency breakdown:**
- Retrieval (BM25 + ANN parallel): ~50ms
- Feature computation (200 docs): ~30ms
- LambdaMART inference (200 docs): ~5ms
- Total: ~85ms

---

## Part 10 — Interview Mastery

### The question map

| If they ask about... | Lead with... |
|---|---|
| Three paradigms | Pointwise→pairwise→listwise, each solves the previous one's failure |
| Why not pointwise | Loss is absolute, not relative; wrong rankings can still have low loss |
| Why not pairwise | All pairs weighted equally; rank 1 swap = rank 100 swap |
| How LambdaMART works | Lambda gradients scale pairwise gradient by \|ΔNDCG\|; GBDT fits these |
| Why NDCG not differentiable | Ranks are step functions; can't backprop |
| What features matter | BM25, PageRank, CTR (position-corrected), dwell time, freshness |
| Position bias in click data | IPS or counterfactual LTR to debias |

### Six questions you must answer cold

**Q: Explain the three LTR paradigms and why we moved from pointwise to listwise.**

Pointwise treats each document independently — train a regression model to predict relevance score. Fails because ranking is relative: a model minimizing MSE on absolute scores can still rank irrelevant documents above relevant ones. Pairwise fixes this by training on document pairs — the loss penalizes wrong orderings directly. Fails because it weights all pairs equally, ignoring that rank-1 errors are catastrophic and rank-100 errors are invisible. Listwise fixes this by computing NDCG-based gradients that weight position importance — errors near the top get large gradients, errors near the bottom get tiny ones.

**Q: Why can't you optimize NDCG directly with gradient descent?**

NDCG depends on the rank of each document, which is a step function of scores — it changes discontinuously when one document crosses another in score. Step functions have gradient zero everywhere and infinity at jump points. Standard backpropagation can't compute useful gradients. LambdaRank sidesteps this by directly defining what the gradient *should* be (scaling the RankNet pairwise gradient by |ΔNDCG|) without deriving it from a differentiable loss. It works empirically even though there's no closed-form loss it's optimizing.

**Q: Walk me through how LambdaMART trains on one iteration.**

For each query in the training set, compute scores for all documents using the current model. For each pair (Di, Dj) where Di is more relevant, compute the lambda: the product of the RankNet gradient and |ΔNDCG from swapping their ranks|. Sum lambda contributions for each document across all pairs. This gives every document a target gradient direction. Fit a shallow decision tree to predict these lambdas from the feature vectors. Add the tree to the ensemble with a small learning rate. Repeat 1000 times. Each tree makes a small correction to the score function, incrementally improving NDCG.

**Q: What features would you use in a ranking model for a medical search engine?**

Query-document relevance: BM25 on title, body, and MeSH term fields separately; dense similarity from a biomedical embedding model (BioBERT). Document quality: journal impact factor, citation count, PubMed central rank, author h-index, publication date. Query-specific: query type classification (symptom lookup vs drug lookup vs clinical procedure) to apply appropriate feature weights. User behavior: click rate and dwell time among healthcare professionals (segmented to avoid general-public noise). Freshness: medical guidelines update frequently — articles older than 5 years on treatment topics should be freshness-penalized.

**Q: How do you handle position bias in click data when training an LTR model?**

Raw CTR is biased — rank 1 gets 17× more clicks than rank 10 regardless of relevance. If you use raw CTR as a training signal, the model learns to put rank 1 documents at rank 1 again, amplifying the bias. Solutions: (1) Counterfactual LTR — randomly shuffle rankings for a small fraction of queries to measure true relevance independent of position; (2) Inverse Propensity Scoring — estimate the probability of clicking position k given no relevance, then divide observed clicks by this propensity to get a relevance-adjusted signal; (3) Unbiased LambdaMART — directly incorporate propensity weights into the lambda gradient computation.

**Q: How would you evaluate whether adding CTR as a feature actually improved your ranking model?**

Offline evaluation: measure NDCG@10 on a held-out set of queries with editorial labels. Compare model with CTR feature vs without — if NDCG improves by ≥1% on a large enough test set, it's signal. Also measure feature importance from the GBDT (split gain or permutation importance). Online evaluation: A/B test — 5% of traffic to model with CTR, 5% to model without. Measure online metrics: user satisfaction (via click-through on top results), task completion (search abandonment rate as a proxy for finding the answer), and long-click rate (dwell time > 30s). Be careful of circular feedback: CTR trained from current clicks improves rankings → changes future clicks → need fresh editorial labels to measure ground truth.

---

## Summary — What to Remember

```
1. LTR treats ranking as ML: learn a function f(features) → score from labeled data
2. Pointwise: predict relevance score per doc — fails because loss is absolute not relative
3. Pairwise: predict which doc ranks higher — fails because all pairs equally weighted
4. Listwise: optimize NDCG directly — requires lambda trick because NDCG not differentiable
5. LambdaRank: scale pairwise gradient by |ΔNDCG| — top-position errors get large gradients
6. LambdaMART: LambdaRank + gradient boosted trees — production standard
7. Features: BM25/dense (relevance), PageRank (quality), CTR+dwell (behavior), freshness
8. Position bias in click logs: use IPS or counterfactual experiments to debias
9. Evaluation: NDCG@10 offline, click metrics + A/B test online
10. Pipeline: retrieval (BM25+dense+RRF) → feature computation → LambdaMART → top-k
```

---

## Quick Reference

| Concept | Formula / Value |
|---|---|
| NDCG@k | `DCG@k / IDCG@k` |
| DCG@k | `Σ (2^rel(i) - 1) / log₂(i+1)` for i=1 to k |
| RankNet loss | `-log(σ(sᵢ - sⱼ))` when Di should rank above Dj |
| Sigmoid | `σ(x) = 1/(1 + e⁻ˣ)` |
| Lambda gradient | `λᵢⱼ = -σ̄ᵢⱼ × \|ΔNDCGᵢⱼ\|` |
| LambdaMART update | `Fₜ(x) = Fₜ₋₁(x) + η × hₜ(x)` where hₜ fits lambdas |
| PageRank | `PR(d) = (1-d)/N + d × Σ PR(v)/out_degree(v)` |
| Freshness decay | `e^(-λ × age_days)` |
| Position bias fix | Inverse Propensity Scoring or counterfactual ranking |
| Industry standard | LambdaMART (LightGBM/XGBoost implementation) |
