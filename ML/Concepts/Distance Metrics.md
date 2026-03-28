# Distance Metrics 

---

## Table of Contents

1. [Intuition](#intuition)
2. [Formal Definition — The Four Metric Axioms](#formal-definition)
3. [Family 1 — Vector Space Metrics](#family-1--vector-space-metrics)
4. [Family 2 — Probability Distribution Metrics](#family-2--probability-distribution-metrics)
5. [Family 3 — String / Sequence Metrics](#family-3--string--sequence-metrics)
6. [Family 4 — Set-Based Metrics](#family-4--set-based-metrics)
7. [Family 5 — Correlation-Based Metrics](#family-5--correlation-based-metrics)
8. [Decision Map — When to Use What](#decision-map--when-to-use-what)
9. [Key Properties Comparison Table](#key-properties-comparison-table)
10. [Practical ML Engineering Context](#practical-ml-engineering-context)
11. [Pitfalls and Failure Modes](#pitfalls-and-failure-modes)
12. [FAANG Interview Questions](#faang-interview-questions)
13. [FAANG Interview Answers](#faang-interview-answers)
14. [Further Reading](#further-reading)

---

## Intuition

A **distance metric** is a formal way to measure difference between two objects.

The critical insight is: **"different" means something fundamentally different depending on what you're measuring.**

- Distance between two cities ≠ distance between two text documents ≠ distance between two probability distributions.
- A **bad choice of distance metric** is one of the most common silent killers in ML systems — the model trains fine, loss goes down, but similarity is being measured in a fundamentally wrong way.

---

## Formal Definition

To qualify as a true **metric**, a function $d(x, y)$ must satisfy four axioms:

| Axiom | Formula | Meaning |
|---|---|---|
| Non-negativity | $d(x, y) \geq 0$ | Distance is never negative |
| Identity | $d(x, y) = 0 \iff x = y$ | Zero distance = same point |
| Symmetry | $d(x, y) = d(y, x)$ | A→B equals B→A |
| Triangle Inequality | $d(x, z) \leq d(x, y) + d(y, z)$ | No shortcut longer than the detour |

> ⚠️ **Engineering note:** Some widely-used functions (KL divergence, cosine distance) **violate these axioms**. This matters — KNN tree pruning, clustering, and certain search structures assume the triangle inequality holds.

---

## Family 1 — Vector Space Metrics

> For data in $\mathbb{R}^n$ — real-valued feature vectors.

---

### 1. Euclidean Distance (L2)

**Intuition:** The straight-line ("as the crow flies") distance between two points.

$$d(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

```
ASCII diagram:

y •
  |  \
  |    \  ← Euclidean (straight line)
  |      \
  •--------• x
```

**Properties:**
- Sensitive to scale — large-range features dominate
- Sensitive to outliers (squaring amplifies differences)
- Assumes features contribute equally and independently
- Creates circular "balls" in 2D

**When to use:**
- Data is normalized / standardized
- Features are on similar scales
- KNN, K-Means, PCA space comparisons
- Image embeddings after L2 normalization

**When it fails:**
- High-dimensional spaces (curse of dimensionality)
- Features on vastly different scales without normalization
- Correlated features (ignores covariance)
- Sparse data

---

### 2. Manhattan Distance (L1 / Taxicab)

**Intuition:** Grid-based city walking distance — no diagonal shortcuts.

$$d(x, y) = \|x - y\|_1 = \sum_{i=1}^{n} |x_i - y_i|$$

```
ASCII diagram:

y •
  |  ↑
  |  |  ← Manhattan (grid walk)
  |  •---→ x
```

**Properties:**
- Less sensitive to outliers than L2 (no squaring)
- More robust in high dimensions than L2
- Creates diamond-shaped (rhombus) "balls"
- NOT rotation-invariant

**When to use:**
- Data with outliers
- Grid/routing problems
- Sparse, high-dimensional data
- Robustness preferred over precision

**When it fails:**
- True Euclidean geometry required
- When rotation invariance matters

---

### 3. Minkowski Distance (Generalized $L_p$)

**Intuition:** The parent of L1 and L2 — a single formula parameterized by $p$.

$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

| $p$ | Reduces to |
|---|---|
| $p = 1$ | Manhattan (L1) |
| $p = 2$ | Euclidean (L2) |
| $p \to \infty$ | Chebyshev: $\max_i \|x_i - y_i\|$ |

**Chebyshev Distance** ($L_\infty$): Maximum difference across any single dimension. Used in chess king-move problems, robotics, warehouse systems.

---

### 4. Cosine Similarity / Distance

**Intuition:** Forget how far apart — ask: **do they point in the same direction?** Captures angular similarity, ignoring magnitude.

$$\text{cos\_sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \cdot \sqrt{\sum_i y_i^2}}$$

$$d_{\cos}(x, y) = 1 - \text{cos\_sim}(x, y)$$

| Value | Angle | Meaning |
|---|---|---|
| $+1$ | $0°$ | Identical direction |
| $0$ | $90°$ | **Orthogonal — no relationship** |
| $-1$ | $180°$ | Opposite direction |

> ⚠️ Cosine *distance* ($1 - \cos$) does **NOT** satisfy the triangle inequality — it is a dissimilarity measure, not a true metric.

**When to use:**
- NLP / text (TF-IDF, word/sentence embeddings)
- Recommender systems (user/item embeddings)
- Sparse, high-dimensional data
- When magnitude doesn't matter, only direction

**When it fails:**
- When magnitude matters
- Zero vectors (undefined — division by zero)
- Near-zero vectors produce unstable results

**Engineering note:** In FAISS, Annoy, ScaNN — cosine is dominant for embedding retrieval. After L2-normalizing embeddings: **L2 distance = cosine distance** (mathematically equivalent). This is a common production trick.

---

### 5. Dot Product Similarity

**Intuition:** Raw inner product — measures both direction AND magnitude.

$$\text{score}(x, y) = x \cdot y = \sum_i x_i y_i$$

Not a proper metric (not non-negative, unbounded). But used everywhere in practice.

**Where it appears:**
- Transformer attention: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- Two-tower recommendation models (dot product at retrieval layer)
- Neural collaborative filtering

**Dot product vs cosine in recsys:**
- Dot product rewards popular items (high magnitude) — good for relevance + popularity
- Cosine ignores popularity — good for pure similarity

---

### 6. Mahalanobis Distance

**Intuition:** Euclidean distance corrected for feature correlations and scale. Asks: "how many standard deviations apart are these two points, accounting for the shape of the data?"

$$d_M(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$$

Where $\Sigma$ is the **covariance matrix** of the data.

```
ASCII diagram:

Standard Euclidean:     Mahalanobis:
○ ○ ○                   ○  ○  ○
○ ● ○   (circles)       ○  ●  ○   (ellipses aligned
○ ○ ○                    ○ ○ ○     with data covariance)
```

**Special case:** If $\Sigma = I$, Mahalanobis = Euclidean.

**Properties:**
- Scale-invariant (automatically handles different-scale features)
- Rotation-invariant
- Accounts for feature correlations
- Requires computing and inverting $\Sigma$: $O(n^2)$ storage, $O(n^3)$ inversion

**When to use:**
- Anomaly detection (Mahalanobis score = classic outlier score)
- Correlated features
- Different-scale features without normalization option

**When it fails:**
- $n_\text{features} > n_\text{samples}$ → $\Sigma$ is singular, cannot invert
- Non-Gaussian data
- Very high dimensions (intractable)

---

### 7. Gower Distance (Mixed-Type Data)

**Intuition:** A metric designed specifically for datasets with **mixed types** — numeric and categorical features together.

$$d_G(x, y) = \frac{1}{p} \sum_{j=1}^{p} d_j(x_j, y_j)$$

Where per feature:
- **Numeric:** $d_j = \frac{|x_j - y_j|}{\text{range}_j}$ (normalized L1)
- **Categorical:** $d_j = 0$ if same, $1$ if different

**When to use:**
- KNN on mixed-type tabular data (age + income + zip code)
- Any dataset where you cannot cleanly one-hot encode categoricals

---

## Family 2 — Probability Distribution Metrics

> For measuring difference between two probability distributions $P$ and $Q$.

---

### 8. KL Divergence (Kullback-Leibler)

**Intuition:** How much information is lost when using $Q$ to approximate $P$? Measures the "surprise" of encoding $P$-distributed data with a $Q$-based code.

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx \quad \text{(continuous)}$$

**Critical properties:**

| Property | Value |
|---|---|
| Symmetric? | ❌ No |
| True metric? | ❌ No |
| Always $\geq 0$? | ✅ Yes (Gibbs' inequality) |
| $= 0$ iff $P = Q$? | ✅ Yes |

**Asymmetry behavior:**

| Direction | Weighting | Behavior |
|---|---|---|
| $D_{KL}(P\|Q)$ — Forward KL | Weighted by $P$ | **Mean-seeking** — $Q$ must cover all modes of $P$ |
| $D_{KL}(Q\|P)$ — Reverse KL | Weighted by $Q$ | **Mode-seeking** — $Q$ collapses to one mode of $P$ |

```
P is bimodal:        █      █

Forward KL (P‖Q):   ███████████   Q spreads across both modes
Reverse KL (Q‖P):        █        Q picks ONE mode and sits there
```

**Where it appears:**
- VAE loss (ELBO = reconstruction - KL term)
- Cross-entropy loss = KL divergence + entropy of true labels
- RL / RLHF: KL penalty in PPO to prevent policy collapse
- Mutual information: $I(X;Y) = D_{KL}(P_{XY} \| P_X P_Y)$

**Derivation — Cross-entropy = KL + Entropy:**

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

$$D_{KL}(P\|Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x)\log P(x) - \sum_x P(x)\log Q(x)$$

$$D_{KL}(P\|Q) = -H(P) + H(P,Q)$$

$$\therefore H(P,Q) = H(P) + D_{KL}(P\|Q)$$

Minimizing cross-entropy loss = minimizing KL divergence (since $H(P)$ is constant w.r.t. model parameters).

---

### 9. Jensen-Shannon Divergence (JSD)

**Intuition:** The symmetric version of KL. Averages both KL directions through a mixture distribution $M = \frac{P+Q}{2}$.

$$JSD(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{1}{2}(P+Q)$$

**Properties:**
- ✅ Symmetric: $JSD(P\|Q) = JSD(Q\|P)$
- ✅ Bounded: $0 \leq JSD \leq 1$ (log base 2)
- ✅ $\sqrt{JSD}$ is a **true metric**
- Used in original GAN as the implicit training objective

---

### 10. Wasserstein Distance (Earth Mover's Distance)

**Intuition:** Minimum *work* (mass × distance) needed to transform distribution $P$ into distribution $Q$. Like moving piles of dirt — the EMD.

$$W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| \, dx \quad \text{(1D, via CDFs)}$$

$$W_p(P, Q) = \left(\inf_{\gamma \in \Gamma(P,Q)} \int \|x-y\|^p \, d\gamma(x,y)\right)^{1/p} \quad \text{(general)}$$

**Why Wasserstein beats KL/JSD for non-overlapping distributions:**

```
P:  ████
Q:          ████

KL Divergence  = ∞         (log(0) appears)
JSD            = 1          (saturates at maximum)
Wasserstein    = finite ✅   (meaningful geometric distance)
```

**Where it appears:**
- WGAN (Wasserstein GAN) — solves mode collapse and training instability
- Distribution shift detection in production ML
- Word Mover's Distance (WMD) for NLP
- Optimal transport research

**Computational cost:** Solving OT = $O(n^3 \log n)$. Sinkhorn algorithm provides fast approximation used in practice.

---

## Family 3 — String / Sequence Metrics

---

### 11. Hamming Distance

**Intuition:** Number of positions where two **equal-length** strings differ.

$$d_H(x, y) = \sum_{i=1}^{n} \mathbf{1}[x_i \neq y_i]$$

**Example:** `"karolin"` vs `"kathrin"` → differ at 3 positions → Hamming = 3

**When to use:**
- Error detection/correction (coding theory)
- Binary feature vector comparison
- DNA sequence comparison (no insertions/deletions)
- LSH (locality-sensitive hashing)

**When it fails:** Strings of different length (undefined).

---

### 12. Edit Distance (Levenshtein)

**Intuition:** Minimum number of single-character insertions, deletions, or substitutions to transform string $A$ into string $B$.

**Example:** `"kitten"` → `"sitting"` = 3 operations.

**Dynamic programming recurrence:**

$$dp[i][j] = \begin{cases} j & \text{if } i=0 \\ i & \text{if } j=0 \\ dp[i-1][j-1] & \text{if } x_i = y_j \\ 1 + \min\big(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]\big) & \text{otherwise} \end{cases}$$

**Complexity:** $O(mn)$ time and space.

**When to use:**
- Spell checking, autocorrect
- DNA/protein sequence alignment
- OCR post-processing
- Fuzzy string matching in search

---

## Family 4 — Set-Based Metrics

---

### 13. Jaccard Similarity / Distance

**Intuition:** Fraction of combined elements shared by two sets.

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}, \quad d_J(A,B) = 1 - J(A,B)$$

Range: $[0, 1]$, where $1$ = identical, $0$ = no overlap.

**Engineering note — MinHash:**
Computing exact Jaccard is expensive on large sets. **MinHash** approximates it in $O(k)$ time using $k$ hash functions:

$$P[\min h(A) = \min h(B)] = J(A, B)$$

This is how Google, Facebook detect near-duplicate web pages at scale.

**When to use:**
- Document similarity (sets of words/shingles)
- Near-duplicate detection (MinHash + LSH)
- Recommender systems (user item sets)
- Biological data (gene sets)

---

## Family 5 — Correlation-Based Metrics

---

### 14. Pearson Correlation Distance

$$d(x, y) = 1 - r_{xy} = 1 - \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2 \cdot \sum_i(y_i-\bar{y})^2}}$$

**When to use:**
- Gene expression clustering
- Financial time series
- When you care about the *shape* of the signal, not magnitude or offset

---

## Decision Map — When to Use What

```
What is your data type?
│
├── Continuous numeric vectors
│   ├── Same scale, uncorrelated features?           → Euclidean (L2)
│   ├── Outliers or high-dimensional?                → Manhattan (L1)
│   ├── Correlated features / different scales?      → Mahalanobis
│   ├── Mixed types (numeric + categorical)?         → Gower Distance
│   └── Direction only matters (embeddings, text)?  → Cosine Similarity
│
├── Recsys / ranking (magnitude + direction)?        → Dot Product
│
├── Probability distributions
│   ├── Approximating P with Q (asymmetric ok)?     → KL Divergence
│   ├── Need symmetry + bounded?                    → Jensen-Shannon
│   └── Non-overlapping support / geometry needed?  → Wasserstein
│
├── Binary strings, error correction?               → Hamming
│
├── Text strings with insertions/deletions?         → Levenshtein
│
├── Sets / sparse binary?                           → Jaccard / MinHash
│
└── Signal shape / time series?                     → Pearson / DTW
```

---

## Key Properties Comparison Table

| Metric | True Metric? | Symmetric? | Scale Invariant? | Best For |
|---|---|---|---|---|
| Euclidean (L2) | ✅ | ✅ | ❌ | Normalized numeric vectors |
| Manhattan (L1) | ✅ | ✅ | ❌ | Robust / high-dim data |
| Chebyshev (L∞) | ✅ | ✅ | ❌ | Grid / chessboard problems |
| Cosine Distance | ❌ | ✅ | ✅ | Text, embeddings, sparse data |
| Dot Product | ❌ | ✅ | ❌ | Recsys, transformers |
| Mahalanobis | ✅ | ✅ | ✅ | Correlated/mixed-scale features |
| Gower | ✅ | ✅ | ✅ | Mixed-type tabular data |
| KL Divergence | ❌ | ❌ | — | Distribution approximation |
| JSD | ❌ | ✅ | — | Symmetric dist. comparison |
| Wasserstein | ✅ | ✅ | — | Non-overlapping distributions |
| Hamming | ✅ | ✅ | — | Equal-length strings/bits |
| Levenshtein | ✅ | ✅ | — | String edit operations |
| Jaccard | ✅ | ✅ | — | Sets, sparse binary data |

---

## Practical ML Engineering Context

### 1. Embedding-Based Retrieval (Google, Meta, Pinterest)

- Billions of items embedded in vector space
- Query → query embedding → nearest neighbor search
- **FAISS:** supports L2 and inner product natively
- **Annoy:** uses cosine; **ScaNN:** L2 and dot product
- **Key trick:** L2-normalize embeddings → L2 distance = cosine distance (no extra cost)
- Wrong metric → silent retrieval quality degradation

### 2. Recommender Systems

- Two-tower models: dot product at retrieval (popularity-aware)
- Item-item similarity: cosine (pure semantic similarity)
- User-item collaborative filtering: cosine on user vectors
- A/B tests measure downstream impact: CTR, conversion, dwell time

### 3. Anomaly Detection (Uber, Netflix)

- Mahalanobis score on feature vectors → outlier flagging
- KL divergence between training vs serving feature distributions → data drift alerts
- Wasserstein for comparing continuous feature distributions across time

### 4. NLP Search (Google, Bing)

- Sparse retrieval: BM25 (term-based, not a metric per se)
- Dense retrieval: cosine on BERT/sentence-transformer embeddings
- Fuzzy search: Levenshtein for typo-tolerant query matching

### 5. Generative Models

- Vanilla GAN: implicitly minimizes JSD → mode collapse when supports don't overlap
- WGAN: minimizes Wasserstein distance → stable gradients, better convergence
- VAE: KL divergence in ELBO regularizes latent space

### 6. Production Distribution Monitoring

- Monitor KL or Wasserstein between training and serving distributions
- Alert when drift exceeds threshold → trigger retraining pipeline
- PSI (Population Stability Index) is a related practical measure in industry

### 7. Locality-Sensitive Hashing (LSH)

Different hash families for different metrics:

| Metric | LSH Family |
|---|---|
| Cosine | Random projection (sign of $\langle x, r\rangle$) |
| Jaccard | MinHash |
| L2 | Random projection with threshold |
| Hamming | Bit sampling |

---

## Pitfalls and Failure Modes

| Pitfall | Metric Affected | Fix |
|---|---|---|
| Features on different scales | Euclidean, Manhattan | Z-score normalize or min-max scale |
| Categorical features fed raw | Any vector metric | One-hot encode, target encode, or use Gower |
| KL divergence with zero-probability regions | KL | Add Laplace smoothing; use JSD or Wasserstein |
| Cosine on zero vector | Cosine | Check for and handle zero vectors before computing |
| Mahalanobis with singular covariance | Mahalanobis | Use pseudo-inverse; reduce dimensions first |
| Euclidean in high dimensions | Euclidean | Use cosine; apply dimensionality reduction |
| Treating cosine distance as a true metric | Cosine | Don't use with tree-based ANN structures assuming triangle inequality |
| KL asymmetry ignored | KL | Explicitly decide: forward or reverse KL? What behavior do you want? |
| Dot product biased toward high-norm items | Dot Product | Normalize if popularity bias is undesired |

---

## FAANG Interview Questions

### 🟡 Medium

**Q1.** You're building a KNN classifier with features: age (0–100), income ($0–$500K), and zip code (categorical). What distance metric do you use? What preprocessing is required?

**Q2.** Cosine similarity between two non-zero vectors is 0. What does this mean geometrically? Give a concrete 2D example.

**Q3.** Prove KL divergence is asymmetric. Give a numeric example where $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$.

### 🔴 Hard

**Q4.** Design the retrieval layer for a recommendation system with 500M item embeddings and 100ms latency budget. Walk through your metric choice, indexing structure, and tradeoffs. Why might you choose dot product over cosine?

**Q5.** Derive why cross-entropy loss is equivalent to minimizing KL divergence between the true label distribution and the model's predicted distribution.

**Q6.** Your production model's input feature distribution has drifted. How do you detect this automatically? What metric, why, and what are the practical challenges?

### 🔵 Research-Level

**Q7.** Why does Wasserstein distance work better than JSD as a GAN training objective, especially early in training? What pathological behavior does JSD produce that Wasserstein avoids?

**Q8.** The curse of dimensionality makes Euclidean distance meaningless in very high dimensions. Explain why, and describe what happens to the ratio of max to min distances as dimensionality grows.

---

## FAANG Interview Answers

### Q1 — KNN with Mixed Features

**Full answer:**

Raw Manhattan is insufficient without normalization — income ($0–$500K) would dominate age (0–100) by 5000×. Age contributes ~1 unit, income contributes ~100,000 units. KNN degenerates to "find nearest income."

**Preprocessing:**
1. Z-score normalize age and income: $x' = (x - \mu)/\sigma$
2. Zip code is nominal categorical — **cannot** be used numerically. Options:
   - Convert to `(latitude, longitude)` then normalize → best for physical proximity
   - Target encode → mean of target per zip (risk of leakage)
   - One-hot encode → 40K+ sparse dimensions (impractical for KNN)

**Best metric:** **Gower Distance** — natively handles mixed types. Normalizes numeric features by range, uses 0/1 for categoricals.

After normalization, L1 or L2 both work. L1 preferred for robustness against income outliers.

---

### Q2 — Cosine Similarity = 0

Cosine similarity = 0 means the vectors are **orthogonal (perpendicular)** — at 90° to each other. This indicates **zero directional relationship**.

**Common mistake:** Thinking it means same direction (that is $\cos = 1$).

**Concrete example:**
$$x = [1, 0], \quad y = [0, 1]$$
$$\cos(x, y) = \frac{(1)(0) + (0)(1)}{\sqrt{1} \cdot \sqrt{1}} = 0$$

In NLP: cosine = 0 between two TF-IDF vectors means zero shared vocabulary — completely unrelated topics.

---

### Q3 — KL Divergence Asymmetry

**Structural reason:** The weighting distribution in front of the log flips:

$$D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} \quad \text{(weighted by P)}$$

$$D_{KL}(Q\|P) = \sum_x Q(x)\log\frac{Q(x)}{P(x)} \quad \text{(weighted by Q)}$$

**Behavioral consequence:**
- Forward KL $(P\|Q)$: penalizes $Q$ for missing mass where $P$ is high → **mean-seeking**, $Q$ covers all modes
- Reverse KL $(Q\|P)$: penalizes $Q$ for having mass where $P$ is low → **mode-seeking**, $Q$ collapses to one mode

This is why VAEs use forward KL (want to cover all data modes) and why some generative models exhibit mode collapse (optimizing reverse KL).

**Relationship to cross-entropy:**

$$H(P, Q) = H(P) + D_{KL}(P\|Q)$$

Minimizing cross-entropy = minimizing KL (entropy of true labels is constant w.r.t. model parameters).

---

### Q4 — 500M Embedding Retrieval System

**Metric choice:**
- **Dot product** preferred over cosine in two-tower recsys because:
  - Trained end-to-end with dot product objective → consistent
  - Rewards popular/high-engagement items (high norm) — often desirable
  - Cosine removes this signal
- If pure semantic similarity: use cosine (L2-normalize embeddings → MIPS reduces to L2 search)

**Indexing:**
- **FAISS IVF-PQ** (Inverted File + Product Quantization):
  - Cluster 500M items into ~$\sqrt{N}$ Voronoi cells
  - At query time: search only top $k$ cells
  - PQ compresses vectors → fits in memory
- **HNSW** (Hierarchical Navigable Small World graphs): higher recall, higher memory

**Tradeoffs:**

| Approach | Recall | Latency | Memory |
|---|---|---|---|
| Exact search | 100% | Too slow | High |
| IVF-PQ | ~90–95% | < 10ms | Low |
| HNSW | ~98% | < 5ms | High |

**Latency budget breakdown (100ms total):**
- Feature extraction: ~10ms
- ANN retrieval: ~10ms
- Ranking (scoring top-K): ~50ms
- Network + overhead: ~30ms

---

### Q5 — Cross-Entropy = KL Divergence

Let $P$ = true label distribution (one-hot for classification), $Q_\theta$ = model predictions.

$$H(P, Q_\theta) = -\sum_x P(x) \log Q_\theta(x)$$

$$D_{KL}(P\|Q_\theta) = \sum_x P(x)\log\frac{P(x)}{Q_\theta(x)} = \sum_x P(x)\log P(x) - \sum_x P(x)\log Q_\theta(x)$$

$$D_{KL}(P\|Q_\theta) = -H(P) + H(P, Q_\theta)$$

Since $H(P)$ is constant w.r.t. model parameters $\theta$:

$$\arg\min_\theta H(P, Q_\theta) = \arg\min_\theta D_{KL}(P\|Q_\theta)$$

**Conclusion:** Minimizing cross-entropy loss is exactly minimizing the forward KL divergence from the true distribution to the model's predicted distribution. ∎

---

### Q6 — Production Distribution Drift Detection

**Metric choice:**
- **KL Divergence:** Fast to compute, interpretable, but requires binning continuous features and undefined when support mismatch
- **Wasserstein (W1):** Handles continuous distributions natively, works even with non-overlapping support, more geometrically meaningful
- **PSI (Population Stability Index):** Industry standard in finance — binned KL variant, interpretable thresholds (PSI > 0.2 = significant drift)

**Practical challenges:**
1. **Reference distribution selection:** Use recent window or full training set?
2. **Binning for continuous features:** Bin boundaries affect sensitivity
3. **High cardinality categoricals:** Distribution comparison is expensive
4. **Multiple features:** Need multiple tests → multiple comparisons problem
5. **Latency of monitoring:** Computing Wasserstein at scale is expensive → use Sinkhorn approximation
6. **Alert thresholds:** How much drift triggers retraining? Calibrate against known degradation events

**Production pipeline:**
```
Serving logs → Feature store → Drift monitor (KL/Wasserstein per feature)
                                        ↓ alert
                              Retraining trigger → Model refresh pipeline
```

---

### Q7 — Wasserstein vs JSD for GANs

**JSD failure mode (early training):**

When generator $G$ is poorly initialized, $P_\text{real}$ and $P_\text{fake}$ have **non-overlapping support** (live on different low-dimensional manifolds in high-dimensional space).

When supports don't overlap:
$$JSD(P_\text{real}\|P_\text{fake}) = \log 2 \quad \text{(constant, maximum)}$$

The gradient of JSD w.r.t. generator parameters = **0**. Generator receives no learning signal. Training stalls or becomes unstable.

**Wasserstein solution:**

Wasserstein distance remains **finite and smooth** even when supports don't overlap — it measures the geometric distance between them. Gradients are meaningful everywhere.

WGAN uses the **Kantorovich-Rubinstein duality** to compute $W_1$ efficiently:

$$W_1(P, Q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x\sim P}[f(x)] - \mathbb{E}_{x\sim Q}[f(x)]$$

Where $f$ is a 1-Lipschitz function (the critic). Enforced via weight clipping or gradient penalty (WGAN-GP).

**Summary:**

| | Vanilla GAN (JSD) | WGAN (Wasserstein) |
|---|---|---|
| Overlapping support | Works | Works |
| Non-overlapping support | Gradient = 0 ❌ | Meaningful gradient ✅ |
| Training stability | Unstable | More stable |
| Mode collapse | Common | Reduced |

---

### Q8 — Curse of Dimensionality and Euclidean Distance

**The core result:**

As dimensionality $d \to \infty$, for random points in $[0,1]^d$:

$$\frac{d_\text{max} - d_\text{min}}{d_\text{min}} \to 0$$

All pairwise distances **concentrate** — the ratio of the maximum to minimum distance approaches 1. Every point becomes equally distant from every other point. Nearest neighbor loses meaning.

**Intuition via variance:**

For $n$ i.i.d. features, each with variance $\sigma^2$:

$$\mathbb{E}[d(x,y)^2] = \sum_{i=1}^{n} \mathbb{E}[(x_i - y_i)^2] = n \cdot 2\sigma^2$$

Distance grows as $O(\sqrt{n})$. The **variance of distances** also grows, but the **relative spread** ($\text{std}/\text{mean}$) shrinks as $O(1/\sqrt{n})$.

**Practical consequences:**
- KNN in raw pixel space (e.g., 256×256 image = 65K dimensions) → meaningless neighbors
- Must apply dimensionality reduction (PCA, UMAP) or use learned embeddings
- Cosine more robust than Euclidean in high dimensions (angular concentration slower)

---

## Further Reading

| Resource | What to read |
|---|---|
| **Pattern Recognition and ML** — Bishop | Ch. 2 (probability distributions), Ch. 6 (kernel methods) |
| **Elements of Statistical Learning** — Hastie et al. | Ch. 13 (prototype methods, KNN) |
| **Information Theory** — Cover & Thomas | Ch. 2 (entropy, KL divergence) |
| **Optimal Transport** — Villani | Chapters 1–3 for Wasserstein foundations |
| **WGAN paper** — Arjovsky et al. (2017) | Original Wasserstein GAN paper |
| **FAISS paper** — Johnson et al. (2017) | Billion-scale similarity search in practice |
| **Word Mover's Distance** — Kusner et al. (2015) | Wasserstein applied to NLP |
| **Mining Massive Datasets** — Leskovec et al. | Ch. 3 (LSH, Jaccard, MinHash) |
| **CS229 Stanford** | Distance metrics in the context of supervised/unsupervised learning |

---

*Generated for FAANG MLE/DSE interview preparation. Cover all sections before interviews.*
