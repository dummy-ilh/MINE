# Chapter 10: k-Nearest-Neighbor (kNN) Distance-Based Outlier Detection

## 10.1 Motivation

Chapters 6–9 all fit some kind of **global model** first (a covariance matrix, a PCA subspace, a kernel boundary) and then measure how far a point is from that model. kNN-distance detection skips model-fitting almost entirely: it asks a purely **local, non-parametric** question — *"how far is this point from its nearest neighbors, compared to how far other points are from theirs?"* This is the simplest member of the density-based family that will culminate in LOF (Ch.11) — understanding kNN-distance thoroughly is what makes LOF's refinement make sense.

## 10.2 Formal Definition

For a point $x$, let $d_k(x)$ be the distance to its $k$-th nearest neighbor. Two common outlier scores:

**k-distance score:**
$$
\text{score}(x) = d_k(x)
$$
Simply the distance to the $k$-th neighbor — large value means the point is far from its neighborhood, i.e., sits in a sparse region.

**Average kNN distance (more stable, commonly preferred):**
$$
\text{score}(x) = \frac{1}{k}\sum_{i=1}^{k} d(x, x_{(i)})
$$
where $x_{(i)}$ is the $i$-th nearest neighbor of $x$. Averaging over all $k$ neighbors (rather than using only the $k$-th) smooths out sensitivity to the choice of exactly which neighbor happens to land at position $k$.

**Decision rule:** rank all points by their score; flag the top $m$ (or those above a percentile threshold) as outliers. Note there's no natural "chi-square-style" universal threshold here (unlike Ch.6's approach) — thresholds are typically set empirically (e.g., top 1% of scores) since this method makes no distributional assumption at all.

## 10.3 Worked Numerical

2D data points: $A=(1,1)$, $B=(1,2)$, $C=(2,1)$, $D=(2,2)$, $E=(8,8)$ (suspected outlier). Use $k=2$.

**Step 1 — compute pairwise distances from E to all others** (Euclidean):
$$
d(E,A) = \sqrt{(8-1)^2+(8-1)^2} = \sqrt{49+49}=\sqrt{98}\approx9.90
$$
$$
d(E,B) = \sqrt{(8-1)^2+(8-2)^2} = \sqrt{49+36}=\sqrt{85}\approx9.22
$$
$$
d(E,C) = \sqrt{(8-2)^2+(8-1)^2} = \sqrt{36+49}=\sqrt{85}\approx9.22
$$
$$
d(E,D) = \sqrt{(8-2)^2+(8-2)^2} = \sqrt{36+36}=\sqrt{72}\approx8.49
$$

Two nearest neighbors of E: D (8.49) and B or C (9.22, tie — take one, say B).
$$
\text{score}(E) = \frac{8.49+9.22}{2} = 8.855
$$

**Step 2 — compute the same for a "normal" point, say D=(2,2):**
$$
d(D,A)=\sqrt{1+1}=1.414,\quad d(D,B)=\sqrt{1+0}=1.0,\quad d(D,C)=\sqrt{0+1}=1.0,\quad d(D,E)=8.49
$$
Two nearest neighbors of D: B (1.0) and C (1.0).
$$
\text{score}(D) = \frac{1.0+1.0}{2}=1.0
$$

**Result:** $\text{score}(E)=8.855$ vs. $\text{score}(D)=1.0$ — nearly a **9× difference**, clearly separating the isolated point from the tightly-clustered ones, with zero distributional assumptions and zero model-fitting beyond computing pairwise distances.

## 10.4 The Critical Weakness — Varying Density (This Sets Up Chapter 11)

Consider adding a second, naturally sparser cluster far from the first: e.g., points $F=(20,20)$, $G=(21,21)$, $H=(22,20)$ — legitimately spread further apart from each other than cluster $\{A,B,C,D\}$, but still a real, coherent cluster (not outliers).

A point in this sparser cluster, say $G$, will have a **larger** average kNN distance to its neighbors ($F$, $H$) than points in the dense cluster $\{A,B,C,D\}$ have to theirs — purely because cluster $\{F,G,H\}$ is naturally more spread out, **not** because $G$ is actually anomalous. Plain kNN-distance would systematically over-flag every point in the sparser-but-legitimate cluster, while potentially under-flagging a point that's only moderately far from a very dense cluster.

**This is the single most important limitation to state proactively in an interview**: kNN-distance is a *global* density comparison — it doesn't account for the fact that different regions of legitimate data can have naturally different densities. Fixing exactly this problem — by comparing a point's density to its *neighbors'* local densities, rather than to a single global scale — is precisely what Local Outlier Factor (Ch.11) does next.

## 10.5 Diagnosis: When to Use kNN-Distance

| Condition | Recommendation |
|---|---|
| Roughly uniform density normal data, no legitimate multi-density clusters | kNN-distance works well, simple and interpretable |
| Data has legitimate clusters of varying density | Poor fit — will over-flag sparse-but-normal regions (§10.4); use LOF instead |
| Very high-dimensional data | Caution — distance metrics become less meaningful as dimensionality grows (curse of dimensionality, formalized in Ch.13); consider dimensionality reduction first (Ch.8) or tree-based methods (Ch.12) |
| Need something simple, fast to implement, no training/model-fitting step | Good default starting point for EDA before reaching for more complex methods |

## 10.6 Production Considerations
- Naive kNN-distance requires computing distances to *all* other points for every query — $O(n^2)$ for a full batch scoring pass. Approximate nearest-neighbor structures (KD-trees, ball trees for low-dimensional data; HNSW, LSH for high-dimensional/large-scale settings) are standard in production to make this tractable at scale.
- Choice of $k$ matters: too small $k$ makes the score noisy/sensitive to single nearby points; too large $k$ smooths over genuinely local structure and starts to resemble a more global density estimate — this bias-variance tradeoff is analogous to choosing $k$ in kNN classification.
- Because there's no fitted global model to store, "retraining" is really just re-indexing new data into the nearest-neighbor structure — often simpler operationally than retraining a covariance matrix or kernel model (Ch.6-9), which is a genuine practical advantage for streaming settings.

## 10.7 Interview Traps
- Presenting kNN-distance as if it "just works" for any dataset, without proactively flagging the varying-density weakness (§10.4) — this is the most common miss, and the varying-density example (two clusters of different natural spread) is a very standard interview follow-up.
- Confusing kNN-distance outlier *scoring* (this chapter) with kNN *classification* (a supervised algorithm) — they share the same distance/neighbor machinery but solve entirely different problems.
- Not knowing that exact kNN search is $O(n^2)$/query and being unable to name any approximate/indexed alternative for production scale.
- Using a single global score threshold without acknowledging that, unlike Mahalanobis/Grubbs (Ch.4, Ch.6), there's no principled parametric threshold here — cutoffs are inherently empirical (e.g., top X%).

## 10.8 L5-Differentiating Talking Points
- Proactively walking through the two-cluster varying-density failure case (§10.4) *before* being asked, and explicitly naming LOF as the direct fix — this sets up the exact narrative arc the interviewer is likely testing for (do you know why LOF exists, not just what it computes).
- Correctly framing kNN-distance as "a fully non-parametric, purely local density proxy" — connecting back to Ch.1's unifying density-estimation lens (§1.2), reinforcing that every method in this curriculum is ultimately estimating $f(x)$ or a stand-in for it, just with different assumptions and tradeoffs.
- Mentioning approximate nearest-neighbor infrastructure (HNSW, KD-trees) as the practical production answer to the $O(n^2)$ scaling concern — shows systems-level maturity beyond the pure algorithm.

## 10.9 Comprehension Check
1. Why does averaging over all $k$ neighbors' distances tend to be more stable than using only the distance to the $k$-th neighbor alone?
2. Construct (in words) a small example with two clusters of different natural density where kNN-distance would incorrectly flag a normal point in the sparser cluster.
3. Why is there no natural chi-square-style threshold for kNN-distance scores, unlike Mahalanobis distance (Ch.6)?
4. What specifically does Local Outlier Factor change relative to plain kNN-distance to fix the varying-density problem? (You can answer at a conceptual level — full derivation comes in Ch.11.)

---
*Next: Chapter 11 — Local Outlier Factor (LOF): full derivation, reachability distance, worked numerical.*
