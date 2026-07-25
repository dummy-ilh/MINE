# Chapter 8: PCA-Based Outlier Detection (Reconstruction Error, Hotelling's T² & Q-Statistic)

## 8.1 Motivation

Chapters 6–7 handle multivariate outliers by modeling the full covariance structure directly. PCA-based detection takes a different but related angle: **project the data onto its principal components (directions of maximum variance), and look for two distinct kinds of abnormality** — (1) a point that's unusual *within* the normal subspace (still consistent with the data's shape, just extreme along a legitimate axis of variation), and (2) a point that doesn't fit the learned subspace *at all* (its residual, after projecting and reconstructing, is large). This decomposition is especially valuable in **high-dimensional** settings where full-covariance methods (Ch.6–7) become unstable — PCA compresses the dimensionality first, which is exactly why it's the standard next step once $p$ gets large.

## 8.2 PCA Recap (Just Enough for This Chapter)

Given centered data matrix $X$ ($n\times p$), PCA finds orthogonal directions (eigenvectors of the covariance matrix $\Sigma$) sorted by the variance they explain (eigenvalues $\lambda_1\ge\lambda_2\ge\dots\ge\lambda_p$):
$$
\Sigma = V\Lambda V^T
$$
where $V$'s columns are the eigenvectors (principal component directions) and $\Lambda=\text{diag}(\lambda_1,\dots,\lambda_p)$.

A point $x$ is projected into **principal component scores**:
$$
t = V^T(x-\mu)
$$
Keeping only the top $k$ components (those explaining most variance) gives a **reconstruction**:
$$
\hat{x} = \mu + V_k V_k^T (x-\mu)
$$
where $V_k$ contains only the first $k$ eigenvectors.

## 8.3 Two Distinct Outlier Signals from PCA

### 8.3.1 Hotelling's T² (outlier *within* the retained subspace)

Measures how extreme a point is **along the retained principal components**, scaled by each component's variance:
$$
T^2(x) = \sum_{i=1}^{k} \frac{t_i^2}{\lambda_i}
$$
where $t_i$ is the score on the $i$-th principal component. Notice this is structurally identical to Mahalanobis distance (Ch.6) restricted to the top $k$ components — in fact, if $k=p$ (keep all components), $T^2$ is **exactly** Mahalanobis distance, just computed in the rotated PCA basis (rotation doesn't change Mahalanobis distance, since $V$ is orthogonal). So $T^2$ is "Mahalanobis distance, computed only using the directions that matter most."

**Threshold:** under multivariate normality, $T^2$ (appropriately scaled) follows an F-distribution; a common approximation for large $n$ uses:
$$
T^2 > \chi^2_{k,\,1-\alpha}
$$
(same chi-square logic as Ch.6, now with $k$ degrees of freedom instead of $p$).

### 8.3.2 Q-statistic / SPE — Squared Prediction Error (outlier *outside* the retained subspace)

Measures how much a point deviates from the subspace entirely — i.e., how bad the reconstruction is:
$$
Q(x) = \|x - \hat{x}\|^2 = \|(x-\mu) - V_kV_k^T(x-\mu)\|^2
$$
This captures anomalies that don't fit the *learned correlation structure* at all — a fundamentally different failure mode than T², which only cares about being extreme along already-known-important directions.

**Threshold (Jackson-Mudholkar approximation):**
$$
Q_\alpha = \theta_1\left[\frac{c_\alpha\sqrt{2\theta_2 h_0^2}}{\theta_1}+1+\frac{\theta_2 h_0(h_0-1)}{\theta_1^2}\right]^{1/h_0}
$$
where $\theta_i = \sum_{j=k+1}^{p}\lambda_j^i$ (sums over the *discarded* eigenvalues) and $h_0 = 1-\frac{2\theta_1\theta_3}{3\theta_2^2}$, $c_\alpha$ is the standard normal critical value. (In an interview, knowing this threshold formula's *name* and that it comes from the discarded eigenvalues is more valuable than memorizing every symbol — the key conceptual point is: **Q's threshold depends only on the variance you threw away when you kept just $k$ components.**)

## 8.4 The Key Conceptual Split (most important takeaway of this chapter)

| Statistic | Answers | Analogous to |
|---|---|---|
| Hotelling's T² | "Is this point extreme along directions the data normally varies in?" | Mahalanobis distance (Ch.6), restricted to top-k components |
| Q-statistic (SPE) | "Does this point violate the learned correlation structure entirely — does it not even live near the fitted subspace?" | A genuinely new signal, not present in Ch.6–7 |

A point can be **large-T² small-Q** (extreme but "normal-shaped," e.g. simply a very large version of a typical pattern) or **small-T² large-Q** (unremarkable along known axes, but violates the underlying structure — e.g., in sensor data, a reading that's individually plausible per-sensor but breaks the physical relationship between sensors). Production anomaly detection systems (particularly in manufacturing/process monitoring, where this technique originates) typically monitor **both simultaneously** — this dual-statistic approach is a strong thing to mention unprompted in an interview.

## 8.5 Worked Numerical (Simplified 2D → 1D Reduction)

Data (2 features, already centered for simplicity), with covariance eigen-decomposition giving:
- $\lambda_1 = 8$ (PC1, dominant direction), $\lambda_2 = 0.5$ (PC2, minor direction)
- Keep $k=1$ component (PC1 only)

Suppose a test point has PCA scores $t_1 = 4$ (score on PC1) and $t_2 = 1.5$ (score on PC2, which is discarded).

**Hotelling's T² (using retained PC1 only):**
$$
T^2 = \frac{t_1^2}{\lambda_1} = \frac{16}{8} = 2.0
$$
Compare to $\chi^2_{1,0.95} = 3.84$. Since $2.0 < 3.84$, **not flagged by T²** — this point isn't extreme along the dominant direction of normal variation.

**Q-statistic (using discarded PC2 only, in this simplified 1-component-discarded case):**
$$
Q = t_2^2 = 1.5^2 = 2.25
$$
If the threshold $Q_\alpha \approx 1.0$ (illustrative, based on the small discarded eigenvalue $\lambda_2=0.5$ — thresholds scale with discarded variance), then $Q=2.25 > 1.0$ → **flagged by Q-statistic.**

**Interpretation:** this point is unremarkable along the main direction the data normally varies in (T² is fine), but it has "leaked" unusually far into the direction the data normally does *not* vary in (high Q) — exactly the small-T²/large-Q profile described in §8.4: a point that breaks the underlying structure without being a simple large-magnitude outlier.

## 8.6 Diagnosis: When PCA-Based Detection Applies

| Condition | Recommendation |
|---|---|
| High-dimensional data ($p$ in the hundreds+) where full $\Sigma$ (Ch.6–7) is unstable | PCA-based — dimensionality reduction is the whole point |
| Correlated/redundant features (sensor arrays, financial ratios) | Excellent fit — PCA explicitly exploits correlation structure |
| Need to distinguish "extreme-but-normal-shaped" vs "structurally broken" anomalies | Use T² and Q together (§8.4) |
| Data has strong nonlinear structure (not well captured by linear subspaces) | Limited — consider autoencoders (Ch.14), the nonlinear generalization of this exact idea |
| Need interpretability (which original features drove the anomaly) | Q-statistic's residual vector can be decomposed back to original features (contribution plots), useful for root-cause diagnosis in production monitoring |

## 8.7 Production Considerations
- This T²/Q framework is the backbone of **Statistical Process Control (SPC)** in manufacturing and process monitoring — sensor arrays on a production line are routinely monitored this way in real time, often visualized as a "T²-Q plot" per unit produced.
- Choosing $k$ (number of retained components) is itself a modeling decision with a direct tradeoff: too few components pushes real, legitimate variation into the Q-statistic (false positives on Q), too many components leaves too little "discarded" variance for Q to be a meaningful, sensitive signal.
- Like Mahalanobis/MCD (Ch.6–7), $\mu$ and the PCA subspace itself must be fit on a trusted/clean reference sample when possible — the same circularity risk applies (an outlier-contaminated training set will distort the discovered principal components).
- The subspace and thresholds drift over time as the underlying process changes — periodic retraining is standard in production SPC systems.

## 8.8 Interview Traps
- Only mentioning reconstruction error (Q) and forgetting Hotelling's T² entirely — many candidates only know "PCA outlier detection = high reconstruction error," missing that this only catches half the picture (structurally-broken points), and misses extreme-but-normal-shaped points entirely.
- Not recognizing that T² with all components retained ($k=p$) is literally identical to Mahalanobis distance — this connection is exactly the kind of cross-chapter link that shows real understanding.
- Assuming a single high/low PCA reconstruction error tells you *which* original feature caused the anomaly — you need the residual contribution vector, not just the scalar Q value, to localize the cause.
- Forgetting PCA (and hence T²/Q) fundamentally assumes **linear** relationships among features — nonlinear anomaly structure will not be well captured (motivates autoencoders, Ch.14).

## 8.9 L5-Differentiating Talking Points
- Explicitly deriving that T² collapses to Mahalanobis distance when $k=p$ — again reinforcing the running theme that later chapters aren't new ideas but refinements/restrictions of earlier ones.
- Proactively describing the T²-vs-Q 2×2 conceptual grid (§8.4) unprompted — this single framing question ("extreme along known variation, or violating the structure entirely?") is one of the most senior-sounding things you can say in an outlier-detection interview.
- Correctly positioning PCA-based detection as the classical, interpretable "linear" bridge between full-covariance multivariate methods (Ch.6–7) and modern nonlinear methods (autoencoders, Ch.14) — shows awareness of the full method landscape rather than treating each technique in isolation.

## 8.10 Comprehension Check
1. Show why Hotelling's T² becomes exactly equal to Mahalanobis distance when all $p$ components are retained.
2. Describe a concrete scenario (in your own domain of choice) where a point could have low T² but high Q, and explain why both statistics are needed to catch it.
3. Why does keeping too few principal components risk inflating false positives specifically on the Q-statistic, not on T²?
4. Why is the Q-statistic threshold formula built from the *discarded* eigenvalues rather than the retained ones?

---
*Next: Chapter 9 — One-Class SVM & SVDD.*
