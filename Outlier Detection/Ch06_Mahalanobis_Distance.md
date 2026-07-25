# Chapter 6: Mahalanobis Distance (Multivariate Outlier Detection)

## 6.1 Motivation — Why Univariate Methods Fail on Multivariate Data

Every method in Chapters 2–5 scores **one feature at a time**. But consider a person who is 10 years old and 180cm tall. Neither value alone is extreme (many 10-year-olds exist, many 180cm-tall people exist), but the **combination** is essentially impossible. Per-feature Z-scores or IQR fences would miss this entirely — this is exactly the trap flagged repeatedly in Ch.2 §2.7 and Ch.3 §3.7. Mahalanobis distance is the first method in this curriculum that scores a point using **all features jointly**, accounting for how they **covary**.

## 6.2 Euclidean Distance's Blind Spot

Ordinary Euclidean distance from the centroid,
$$
d_{euclid}(x) = \sqrt{(x-\mu)^T(x-\mu)}
$$
treats every feature as equally scaled and uncorrelated. Two problems:
1. **Scale sensitivity**: a feature measured in thousands (income) dominates a feature measured in single digits (years of experience) even if both are equally "informative."
2. **Ignores correlation**: if two features are highly correlated (height and weight), a point that breaks that correlation (very tall, very light) is unusual — but Euclidean distance, which treats axes independently, won't specially penalize it.

## 6.3 Mahalanobis Distance — Formula

$$
D_M(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}
$$

where $\mu$ is the mean vector and $\Sigma$ is the covariance matrix of the data.

**Intuition:** this is Euclidean distance computed in a *transformed coordinate system* where the data has been "whitened" — rescaled so every direction has unit variance and axes are decorrelated. $\Sigma^{-1}$ does two jobs simultaneously: it **rescales** each feature by its own variance (fixing problem 1) and **rotates/corrects** for correlation between features via the off-diagonal covariance terms (fixing problem 2).

**Special case check:** if features are uncorrelated and each has variance $\sigma_i^2$, then $\Sigma$ is diagonal with entries $\sigma_i^2$, and:
$$
D_M(x) = \sqrt{\sum_i \frac{(x_i-\mu_i)^2}{\sigma_i^2}}
$$
This is literally **the sum of squared per-feature Z-scores** — Mahalanobis distance collapses to a multivariate generalization of Ch.2's Z-score exactly when there's no correlation between features. This is the cleanest way to see Mahalanobis as "Z-score, generalized to multiple correlated dimensions."

## 6.4 Distribution & Threshold

Under multivariate normality, the **squared** Mahalanobis distance follows a **chi-square distribution**:
$$
D_M(x)^2 \sim \chi^2_p
$$
where $p$ = number of features (dimensions). This gives a principled threshold: flag $x$ as an outlier if
$$
D_M(x)^2 > \chi^2_{p,\,1-\alpha}
$$
e.g., for $p=2$ and $\alpha=0.05$, $\chi^2_{2,0.95} = 5.99$.

This is the direct multivariate analog of Ch.2's "$|z|>3$ under normality corresponds to a specific tail probability" — same logic, now in $p$ dimensions, using the chi-square distribution instead of the normal.

## 6.5 Worked Numerical

Two features: Height (cm), Weight (kg). Sample data:

| Height | Weight |
|---|---|
| 160 | 55 |
| 165 | 60 |
| 170 | 65 |
| 175 | 70 |
| 180 | 75 |
| 150 | 90 |  ← suspected outlier (short but heavy)

**Step 1 — mean vector** (using first 5 "normal" points for illustration of $\mu, \Sigma$, then scoring the 6th):
$$
\mu = (170, 65)
$$
(mean of `[160,165,170,175,180]` = 170; mean of `[55,60,65,70,75]` = 65)

**Step 2 — covariance matrix** (using the 5 normal points):
$$
\text{Var}(H) = \frac{(160-170)^2+(165-170)^2+(170-170)^2+(175-170)^2+(180-170)^2}{4} = \frac{100+25+0+25+100}{4}=62.5
$$
$$
\text{Var}(W) = \frac{(55-65)^2+(60-65)^2+0+(70-65)^2+(75-65)^2}{4} = \frac{100+25+0+25+100}{4}=62.5
$$
$$
\text{Cov}(H,W) = \frac{(-10)(-10)+(-5)(-5)+0+(5)(5)+(10)(10)}{4} = \frac{100+25+0+25+100}{4}=62.5
$$

So (since height and weight increase together perfectly linearly in this toy data):
$$
\Sigma = \begin{pmatrix} 62.5 & 62.5 \\ 62.5 & 62.5 \end{pmatrix}
$$

This particular $\Sigma$ is actually singular (perfectly collinear toy data) — for teaching purposes let's nudge it to be invertible by assuming slight noise gives:
$$
\Sigma = \begin{pmatrix} 62.5 & 60 \\ 60 & 62.5 \end{pmatrix}
$$
(a small correlation adjustment purely so we can invert it for this walkthrough)

**Step 3 — invert Σ:**
$$
\det(\Sigma) = 62.5\times62.5 - 60\times60 = 3906.25-3600=306.25
$$
$$
\Sigma^{-1} = \frac{1}{306.25}\begin{pmatrix}62.5 & -60\\-60&62.5\end{pmatrix} = \begin{pmatrix}0.2041 & -0.1959\\-0.1959&0.2041\end{pmatrix}
$$

**Step 4 — score the suspect point** $x=(150, 90)$:
$$
x-\mu = (150-170,\ 90-65) = (-20,\ 25)
$$

$$
\Sigma^{-1}(x-\mu) = \begin{pmatrix}0.2041 & -0.1959\\-0.1959&0.2041\end{pmatrix}\begin{pmatrix}-20\\25\end{pmatrix} = \begin{pmatrix}0.2041(-20)+(-0.1959)(25)\\-0.1959(-20)+0.2041(25)\end{pmatrix}
$$
$$
= \begin{pmatrix}-4.082-4.898\\3.918+5.103\end{pmatrix}=\begin{pmatrix}-8.98\\9.02\end{pmatrix}
$$

$$
D_M^2 = (x-\mu)^T \Sigma^{-1}(x-\mu) = (-20)(-8.98)+(25)(9.02) = 179.6+225.5 = 405.1
$$
$$
D_M = \sqrt{405.1} \approx 20.13
$$

**Step 5 — compare to threshold:** $\chi^2_{2,0.95}=5.99$. Since $D_M^2=405.1 \gg 5.99$, this point is an **extremely strong multivariate outlier** — as expected, since "short and heavy" breaks the strong height-weight correlation in this data.

**Contrast with Euclidean distance:** $d_{euclid} = \sqrt{(-20)^2+25^2} = \sqrt{400+625}=\sqrt{1025}\approx32.0$ — Euclidean distance also flags it as far away, but gives no calibrated statistical threshold and, critically, wouldn't distinguish this point from an equally-far point that moves *along* the natural height-weight correlation (e.g., someone both taller and heavier than average by the same raw distance) — Mahalanobis specifically penalizes deviation *from the correlation structure*, not just raw distance from the centroid.

## 6.6 Diagnosis: When Mahalanobis Applies (and When It Breaks)

| Condition | Mahalanobis appropriate? |
|---|---|
| Multivariate roughly-Gaussian data | Yes — textbook use case |
| Features are correlated | Yes — this is exactly what it's built for, unlike per-feature Z-score |
| High dimensionality ($p$ close to or exceeding $n$) | No — $\Sigma$ becomes singular/poorly estimated, can't invert reliably |
| Outliers already present in the data used to estimate $\mu, \Sigma$ | Problematic — $\mu$ and $\Sigma$ themselves are **not robust**, so outliers can distort the very covariance structure used to detect them (same masking issue as Ch.2, now in matrix form) — motivates **robust covariance estimation** (Minimum Covariance Determinant), covered next chapter |
| Non-Gaussian / multimodal data | No — chi-square threshold assumes multivariate normality; clusters or skew break the calibration |

## 6.7 Production Considerations
- Computing and inverting $\Sigma$ for high-dimensional feature sets (hundreds+ of features) is computationally expensive ($O(p^3)$ for inversion) and numerically unstable if features are collinear — regularization (adding a small ridge term to $\Sigma$'s diagonal) is standard practice.
- $\mu$ and $\Sigma$ must be refreshed as the underlying data distribution drifts — a covariance matrix estimated on last year's data may not reflect this year's feature relationships (e.g., new product categories changing the price-quantity correlation in an e-commerce fraud model).
- Because $\Sigma$ itself can be contaminated by the very outliers you're trying to detect, production systems often estimate $\mu,\Sigma$ on a *cleaned or trusted reference subset*, not the raw live data, to avoid the circularity problem.

## 6.8 Interview Traps
- Computing Mahalanobis distance but forgetting to check invertibility of $\Sigma$ — a very common gotcha when features are highly collinear (near-singular covariance matrix).
- Not connecting Mahalanobis back to Z-score as the diagonal-covariance special case — this connection is a strong signal of conceptual (not just formulaic) understanding.
- Forgetting that $\mu, \Sigigma$ are themselves non-robust estimates — assuming Mahalanobis is automatically "the robust multivariate method" when in fact its naive form inherits the same non-robustness as the classic mean/SD (Ch.2).
- Applying the chi-square threshold without checking (or at least mentioning) the multivariate normality assumption.

## 6.9 L5-Differentiating Talking Points
- Deriving the diagonal-covariance special case live (§6.3) to show Mahalanobis isn't a new idea but "Z-score expressed correctly for correlated multivariate data" — this is the single best way to demonstrate deep, connected understanding of the curriculum so far.
- Proactively flagging that $\mu,\Sigma$ estimation itself is vulnerable to the outliers you're trying to detect, and naming Minimum Covariance Determinant / robust covariance (Ch.7) as the fix — mirrors the exact "each chapter fixes the last chapter's failure mode" narrative that's been built since Ch.2→Ch.5.
- Correctly scoping the curse of dimensionality limitation (high $p$, unstable $\Sigma^{-1}$) as the reason production systems often reach for PCA-based (Ch.8) or tree/ensemble-based (Ch.12) methods instead once feature counts get large.

## 6.10 Comprehension Check
1. Show why Mahalanobis distance reduces exactly to a sum of squared Z-scores when features are uncorrelated.
2. Why is the *squared* Mahalanobis distance the one that follows a chi-square distribution, rather than the unsquared distance itself?
3. Explain the circularity problem: how can the very covariance matrix used to detect outliers be corrupted by those same outliers?
4. In 2D data with features (Age, Income) where older people tend to earn more (positive correlation), would a young, extremely high-income person be flagged more strongly by Mahalanobis distance than by Euclidean distance from the centroid? Why?

---
*Next: Chapter 7 — Robust Covariance Estimation (Minimum Covariance Determinant) & Elliptic Envelope.*
