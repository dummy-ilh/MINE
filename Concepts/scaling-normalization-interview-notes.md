# Scaling / Normalization — Interview Notes (End-to-End, With Examples)

## 1. Why Scaling Matters At All

Features often live on wildly different numeric scales (e.g., "age" in [0-100] vs. "income" in [0-500000]). Many algorithms are **sensitive to the absolute magnitude** of feature values, not just their relative relationships — for those algorithms, a feature with a naturally larger numeric range can dominate the objective function or distance calculation purely because of its units, not because it's actually more important.

```
Without scaling — Euclidean distance between two points in (Age, Income) space:

  Point A: (Age=25, Income=50000)
  Point B: (Age=45, Income=52000)

  distance = √[(45-25)² + (52000-50000)²] = √[400 + 4,000,000] ≈ 2000.1

  → Income's difference (2000) COMPLETELY dominates the distance,
    even though a 20-year age gap is arguably just as meaningful
    as a $2000 income gap. Age is rendered almost irrelevant.
```

**Which algorithms actually need scaling — know this table cold:**

| Sensitive to scale (scaling required/strongly recommended) | Insensitive to scale (scaling optional/unnecessary) |
|---|---|
| k-NN, k-Means (distance-based) | Decision Trees, Random Forest, Gradient Boosting (split on thresholds per-feature, unaffected by monotonic rescaling) |
| SVMs (especially RBF/kernel-based) | Naive Bayes (works on probabilities, not raw distances) |
| PCA (variance-based — high-variance features dominate components) | |
| Linear/Logistic Regression **with regularization** (L1/L2 penalize coefficient magnitude, which depends on feature scale) | Linear/Logistic Regression **without regularization** (coefficients simply absorb the scale, unregularized loss unaffected) — though scaling still helps convergence speed |
| Neural Networks (gradient descent converges much faster and more stably on similarly-scaled inputs) | |
| Any gradient-descent-optimized model, for optimization stability even if not strictly "required" | |

**Key interview point:** tree-based models are scale-invariant because they split on `feature > threshold` — multiplying a feature by any positive constant doesn't change which rows fall on which side of any threshold, so the tree structure is literally identical either way. This is a very clean, precise fact worth stating exactly.

---

## 2. Standardization (Z-score Normalization)

$$x' = \frac{x - \mu}{\sigma}$$

where $\mu$ is the feature's mean and $\sigma$ its standard deviation (both computed on the **training set only**).

```
Original "Income":  [30000, 50000, 70000, 90000, 110000]
μ = 70000,  σ ≈ 28284.3

Standardized:
  30000 → (30000-70000)/28284.3 = -1.414
  50000 → (50000-70000)/28284.3 = -0.707
  70000 → (70000-70000)/28284.3 =  0.000
  90000 → (90000-70000)/28284.3 =  0.707
  110000→ (110000-70000)/28284.3=  1.414
```

- Result has **mean 0, standard deviation 1** — but is **not** bounded to a fixed range (an extreme outlier can still produce a very large standardized value).
- Does **not** assume any particular distribution shape, but works best (is most interpretable) when the feature is roughly Gaussian/symmetric — heavily skewed data may need a transform (log, Box-Cox) before standardizing.
- **Preferred default for:** PCA, linear/logistic regression with regularization, SVMs, neural networks, any algorithm assuming/benefiting from zero-centered, unit-variance input.

---

## 3. Min-Max Normalization (Rescaling)

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Rescales to a fixed range, typically $[0, 1]$ (or sometimes $[-1, 1]$ with a small formula variant).

```
Original "Age":  [22, 35, 41, 58, 67]
min = 22, max = 67

Min-Max scaled:
  22 → (22-22)/(67-22) = 0.000
  35 → (35-22)/(67-22) = 0.289
  41 → (41-22)/(67-22) = 0.422
  58 → (58-22)/(67-22) = 0.800
  67 → (67-22)/(67-22) = 1.000
```

- Guarantees output in exactly $[0,1]$ — useful when a bounded range is specifically required (e.g., pixel intensity normalization for image models, or algorithms/layers that expect bounded input).
- **Critical weakness: extremely sensitive to outliers**, since min and max are determined by the single most extreme values in the data:

```
Same "Age" data but with one outlier added: [22, 35, 41, 58, 67, 150]
New min=22, max=150

Min-Max scaled:
  22  → 0.000
  35  → 0.102   ← previously 0.289! The entire scale got compressed
  41  → 0.148   ← previously 0.422!
  58  → 0.281   ← previously 0.800!
  67  → 0.352   ← previously 1.000!
  150 → 1.000
```

One outlier (150) completely compressed every other value toward the bottom of the range — this is the single most important weakness to be able to state precisely in an interview, and it's why min-max scaling is generally avoided when outliers are present (prefer standardization or robust scaling instead, see §4).

---

## 4. Robust Scaling

$$x' = \frac{x - \text{median}}{\text{IQR}} \qquad \text{where IQR} = Q_3 - Q_1 \text{ (75th percentile − 25th percentile)}$$

```
Original "Income" with an outlier: [30000, 50000, 70000, 90000, 110000, 2000000]
Median = 80000 (midpoint of sorted middle values, using standard interpolation)
Q1 = 55000, Q3 = 95000  →  IQR = 40000

Robust scaled:
  30000    → (30000-80000)/40000  = -1.250
  50000    → (50000-80000)/40000  = -0.750
  70000    → (70000-80000)/40000  = -0.250
  90000    → (90000-80000)/40000  =  0.250
  110000   → (110000-80000)/40000 =  0.750
  2000000  → (2000000-80000)/40000= 48.000   ← the outlier itself still looks extreme
                                                (correctly — it IS extreme), but it
                                                no longer distorts everyone ELSE's
                                                scaled values, unlike min-max above
```

- Uses **median and IQR instead of mean and standard deviation** — both are robust statistics (resistant to extreme values), so a single huge outlier doesn't drag the reference points around the way it drags the mean/std or the min/max.
- **Best choice when your data has known outliers you don't want to (or can't) remove**, but still need meaningful, non-distorted relative scaling among the bulk of "normal" values.
- Note the outlier itself still gets a large scaled value (48.0 above) — robust scaling doesn't hide or clip outliers, it just prevents them from *distorting the scale applied to everything else*, which is a distinct and important difference from clipping/winsorizing.

---

## 5. Normalization to Unit Norm (Vector Normalization / L2 Normalization)

Rescales each **row** (not column/feature) so the vector of feature values has unit norm (length 1).

$$x'_i = \frac{x_i}{\|\mathbf{x}\|_2} = \frac{x_i}{\sqrt{\sum_j x_j^2}}$$

```
Row vector: [3, 4]
L2 norm = √(3² + 4²) = √25 = 5

Normalized: [3/5, 4/5] = [0.6, 0.8]
  → new vector has length exactly 1: √(0.6² + 0.8²) = √(0.36+0.64) = √1 = 1
```

- **This is fundamentally different from standardization/min-max/robust scaling**, which all operate **column-wise** (rescaling one feature at a time, across all rows). Unit-norm normalization operates **row-wise** (rescaling one sample's entire feature vector, so its direction is preserved but its magnitude becomes 1).
- Commonly used in **text/TF-IDF representations** (documents of different lengths naturally produce vectors of different magnitudes — normalizing removes raw length as a confound so that cosine similarity/comparisons focus on the *direction*/composition of the vector, i.e., relative word importance, not document length) and in some neural network embedding comparisons (cosine similarity is mathematically equivalent to a dot product between L2-normalized vectors).
- A very common and important interview distinction: "normalization" and "standardization" are used loosely and inconsistently across textbooks/libraries — always clarify **row-wise (per-sample, unit norm) vs. column-wise (per-feature, e.g., z-score or min-max)** when the terms come up, since both get called "normalization" in different contexts.

---

## 6. Log / Power Transforms (for Skewed Data)

Not strictly "scaling" in the min-max/z-score sense, but frequently applied **before** scaling to fix a skewed distribution's shape.

```
Original "Income" (right-skewed, most values clustered low, long tail of high earners):

Histogram:  ████████████████
            ████████████
            ████████
            ████
            ██
            █                    ← long thin tail out to very high incomes
            ─────────────────────────────────►

After log transform (log(income)):

Histogram:      ████████
              ████████████
            ████████████████
              ████████████
                ████████         ← much more symmetric, closer to Gaussian
            ─────────────────────────────────►
```

- **Log transform** ($\log(x)$, or $\log(x+1)$ to handle zeros) compresses large values much more than small ones, pulling in long right tails and making the distribution more symmetric/Gaussian-like — very common for income, population, word counts, and other naturally multiplicative/exponential-feeling quantities.
- **Box-Cox transform** is a generalized, parameterized version ($\frac{x^\lambda - 1}{\lambda}$ for $\lambda \neq 0$, $\log(x)$ for $\lambda=0$) that finds the optimal $\lambda$ to best normalize the specific distribution's shape — requires strictly positive values.
- **Yeo-Johnson transform** is a variant of Box-Cox that also works with zero and negative values, extending the same idea to a broader range of real-world features.
- **Why this matters for scaling specifically:** standardization (z-score) works best/is most interpretable when the underlying distribution is roughly symmetric — applying a log/Box-Cox transform first, then standardizing, often gives much better-behaved input for linear models and neural networks than standardizing the raw skewed feature directly.

---

## 7. Summary Comparison Table

| Method | Formula | Output range | Outlier sensitivity | Preserves distribution shape? | Best for |
|---|---|---|---|---|---|
| Standardization (Z-score) | $(x-\mu)/\sigma$ | Unbounded, mean 0, std 1 | Moderate (mean/std both affected by outliers, but less catastrophically than min-max) | Yes, just recentered/rescaled | PCA, regularized linear models, SVMs, neural nets — general default |
| Min-Max | $(x-min)/(max-min)$ | Fixed $[0,1]$ | **High** — single outlier compresses everything else | Yes | Bounded-input requirements (image pixels, some NN layers), NO known outliers |
| Robust Scaling | $(x-\text{median})/\text{IQR}$ | Unbounded | **Low** — median/IQR resistant to outliers | Yes | Data with known/unremovable outliers |
| Unit Norm (L2, row-wise) | $x_i / \|\mathbf{x}\|_2$ | Vector length = 1 | Depends on the vector's composition | No — changes relative row magnitude, not per-feature shape | Text/TF-IDF vectors, cosine-similarity-based comparisons |
| Log / Box-Cox / Yeo-Johnson | $\log(x)$ or parameterized | Unbounded (shape-dependent) | Reduces the *impact* of large values by compressing them | **No — deliberately changes shape** to reduce skew | Right-skewed data (income, counts) — usually applied before z-score/min-max |

---

## 8. Common Pitfalls (interviewers love probing these)

1. **Fitting the scaler (mean/std, min/max, median/IQR) on the full dataset before splitting into train/val/test.** Leaks validation/test statistics into the scaling applied to training data — the exact same leakage class as fitting a target encoder or imputer on the full dataset (see Cross-Validation and Missing Data notes). Always fit on training data only, then apply (transform, don't re-fit) to validation/test.
2. **Scaling one-hot encoded or binary columns.** Rarely necessary or meaningful — 0/1 columns don't have a "scale problem" in the same sense continuous features do; scaling them can sometimes actively hurt interpretability without helping optimization.
3. **Using min-max normalization on data with outliers without realizing the compression effect.** As shown in §3, a single extreme value can crush the effective resolution of every other value into a tiny sub-range — silently destroying most of the useful signal among "normal" values.
4. **Forgetting that tree-based models don't need scaling at all**, and wasting effort scaling features for a Random Forest or GBM pipeline where it provides zero benefit (harmless, but unnecessary preprocessing complexity).
5. **Confusing row-wise (unit-norm) and column-wise (z-score/min-max) normalization**, since both get informally called "normalization" — always clarify which axis is being rescaled, since they solve entirely different problems (per-sample vector length vs. per-feature scale).
6. **Standardizing heavily skewed data without a prior log/power transform**, then being surprised the standardized feature still has a long tail — standardization only recenters and rescales, it does **not** change distribution shape; skew transforms are a separate, often-necessary earlier step.
7. **Re-scaling after imputing missing values with a placeholder like -999 without an indicator column** (see Missing Data notes §6) — a sentinel value gets swept into the same mean/std or min/max calculation as real data, badly distorting the scaler's fitted parameters unless it's handled first.
8. **Applying different scalers to train and test (e.g., accidentally re-fitting MinMaxScaler on the test set) instead of reusing the training-fitted transform.** A very common implementation bug — always call `.fit()` once (on train) and `.transform()` (not `.fit_transform()`) on validation/test.

---

## 9. FAANG-Level Interview Q&A

**Q1: You're training a k-NN classifier on features "age" (range 0-100) and "annual income" (range 0-500,000). Without scaling, what happens, and how would you fix it?**
Euclidean distance calculations will be almost entirely dominated by income, since its absolute numeric range is thousands of times larger than age's — a genuinely large age difference (say, 40 years) contributes a tiny fraction of the total distance compared to even a modest income difference (say, $5,000), effectively making age irrelevant to the model's neighbor-finding regardless of how predictive it actually is. The fix is standardization (or min-max, if no outliers) applied to both features before computing distances, so each feature contributes to the distance calculation based on its actual relative variation, not its arbitrary raw units.

**Q2: Why don't decision trees or Random Forest require feature scaling, while logistic regression with L2 regularization does?**
Trees split on `feature > threshold` — this decision is unaffected by any monotonic rescaling of the feature (multiplying by a positive constant, or shifting by an additive constant, doesn't change which rows fall on which side of any threshold), so the tree's structure and predictions are mathematically identical whether or not you scale. Logistic regression with L2 regularization penalizes the *squared magnitude* of coefficients — if one feature has a naturally huge numeric range, its optimal coefficient will naturally be tiny to compensate, and the regularization penalty then unfairly shrinks that feature's effective contribution relative to a similarly-important feature that happens to have a smaller natural range; scaling ensures the regularization penalty is applied fairly across features regardless of their original units.

**Q3: A colleague min-max scales a feature to [0,1], not realizing there's a single data-entry error making one value 1000x larger than everything else. What goes wrong, and how would you detect and fix it?**
The erroneous extreme value becomes the new max, and every other (correct) value gets compressed into a tiny sliver near 0 — effectively destroying almost all the discriminative resolution among the legitimate data points, since they now all cluster extremely close together after scaling. I'd detect this by checking summary statistics (min/max, or a quick histogram/boxplot) before scaling — a max value orders of magnitude beyond the 99th percentile is a strong signal of either a genuine extreme outlier or a data entry error — and fix it either by correcting/removing the erroneous point, or switching to robust scaling (median/IQR-based) which wouldn't be distorted by the single bad value in the first place.

**Q4: You standardize a right-skewed income feature (mean 0, std 1) and feed it into a linear regression. Performance is mediocre. What would you check, and why?**
Standardization only recenters and rescales the distribution — it does **not** change its shape, so a right-skewed feature remains right-skewed after standardizing, just with a mean of 0 and std of 1. If the true relationship with the target is closer to linear in log-income than raw income (a very common real-world pattern for income, price, and count-type features), the linear model will fit poorly regardless of scaling, since scaling doesn't fix a shape/functional-form mismatch. I'd apply a log or Box-Cox transform first to address the skew/shape issue, then standardize the transformed feature — these are two separate problems (shape vs. scale) that need two separate fixes.

**Q5: When would you deliberately choose NOT to scale features, even for an algorithm that's technically scale-sensitive?**
If the model is tree-based (Random Forest, XGBoost, LightGBM) — scaling provides zero benefit since trees are scale-invariant, so skipping it is simply saving unnecessary preprocessing effort, not a real tradeoff. Alternately, unregularized linear regression is also mathematically invariant to feature scaling in terms of final predictions (the coefficients simply absorb the scale, and the loss surface's global optimum doesn't change) — though I'd note that gradient-descent-based optimization of that same unregularized model can still converge much faster/more stably with scaled inputs even if the final mathematical optimum is identical, so scaling is often still practically worth it for optimization reasons even when not "required" for correctness.

**Q6: Explain why unit-norm (L2) normalization is fundamentally different from standardization, even though both are sometimes called "normalization."**
Standardization operates **column-wise** — for a given feature (column), it rescales that feature's values across all samples (rows) to have mean 0 and std 1, changing each feature's scale independently of the others. Unit-norm normalization operates **row-wise** — for a given sample (row), it rescales that entire feature vector so its overall length (L2 norm) equals 1, changing the relationship between features *within* a single sample while leaving the direction of that sample's vector unchanged. They solve different problems: standardization addresses "features on different scales dominating each other," while unit-norm addresses "samples of different overall magnitude (e.g., document length in TF-IDF) dominating comparisons between samples." Conflating the two, given they share the word "normalization," is a common source of confusion.

**Q7: Your team fits a StandardScaler on the combined train+test dataset before splitting for cross-validation, reasoning "we're just computing summary statistics, not looking at labels, so it can't be leakage." Do you agree?**
No — even though it doesn't use labels directly, this still constitutes information leakage, because the scaling parameters (mean, std) used to transform the training data now depend partly on data that should be entirely unseen during training. In a small dataset this can meaningfully shift the fitted parameters and thus the scaled training values, giving the model subtly different (indirectly test-set-influenced) inputs than it would see in genuine production — and it breaks the strict train/test separation that cross-validation is designed to guarantee, making performance estimates optimistically biased, even if the effect size is often small in large datasets. The correct practice is to fit the scaler on the training fold only, then transform (not re-fit) validation/test data with those training-derived parameters.

**Q8 (clever): Can scaling ever change a model's predictions for an algorithm that's supposedly "scale-invariant" like unregularized linear regression, in practice (not just in theory)?**
Yes, in practice, even though the theoretical global optimum is unaffected — numerical optimization algorithms (gradient descent, or even some implementations of ordinary least squares involving matrix conditioning) can behave differently or converge to slightly different points due to floating-point precision and conditioning issues when features have vastly different scales, especially with regularization-free but iteratively-optimized solvers, or when the feature matrix is poorly conditioned (near-collinear features with very different scales can amplify numerical instability in matrix inversion). This is a good "in theory vs. in practice" distinction to draw — the closed-form mathematical answer and the actual floating-point computed answer aren't always identical.

---

## 10. One-Line Interview Closers

- *"Scaling is required exactly when the algorithm's math cares about absolute magnitude — trees split on thresholds and don't care, distance and gradient-based methods absolutely do."*
- *"Min-max and robust scaling solve the same problem differently — min-max trusts the extremes, robust scaling deliberately ignores them, and that's the whole reason to pick one over the other."*
- *"Standardizing a skewed feature doesn't fix the skew — it just recenters it; shape and scale are two different problems that need two different fixes."*
