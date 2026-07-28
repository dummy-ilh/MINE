# Chapter 2 — Filter Methods

Chapter 1 previewed filter methods as "score each feature independently, no model required." This chapter makes that concrete: the actual statistical tests used, what each one assumes, and a worked example showing exactly where the simplest filter (correlation) misses a relationship that a more sophisticated one (mutual information) catches.

## 2.1 The general filter recipe

Every filter method follows the same three-step pattern:

1. Pick a scoring function that measures "how related is this one feature to the target?"
2. Compute that score for every feature independently (feature A's score doesn't depend on feature B at all).
3. Keep the top-k features by score, or keep every feature whose score clears some threshold.

The entire family differs only in step 1 — which scoring function you pick — and that choice depends on whether your feature and target are continuous or categorical.

## 2.2 Correlation coefficient (continuous feature, continuous target)

**What it measures:** the Pearson correlation coefficient r measures the strength of a **linear** relationship between a feature and the target, ranging from −1 (perfect negative linear relationship) to +1 (perfect positive linear relationship), with 0 meaning no linear relationship.

**Formula, in words:** r is the covariance between the feature and target, divided by the product of their individual standard deviations — this normalization is what keeps r bounded between −1 and +1 regardless of the features' original units or scale.

**The critical limitation, stated up front:** correlation only detects *linear* relationships. A feature can be a perfect, deterministic predictor of the target and still have a correlation of exactly zero, if the relationship is non-linear in the right way. Section 2.5 works through exactly this case numerically, because it's one of the most common "gotcha" interview questions on this topic.

## 2.3 Chi-squared test and ANOVA F-test (categorical combinations)

Correlation assumes both feature and target are continuous. Two other tests cover the categorical cases:

- **Chi-squared test** (categorical feature, categorical target): builds a contingency table (counts of each feature-category × target-category combination) and tests whether the observed counts differ from what you'd expect if the feature and target were completely independent. A large chi-squared statistic means the feature and target categories are strongly associated — the feature carries information about which target class an example belongs to.

- **ANOVA F-test** (categorical feature, continuous target): tests whether the *mean* of the continuous target differs significantly across the different categories of the feature. If a feature's categories all produce roughly the same average target value, the F-statistic is low (the feature isn't discriminating anything); if the categories produce very different average target values, the F-statistic is high.

**The pattern to remember:** picking the right filter test is mostly a matter of correctly identifying the (feature type, target type) combination — continuous/continuous → correlation, categorical/categorical → chi-squared, categorical/continuous (or continuous feature/categorical target, handled symmetrically) → ANOVA F-test.

## 2.4 Mutual information — catching non-linear relationships

**What it measures:** mutual information (MI) comes from information theory and measures **how much knowing the feature's value reduces your uncertainty about the target** — regardless of whether that relationship is linear, quadratic, U-shaped, or anything else. MI is zero if and only if the feature and target are *truly statistically independent* — no relationship of any shape exists.

**Why this matters:** correlation can only see straight-line relationships; mutual information can see any relationship at all. The cost is that MI is somewhat more expensive to estimate accurately (it typically requires estimating probability distributions from your data, e.g., via binning or a k-nearest-neighbors-based estimator, rather than a simple closed-form formula like correlation's).

## 2.5 Worked example: where correlation fails and mutual information succeeds

Suppose you have a feature x that takes values evenly spread from −10 to +10, and the target is generated as:

**y = x²**

This is about as clean a deterministic, fully-predictive relationship as you can construct — if you know x, you know y exactly, every time.

**Now compute the correlation.** For every positive value of x (say, x=5, giving y=25), there's a matching negative value (x=−5, giving the *same* y=25). The relationship is symmetric around x=0: as x increases from very negative to very positive, y first decreases (from x=−10 to x=0) and then increases (from x=0 to x=10) — there's no consistent *linear* direction to the relationship at all. Working through the correlation formula on data generated this way, **r comes out to (very close to) exactly 0** — despite y being a perfectly deterministic, 100%-predictable function of x.

**A filter method using correlation alone would discard this feature entirely** — it would look like pure noise, scoring at or near zero, right alongside genuinely useless random features.

**Now compute mutual information instead.** Because knowing x tells you y exactly (there's zero remaining uncertainty about y once you know x), the mutual information between x and y is **at its maximum possible value** for this pair of variables — correctly identifying x as a perfectly informative feature.

**The takeaway, stated as the interview-ready line:** *"Correlation-based filters can completely miss non-linear relationships, even deterministic ones — mutual information doesn't have this blind spot, which is why it's the safer default filter when you don't already know the relationship is roughly linear."*

## 2.6 Variance threshold — the simplest possible filter

**What it does:** doesn't even look at the target at all — simply drops any feature whose variance across the dataset falls below some threshold (most commonly used to drop near-constant features, e.g., a feature that's the same value for 99% of rows).

**Why it's useful despite being so simple:** a near-constant feature carries almost no information *by construction* — regardless of its relationship to the target, there's barely any variation for a model to use. This is a cheap first pass to run before any of the more sophisticated tests above, since it removes obviously useless features without needing to touch the target variable at all.

## 2.7 Pros, cons, and when to reach for filter methods

**Pros:**
- Extremely fast — no model training required, each feature scored independently in roughly linear time.
- Model-agnostic — the same feature ranking can inform a decision tree, a linear model, or a neural net.
- A good first pass on very high-dimensional data (thousands of features), where wrapper methods (Chapter 3) would be computationally infeasible to even attempt.

**Cons:**
- **Ignores feature interactions** — two features that are each individually weak but jointly powerful (e.g., "the product of feature A and feature B predicts the target well, but neither alone does") will both score low and get discarded, even though together they're valuable.
- **Ignores redundancy between features** — two highly correlated features will both get high individual scores and both get kept, even though one of them is nearly duplicate information the model doesn't need twice.
- Because of both points above, filter methods are typically used as a **first-pass, cheap filter** to cut down from a very large feature set to a more manageable one — followed by a wrapper or embedded method (Chapters 3–4) that can account for interactions and redundancy on the reduced set.

---

**Next: Chapter 3 — Wrapper Methods**, where feature selection stops being independent-per-feature and starts being an actual search over subsets — forward selection, backward elimination, and Recursive Feature Elimination, along with why cross-validation is essential to keep the search itself from overfitting.
