# Chapter 7 — Feature Importance for Linear Models

Chapters 5 and 6 covered tree-based and model-agnostic importance methods. Linear and logistic regression models have their own, more direct route to importance — the coefficients themselves — but that directness comes with a trap that's a very common interview question: **raw coefficient size, by itself, tells you almost nothing about real-world importance.**

## 7.1 Why raw coefficient magnitude is meaningless without standardization

**The trap, stated plainly:** in a linear model y = w₁x₁ + w₂x₂ + ... + b, it's tempting to look at the fitted weights and assume "the feature with the biggest |w| must be the most important." This is wrong whenever features are measured on different scales — which is nearly always.

**Worked example showing exactly how scale distorts this.** Suppose you're predicting house price, and one feature is `square_footage` (typical range: 500–5,000) and another is `number_of_bedrooms` (typical range: 1–6). Suppose the true underlying relationship is that each additional square foot adds $100 to the price, and each additional bedroom adds $8,000 to the price.

- The fitted coefficient on `square_footage` will be **w ≈ 100** (dollars per square foot).
- The fitted coefficient on `number_of_bedrooms` will be **w ≈ 8,000** (dollars per bedroom).

Looking at raw coefficients, `number_of_bedrooms` (8,000) looks 80× more "important" than `square_footage` (100) — but that's purely an artifact of the units each feature happens to be measured in. If you had instead measured square footage in *thousands* of square feet, the coefficient would be 100,000 instead of 100 — same underlying relationship, wildly different-looking coefficient, purely from a unit change. **The raw coefficient's size is contaminated by the feature's scale, and comparing coefficients across features with different scales is comparing apples to oranges.**

## 7.2 Standardized coefficients — the fix

**The fix:** before fitting (or interpreting) the model, standardize every feature to have mean 0 and standard deviation 1 (subtract each feature's mean, divide by its standard deviation). After standardization, each coefficient represents "how many standard deviations does the prediction change, per one standard-deviation change in this feature" — a unit-free, directly comparable quantity across every feature, regardless of the feature's original scale.

**Re-doing the house price example with standardization.** Suppose `square_footage` has a standard deviation of 1,000 sq ft across your dataset, and `number_of_bedrooms` has a standard deviation of 1 bedroom.

- One standard deviation of `square_footage` (1,000 sq ft) corresponds to a $100,000 price change (1,000 × $100/sqft) → standardized coefficient ≈ 100,000 (in standardized units).
- One standard deviation of `number_of_bedrooms` (1 bedroom) corresponds to a $8,000 price change → standardized coefficient ≈ 8,000 (in standardized units).

Now `square_footage`'s standardized coefficient (100,000) correctly comes out much larger than `number_of_bedrooms`'s (8,000) — reflecting that a "typical" swing in square footage actually moves the price a lot more than a "typical" swing in bedroom count, which is the real-world importance question you actually wanted answered, not an artifact of arbitrary units.

**The interview-ready rule:** *"Never compare raw linear model coefficients across features unless every feature is on an identical, meaningful scale — standardize first, or you're comparing artifacts of unit choice, not real importance."*

## 7.3 Statistical significance vs. practical importance

Even after standardizing, there's a second distinction worth being crisp about: **statistical significance (a p-value) is not the same question as practical importance (how much the coefficient actually matters).**

**What a p-value on a coefficient tells you:** roughly, "if this feature truly had zero effect on the target, how likely would we be to see a coefficient at least this large, purely by chance, given our sample size?" A small p-value means the estimated relationship is unlikely to be pure noise — but it says nothing about the *size* of the effect.

**Why this distinction matters practically:** with a large enough dataset, even a genuinely tiny, practically irrelevant effect can produce a very small (highly "significant") p-value — significance testing gets more sensitive as sample size grows, so a large dataset can flag a coefficient as "statistically significant" even when its standardized magnitude is negligible and it would never meaningfully change a real-world decision. Conversely, a feature with a large, practically important standardized coefficient can still show a large (non-"significant") p-value if your sample size is small and the estimate is noisy.

**The interview-ready framing:** *"A small p-value tells you the estimated relationship probably isn't pure noise — it doesn't tell you the relationship is big enough to matter. Always look at the standardized coefficient's size alongside its significance, not one without the other."* A confidence interval around the standardized coefficient is often more informative than the p-value alone, since it directly shows the plausible range of the effect's actual size, not just whether zero is excluded from that range.

## 7.4 Multicollinearity and coefficient instability

**The problem:** when two or more features are highly correlated with each other, the linear model has genuine difficulty figuring out how to "split credit" between them — small changes in the training data (or even just resampling) can cause the coefficients to swing wildly, sometimes even flipping sign, while the model's overall *predictions* stay nearly identical. This makes coefficient-based importance unreliable specifically in the presence of multicollinearity — the same underlying issue that showed up for Lasso in Chapter 4 (§4.3, correlated features arbitrarily splitting credit) and for permutation importance in Chapter 5 (§5.4, correlated features masking each other's true importance) — it's a recurring theme across nearly every importance method in this topic, not a quirk unique to linear models.

**Variance Inflation Factor (VIF) — the standard diagnostic.** VIF quantifies how much a given feature's coefficient variance is "inflated" due to its correlation with the other features. It's computed by: regressing that one feature against all the *other* features (using them to predict it, as if it were itself the target), getting an R² from that regression (how well the other features predict this one), and computing:

**VIF = 1 / (1 − R²)**

**Interpreting this formula intuitively:** if a feature is completely uncorrelated with every other feature, R² from that auxiliary regression is 0, giving VIF = 1/(1−0) = 1 — no inflation at all. If a feature is almost perfectly predictable from the others (say R² = 0.9), VIF = 1/(1−0.9) = 10 — its coefficient's variance is 10× larger than it would be without that redundancy, meaning the coefficient estimate is far less stable/trustworthy. A commonly used rule of thumb treats VIF > 5 or VIF > 10 as a signal of concerning multicollinearity worth addressing — either by dropping one of the redundant features, combining them, or switching to a method (like Ridge/L2 regularization) that's more robust to this instability than plain unregularized linear regression.

**The practical consequence for feature importance specifically:** if you see two features with unstable, sign-flipping, or suspiciously large-magnitude coefficients, check their VIF before concluding anything about their real importance — the instability may be telling you about redundancy between features, not about either feature's true relationship with the target.

---

**Next: Chapter 8 — Pitfalls and Gotchas**, pulling together the recurring failure modes that have appeared throughout this topic — correlated features splitting credit, data leakage through feature selection done before the train/test split, high-cardinality bias, and the difference between predictive importance and causal importance.
