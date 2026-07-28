# Chapter 3 — Intrinsic Importance: Linear & Generalized Linear Models

Chapter 2 covered the tree-based side of the intrinsic row from Chapter 1's table. This chapter covers the other model type with a fully "free" built-in importance signal: linear and generalized linear models, where the coefficients themselves are the importance measure — provided you handle the traps in this chapter correctly.

## 3.1 Raw vs. standardized coefficients — the scale-distortion trap, formalized

**The core fact:** in y = w₁x₁ + w₂x₂ + ... + b, coefficient wᵢ represents "how much does y change per one-unit change in xᵢ." That's only comparable across features if "one unit" means something comparable across features — which it almost never does by default (dollars vs. bedrooms vs. years vs. a 0/1 flag are not comparable units).

**The fix, formally.** Standardize each feature before fitting: xᵢ' = (xᵢ − μᵢ) / σᵢ, where μᵢ and σᵢ are that feature's mean and standard deviation across the training set. Fitting on standardized features gives coefficients wᵢ' that represent "how many standard deviations does y change, per one standard-deviation change in xᵢ" — a unit-free quantity, directly comparable across every feature regardless of original scale.

**The relationship between raw and standardized coefficients**, worth knowing explicitly rather than just re-fitting each time: wᵢ' = wᵢ · σᵢ (the standardized coefficient is the raw coefficient scaled up by that feature's own standard deviation) — so you can convert between them without refitting, as long as you know each feature's standard deviation.

**Worked recap (compressed from prior material, now stated as a formula you can reuse):** if `square_footage` has raw coefficient w=100 ($/sqft) and σ=1,000 sqft, its standardized coefficient is 100 × 1,000 = 100,000. If `bedrooms` has raw coefficient w=8,000 ($/bedroom) and σ=1, its standardized coefficient is 8,000 × 1 = 8,000. Comparing 100,000 to 8,000 (standardized) rather than 100 to 8,000 (raw) is what correctly shows `square_footage`'s typical real-world swing matters more.

## 3.2 Confidence intervals as a richer alternative to p-values

You've seen the p-value-vs-practical-importance distinction before; here's the sharper, more actionable version of that idea. Rather than reporting only a p-value (which answers a yes/no-ish "is this distinguishable from zero" question), report a **confidence interval around the standardized coefficient** — e.g., "95% CI: [0.15, 0.42]" tells you both that the effect is likely nonzero (the interval excludes 0) **and** roughly how large it plausibly is (somewhere between a modest and a fairly substantial standardized effect), in one number. A p-value alone collapses both of these into a single significance/non-significance verdict and throws away the magnitude information a CI preserves.

**Interview-ready framing:** *"I'd report the confidence interval on the standardized coefficient rather than just the p-value — the interval tells you both whether the effect is likely real and how big it plausibly is, which is exactly the two things 'importance' needs to convey."*

## 3.3 Variance Inflation Factor (VIF), derived step by step

**The setup question VIF answers:** how much has feature j's coefficient variance been inflated purely because it's correlated with the other features in the model, versus what its variance would be if it were uncorrelated with everything else?

**The derivation:**
1. Take feature j, and run a separate auxiliary regression predicting **feature j itself** from all the *other* features in your model (treating xⱼ as if it were the target, and every other xᵢ as the predictors).
2. Get the R² from this auxiliary regression — this tells you how well the other features, collectively, can reconstruct feature j. If R² is high, feature j is largely redundant with the others; if R² is near 0, feature j provides information the others don't have.
3. Compute: **VIF_j = 1 / (1 − R²_j)**

**Why this specific formula, and what it means geometrically:** the variance of a coefficient estimate in linear regression is inversely related to how much *unique* variation that feature contributes once the other features are accounted for — R²_j directly measures how much of feature j's variation is *not* unique (it's the fraction explainable by the other features). As R²_j → 1 (feature j is almost perfectly redundant with the others), (1 − R²_j) → 0, and VIF_j → ∞ — the coefficient variance blows up, because the model has almost no way to isolate feature j's individual effect from its correlated partners'. As R²_j → 0 (feature j is completely independent of the others), VIF_j → 1 — no inflation at all, the cleanest possible case.

**Practical thresholds and what to do:** VIF > 5 is commonly treated as worth investigating; VIF > 10 is commonly treated as a real problem. Responses include: dropping one of the redundant features, combining correlated features into a single derived feature (e.g., a composite index), or switching to Ridge/L2 regression, which handles correlated features far more gracefully than plain OLS by shrinking correlated coefficients together rather than letting them swing wildly (this connects back to the L1-vs-L2 geometric picture from your feature selection material — L2's smooth circular constraint region doesn't have L1's instability-inducing corners).

## 3.4 Odds ratios — a classification-specific, more interpretable importance lens

**Why logistic regression coefficients need special handling.** In logistic regression, the model fits log-odds as a linear function of the features: log(p / (1−p)) = w₁x₁ + ... + b. A raw coefficient wᵢ therefore represents "change in log-odds per unit change in xᵢ" — a quantity that's mathematically clean but not intuitive for most audiences (nobody thinks in log-odds naturally).

**The odds ratio fix:** exponentiate the coefficient: **OR_i = e^{w_i}**. This converts "change in log-odds" into "multiplicative change in odds" — a much more interpretable statement: *"OR = 1.5 means a one-unit increase in this feature multiplies the odds of the positive outcome by 1.5 (a 50% increase in odds), holding other features fixed."* An OR of exactly 1 means no effect; OR < 1 means the feature decreases the odds of the positive outcome; OR > 1 means it increases them.

**Why this matters for "importance" specifically, not just interpretability in general:** odds ratios give you a natural, audience-friendly way to rank and communicate feature importance for a classification model without requiring your audience to understand log-odds at all — a compliance reviewer or a business stakeholder can directly grasp "this feature roughly doubles the odds" in a way "this feature adds 0.69 to the log-odds" simply doesn't land. Just remember: odds ratios still require standardization first (§3.1) if you want to compare *across* features fairly — an odds ratio computed on a raw, unstandardized feature carries exactly the same scale-distortion trap as a raw linear coefficient.

## 3.5 Quick self-check before Chapter 4

- Can you convert between a raw and standardized coefficient given only the feature's standard deviation, without refitting the model?
- Can you derive, from scratch, why VIF → ∞ as a feature becomes perfectly redundant with the others?
- Given a logistic regression coefficient of 0.4, can you compute the corresponding odds ratio and state what it means in plain language?

---

**Next: Chapter 4 — Permutation Importance in Depth**, covering the full procedure, why it fixes MDI's cardinality bias mechanistically, its own correlated-feature and extrapolation failure modes, and conditional/grouped variants that address them.
