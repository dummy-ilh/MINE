# Chapter 9 — Practical Synthesis

This is the last chapter, and like Chapter 10 of your Fairness & Responsible AI prep, it's a rehearsal rather than new material. We'll take a single scenario end-to-end through diagnosis, selection, importance validation, and deliberately catch a leakage bug along the way — then close with a decision framework and practice interview questions with strong answer structures.

## 9.1 End-to-end worked case: 200 features down to the ones that matter

**Setup:** you're given a dataset with 200 raw features predicting loan default, and asked to reduce it to a manageable, trustworthy feature set for a production model.

**Step 1 — Cheap first pass with filter methods (Chapter 2).** With 200 features, a wrapper method's roughly quadratic cost (Chapter 3, §3.5) is still fine computationally, but it's wasteful to spend that budget on features that are obviously useless. Run a variance threshold first to drop near-constant features (say, this removes 30 features that are the same value for 99%+ of rows), then compute mutual information (not plain correlation, since some relationships in the raw features may be non-linear per Chapter 2, §2.5) between each remaining feature and the target, keeping the top 60 by MI score.

**Step 2 — Catch the leakage bug.** Here's the moment worth rehearsing explicitly, since it's the classic trap from Chapter 8, §8.2: suppose your initial instinct is to compute that mutual information score across your *entire* dataset before splitting into train/test. **Stop — this is exactly the leakage pattern from Chapter 8.** The fix: split into train/test (or set up your cross-validation folds) *first*, then compute mutual information using only the training fold's data, selecting the top-60 features based only on what's visible in training. If you're validating with cross-validation, this selection step needs to happen fresh inside each fold, not once on the full training set.

**Step 3 — Embedded refinement (Chapter 4).** With 60 features remaining (a workable number for a wrapper-style refinement, or simply an embedded method directly), fit a Lasso logistic regression with cross-validated λ selection. Suppose this drives 25 of the 60 coefficients to exactly zero, leaving 35 nonzero features. Check for suspiciously unstable or sign-flipping coefficients among the survivors (Chapter 7, §7.4) — suppose two features, `total_credit_lines` and `active_credit_lines`, show unstable coefficients; a quick VIF check confirms VIF > 12 for both, so you combine them into a single derived feature (or drop one) before moving on.

**Step 4 — Validate with SHAP (Chapter 6).** Train your final model (say, a gradient-boosted tree) on the resulting ~34 features, and run TreeSHAP to get both a global summary plot and spot-check a few individual predictions. Suppose the SHAP summary broadly agrees with the Lasso-survived feature set (the same handful of features — `credit_score`, `debt_to_income`, `recent_delinquencies` — dominate both), which is a good consistency check across two independently-derived importance signals (Lasso's embedded selection vs. SHAP's post-hoc explanation) — agreement between different methods is itself useful evidence that you've found genuine signal rather than an artifact of any one method.

**Step 5 — Sanity-check against MDI's known bias (Chapter 5, §5.2).** Suppose your gradient-boosted tree's built-in MDI importance ranks a high-cardinality feature (`employer_industry_code`, with 400 distinct codes) surprisingly high — higher than SHAP or permutation importance suggest. Recognizing MDI's cardinality bias from Chapter 5, you don't trust this ranking at face value — you compute permutation importance for this specific feature and find its true contribution is much smaller, confirming MDI inflated it exactly as expected.

**End state:** a documented, ~34-feature model, with a clear record of which method flagged which features, why two candidates got merged due to multicollinearity, and confirmation (via cross-checking SHAP against MDI) that the final ranking isn't an artifact of any single method's known bias.

## 9.2 A general decision framework

Boiled down to a repeatable sequence for any new feature selection/importance scenario:

1. **Start cheap and broad.** Variance threshold, then a filter method (mutual information as the safer default over correlation, since it catches non-linear relationships too) to cut a very large feature set down to a workable size (Chapter 2).
2. **Check for leakage before computing anything target-dependent.** Is your train/test split (or CV fold structure) already in place before you compute any statistic that uses the target? If not, fix that first (Chapter 8, §8.2) — this should be a reflexive first check, not an afterthought.
3. **Refine with an embedded method if you have a compatible model type in mind** (Lasso/Elastic Net for linear, natural tree-based selection for tree models) — this is usually the most cost-effective refinement step (Chapter 4).
4. **Use a wrapper method only if you specifically need to search across feature subsets more thoroughly than an embedded method allows, and can afford the compute** (Chapter 3) — often not necessary if step 3 already got you to a good, small feature set.
5. **Explain the final model's reliance on features using permutation importance or SHAP, not raw MDI alone** — especially if your feature set mixes cardinalities (Chapters 5–6).
6. **Check correlated/redundant features explicitly (VIF, or a correlation matrix) whenever an importance ranking looks surprising** — a feature ranking low or unstable is often a redundancy signal, not proof of irrelevance (Chapter 8, §8.1).
7. **Remember predictive importance isn't causal importance** — if the downstream decision involves an actual intervention, not just prediction, flag that a causal analysis is needed beyond whatever this topic's methods provide (Chapter 8, §8.4).

## 9.3 Practice interview questions with strong answer structures

**"How would you reduce a 500-feature dataset down to the 20 that matter?"**
Structure your answer in the same order as 9.2: cheap filter pass first (name mutual information specifically, and why over correlation), explicit mention of doing this only within the training fold to avoid leakage (volunteering this before being asked is a strong signal), then an embedded method appropriate to whatever model you'll actually deploy, then a validation pass with permutation importance or SHAP to confirm the final ranking isn't an MDI artifact if a tree model is involved.

**"Why might a feature have high importance in a random forest but a small coefficient in logistic regression?"**
This tests whether you understand that different models capture different kinds of relationships. A strong answer: a linear model's coefficient only captures a feature's *linear, unconditional* relationship with the target — if the feature's relationship with the target is non-linear (Chapter 2, §2.5's y=x² pattern) or only matters in combination with another feature (an interaction effect), a tree-based model can capture that relationship easily (trees naturally model non-linearities and interactions through sequences of splits), while a plain linear model's single coefficient would show that same feature as weak or negligible, since it's only looking for a straight-line relationship.

**"Your feature importance ranking says a feature doesn't matter, but domain experts insist it should. What do you check?"**
Walk through the checklist from Chapter 8 in order: (1) check correlation with other features in the model — is this feature's signal being absorbed by a correlated partner (§8.1)? (2) if it's a tree-based model, check whether you're looking at MDI specifically, and re-check with permutation importance, especially if this feature has notably different cardinality than others (§8.3); (3) check whether the relationship might be non-linear or interaction-dependent in a way a linear-model coefficient would miss (Chapter 7); (4) consider whether the domain experts might be thinking causally ("this factor really does drive outcomes") while your metric is purely predictive/associational (§8.4) — these can genuinely diverge, and reconciling that distinction explicitly is often exactly what the conversation needs.

**"Walk me through why L1 regularization gives you sparse solutions but L2 doesn't."**
Give the geometric picture directly (Chapter 4, §4.2): diamond-shaped constraint region with corners on the axes (L1) vs. circular constraint region with no corners (L2) — an elliptical loss contour is disproportionately likely to first touch the diamond exactly at a corner, which is where a coefficient is exactly zero, whereas it can touch the smooth circle's boundary anywhere, essentially never landing exactly on an axis.

**"What's the difference between feature selection and feature importance — aren't they the same thing?"**
No — feature selection (Chapters 2–4) is about deciding which features to *include* in the model at all, typically done before or during training; feature importance (Chapters 5–7) is about explaining which features a model *already relying on which features*, typically computed after training, for interpretability or debugging purposes. A feature can be excluded by selection but you'd never compute its "importance" (it's not in the model); conversely, a feature can survive selection and still turn out to have low importance in the trained model, which itself is useful information — perhaps it should be reconsidered for removal in a future iteration.

---

**That's all nine chapters.** You now have the full arc: why fewer features can help, not hurt (Ch1) → three families of selection methods, cheapest to most expensive (Ch2–4) → explaining a trained model's reliance on features, tree-based then model-agnostic then linear (Ch5–7) → the recurring pitfalls that tie everything together (Ch8) → and a rehearsed end-to-end case plus practice questions (Ch9). Let me know if you'd like a condensed one-page cheat sheet of formulas and definitions for last-minute review, matching the one I offered for your Fairness chapters.
