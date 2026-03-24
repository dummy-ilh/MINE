# Master Q&A — Cross-Topic Interview Questions

> The questions that connect concepts across methods. These are the ones that separate candidates who memorised methods from those who understand them.

---

## How to Use This File

These questions deliberately span multiple topics. A good answer draws on concepts from 2–4 different files in this folder. The topics each question covers are listed so you know what to revise.

Questions are grouped by the type of reasoning they test:

1. [Method Comparison Questions](#1-method-comparison-questions)
2. [Contradiction and Disagreement Questions](#2-contradiction-and-disagreement-questions)
3. [Failure Mode Questions](#3-failure-mode-questions)
4. [Production and Deployment Questions](#4-production-and-deployment-questions)
5. [Causal Reasoning Questions](#5-causal-reasoning-questions)
6. [Fairness and Regulation Questions](#6-fairness-and-regulation-questions)
7. [Design Questions — "How Would You..."](#7-design-questions--how-would-you)
8. [Trap Questions — Common Wrong Answers](#8-trap-questions--common-wrong-answers)

---

## 1. Method Comparison Questions

---

**Q1. LIME says feature A is the most important for this prediction. SHAP says feature B is. Both are applied to the same model and same sample. Why might they disagree, and which should you trust?**

*Topics: LIME, SHAP, bias-variance*

They answer slightly different questions. LIME fits a local linear model using random perturbations and returns coefficients. SHAP computes the average marginal contribution across all feature orderings with exact arithmetic (TreeSHAP) or a principled approximation (KernelSHAP).

Reasons for disagreement: (1) LIME has high variance — if run again it might agree with SHAP. Run LIME 10 times and check consistency. (2) LIME's linear approximation may be poor here — if R² (local fidelity) is low, LIME is fitting noise. (3) If A and B are correlated, LIME's perturbations create unrealistic (A=on, B=on) combinations, inflating or deflating A's coefficient. (4) LIME and SHAP use different expectations for "absent features" — LIME turns them off by random replacement; SHAP uses marginal averaging over background data.

Trust SHAP if: the model is a tree (TreeSHAP is exact), features are not heavily correlated, and you've verified the efficiency check. Trust LIME less unless: it's a text or image model where LIME's perturbations are more natural.

---

**Q2. Permutation importance says feature C is most important globally. SHAP's beeswarm plot shows feature C has a wide spread of both positive and negative values. What's the story?**

*Topics: permutation importance, SHAP plots, heterogeneity*

These results are consistent and together tell a richer story than either alone.

Permutation importance measures how much overall model performance drops when C is shuffled — a large drop means C matters globally. But it's unsigned — it doesn't say whether high or low C values are risky.

The beeswarm with wide spread in both positive and negative SHAP values means: C has a strong effect on predictions, but the direction depends on the sample. For some customers, high C increases predictions (positive SHAP); for others, high C decreases predictions (negative SHAP). This pattern indicates a non-monotone relationship or a strong interaction with another feature.

Next step: look at the SHAP dependence plot for C, coloured by another feature. If two distinct bands appear (one where high C is positive, one where it's negative), there's an interaction between C and that other feature. The permutation importance correctly captured the magnitude; the beeswarm revealed the heterogeneity.

---

**Q3. You have a Random Forest. Compare: (a) Gini importance, (b) permutation importance on train, (c) permutation importance on test, (d) TreeSHAP. Rank them by reliability for feature selection.**

*Topics: MDI, permutation, SHAP, train vs test, bias-variance*

Reliability ranking, best to worst: **TreeSHAP on test > permutation on test > permutation on train > Gini importance.**

TreeSHAP on test: exact (no sampling variance), uses held-out data (no overfit bias), accounts for all feature combinations. Best for understanding what the model relies on for generalisation.

Permutation on test: approximate (sampling variance from n_repeats), uses held-out data (no overfit bias), model-agnostic. Slightly worse than TreeSHAP for trees, but competitive.

Permutation on train: uses training data → overfit bias. Noise features that were memorised appear important. Never use for feature selection.

Gini importance: always uses training data (overfit bias), biased toward high-cardinality features, arbitrary credit between correlated features. Use only as a fast sanity check, never for selection decisions.

---

**Q4. How does PDP relate to a SHAP dependence plot? What can each show that the other can't?**

*Topics: PDP, SHAP dependence plot, ICE*

Both show how a feature's values relate to model predictions — but from different perspectives.

PDP shows: the average prediction across the dataset when feature j is fixed at value v. It answers "what does the model predict on average at this feature value?" One curve, no individual variation shown.

SHAP dependence plot shows: for each sample, the SHAP value for feature j plotted against that sample's actual value of j. It answers "how much did feature j contribute to this prediction?" and shows the per-sample distribution.

PDP can show: absolute prediction levels (y-axis is in prediction units). SHAP dependence shows deviations from baseline (y-axis is SHAP value, the contribution).

SHAP dependence can show what PDP can't: (1) heterogeneity — the spread of dots at each x value shows whether all samples behave similarly or differently. (2) Interactions — colouring by a second feature reveals whether j's effect depends on that feature. (3) No extrapolation issue — each dot is a real sample, not an artificially constructed one.

PDP can show what SHAP dependence can't: absolute predictions, which is sometimes more intuitive for non-technical audiences.

---

**Q5. A colleague wants to use ALE for all analyses going forward and never use PDP again. Is this the right policy?**

*Topics: PDP, ALE, bias-variance*

Too strong a policy, but understandable. ALE is strictly better than PDP when features are correlated, which makes it the safer default. However:

PDP is still appropriate when: features are genuinely uncorrelated (|r| < 0.3 for all pairs). In this case, PDP is simpler to interpret (y-axis shows absolute predictions) and slightly faster to compute.

ALE has its own limitations: the y-axis shows centred deviations, not absolute predictions — which is less intuitive for non-technical stakeholders. The choice of B (bins) introduces a bias-variance trade-off that PDP doesn't have. And ALE confidence bands widen at the tails of the distribution where few samples exist.

Better policy: check correlations first. If any feature pair has |r| > 0.5, use ALE for those features. For uncorrelated features, PDP is fine. Always look at ICE alongside either method.

---

## 2. Contradiction and Disagreement Questions

---

**Q6. You train a Random Forest and an XGBoost on the same dataset, both achieving AUC=0.88. The RF says feature X (a zip code) has importance 0.38. The XGBoost says importance 0.12. How do you reconcile this?**

*Topics: Rashomon set, bias-variance, correlation*

This is the Rashomon effect — two equally accurate models attribute different importances to the same feature. Neither is wrong; they are each correct for their respective models.

The high discrepancy for zip code specifically suggests: zip code is correlated with other features (income, education — which often have geographic patterns). The RF and XGBoost found different paths through the correlated feature space: the RF relies heavily on zip code as a proxy; the XGBoost distributes that information across multiple correlated features.

Practical response: (1) check pairwise correlations involving zip_code. (2) Compute grouped permutation importance treating zip_code and its correlated features as a group — that group importance will be more consistent across models. (3) Report a range for zip_code importance: "between 0.12 and 0.38 depending on model." (4) For features where both models agree (low std), the importance is robust.

Don't report just one model's importance as "the" answer.

---

**Q7. You run permutation importance twice with different random seeds. Feature D has importance 0.12 in run 1 and 0.08 in run 2. Is this a problem?**

*Topics: permutation importance, bias-variance, n_repeats*

Whether this is a problem depends on scale. The gap is 0.04 — a 33% relative difference. Whether this matters depends on: (a) what you're trying to do with the importance, and (b) the standard deviation across n_repeats within each run.

If n_repeats=1 in each run, this is expected — one permutation is a single random event with high variance. Solution: always use n_repeats ≥ 10, ideally 20–30.

If n_repeats=20 in each run, then 0.12 and 0.08 are means of 20 shuffles each, and the disagreement is more concerning. Check the standard deviation across repeats: if std ≈ 0.02 for both runs, then the 95% CI for each overlaps (roughly 0.08–0.16 vs 0.04–0.12 with std=0.02), so the values are statistically consistent. But if the gap (0.04) is larger than the CI width, there is a real instability — possibly from using a different test set sample.

Are you using the exact same test set? If the test set is randomly split and differs between runs, the difference reflects test set variance, not permutation variance.

---

**Q8. SHAP shows feature E has a large positive global mean |SHAP|, but the SHAP dependence plot for E is nearly flat. How can this be?**

*Topics: SHAP plots, global vs local*

The mean |SHAP| is the mean of the absolute values. A large mean |SHAP| with a flat dependence plot (SHAP value doesn't change with feature E's value) appears contradictory but has a specific cause: the SHAP values for E are large in magnitude but their sign or magnitude depends on OTHER features, not on E's value itself.

This is the signature of a strong interaction effect. The dependence plot shows SHAP(E) vs E's value. If the spread of SHAP values is large but doesn't correlate with E's value (flat curve), it means: SHAP(E) is determined by the combination of E with another feature, not by E alone.

Solution: colour the dependence plot by each other feature in turn. When you colour by the right interaction feature F, two bands will emerge — SHAP(E) is positive when F is high and negative when F is low (or vice versa). The global mean |SHAP| correctly captures that E matters, but only in combination with F.

---

**Q9. For the same prediction, a counterfactual says "reduce debt by $10k." A SHAP waterfall plot shows debt_ratio SHAP = +0.30. These feel like they're saying the same thing but differently. Are they actually consistent?**

*Topics: SHAP, counterfactuals, local vs global*

They are related but not the same, and a careful answer distinguishes them.

SHAP waterfall: "Compared to the average prediction, this customer's debt_ratio = 0.82 contributed +0.30 to pushing their risk above average." This is an attribution — how much debt explains the current prediction.

Counterfactual: "If debt_ratio were 0.55 instead of 0.82, the prediction would flip from denied to approved." This is a contrast — what change leads to a different outcome.

They are consistent in direction (debt is bad, reducing it helps) but are different quantities. A SHAP value of +0.30 doesn't directly tell you how much you need to reduce debt to flip the prediction — the model is nonlinear, and the threshold may require more or less change than proportional reasoning would suggest. The counterfactual is a more precise operational answer: here is the exact change needed. SHAP is more informative about attribution: here is how much blame debt deserves for the current outcome.

---

## 3. Failure Mode Questions

---

**Q10. You have a dataset where num_rooms and house_size are correlated at r=0.93. Your permutation importance shows both have importance ~0.02 (very low), while age has importance 0.15. A business expert insists house size should be the most important predictor. Who is right?**

*Topics: correlated features, permutation importance, substitute problem*

The business expert is likely right, and the importance estimates are wrong. This is the substitute problem.

When num_rooms is permuted, house_size (correlation 0.93) substitutes perfectly — the model barely loses performance. Same in reverse. Both appear unimportant individually, even though together they're the dominant predictor.

To verify: run grouped permutation importance, permuting num_rooms and house_size together. The group importance will likely be much larger (say, 0.35) — correctly showing they collectively dominate.

The age importance of 0.15 looks large only because it has no strong substitute. Age is genuinely moderately important, but it's not more important than house size — it's just measurable without correlation confounding.

Action: report group importance. Tell the business expert the model uses house size and rooms as a group (importance 0.35), and age separately (0.15). The individual num_rooms/house_size split within the group is ambiguous.

---

**Q11. A model has train accuracy = 0.97 and test accuracy = 0.74. What effect does this have on every XAI method you apply?**

*Topics: bias-variance, train vs test, overfitting, all methods*

The large gap (0.23) indicates severe overfitting. Effects on each method:

**Permutation importance (train set):** Noise features that were memorised will appear important. Genuinely useless features may have importance 0.10–0.20 on training data. Don't use train-set importance for this model.

**Permutation importance (test set):** Correctly identifies only genuinely informative features. Noise features correctly appear near zero. This is the right approach.

**MDI / Gini:** Always uses training data → severely inflated importance for memorised features. Unreliable for this model.

**TreeSHAP (train set):** SHAP values will be large for noise features — correctly describing what the overfit model does, but not what generalises. Don't use for interpretation.

**TreeSHAP (test set):** Correctly shows near-zero SHAP for noise features. Use this.

**PDP and ALE:** If computed on training data, curves will show spurious patterns that don't generalise. Use held-out data.

**LIME:** Will have high variance generally (small test set for this overfit model), and may show noise features with non-trivial coefficients.

**Global surrogate:** Will have artificially high fidelity on training data (the overfit model is "predictable" on training data), but the rules learned will be overfit rules.

**The meta-answer:** An overfit model is the worst case for all XAI methods. The right response is to fix the model first (regularise, more data), then apply XAI.

---

**Q12. You apply LIME to a credit scoring model and the local R² (fidelity) is 0.41. What does this mean and what do you do?**

*Topics: LIME, faithfulness, bias*

R²=0.41 means LIME's local linear model explains only 41% of the black-box model's variance in the neighbourhood of this sample. The explanation is largely wrong — it's fitting noise, not the model's true local behaviour.

Causes: (1) The model has a strong discontinuity near this sample (decision boundary nearby). (2) The model has interaction effects that a linear model structurally cannot capture. (3) The kernel width σ is too large, so LIME is approximating a heterogeneous region.

What to do: (1) Try decreasing σ (smaller neighbourhood, more locally linear). (2) Try increasing n_samples (more stable regression). (3) If R² remains low: the model is genuinely non-linear near this point and LIME cannot faithfully explain it. (4) Use SHAP instead — TreeSHAP doesn't rely on a linear approximation and will give an exact attribution.

For a model that returns low R² in many explanations: LIME is not the right tool. Use TreeSHAP (tree model) or KernelSHAP (any model).

---

**Q13. You compute a global surrogate decision tree with R²=0.65 and report its rules to the business team. What can go wrong?**

*Topics: surrogate models, faithfulness, global vs local*

R²=0.65 means 35% of the black box's behaviour is not captured. The rules extracted from the surrogate may be accurate for 65% of decisions but systematically wrong for the remaining 35%.

The specific risks: (1) The business team may act on rules that don't actually apply to the cases where the model diverges from the surrogate (the complex boundary cases — often the most important ones). (2) They have no way of knowing for any specific decision whether the surrogate's rule applies or whether it's in the 35% blind spot. (3) The surrogate captures average behaviour — it may poorly represent minority subgroups or tail cases where the model behaves differently.

What you should have done: reported the fidelity R²=0.65 prominently alongside every rule. Noted that rules are an approximation of the model. For individual decisions, used local methods (SHAP, counterfactuals) instead of the global surrogate. If R²<0.75, strongly reconsidered whether surrogate rules should be presented as explanations at all.

---

## 4. Production and Deployment Questions

---

**Q14. How do you set up a monitoring system that alerts you when a model's feature importance structure has changed significantly since deployment?**

*Topics: permutation importance, bias-variance, production*

The core idea: periodically re-compute feature importance on recent production data and compare to a baseline computed at deployment time.

Implementation:

(1) At deployment: compute and store baseline permutation importances on a held-out validation set. Record mean and std per feature.

(2) Periodically (weekly or daily): collect a sample of recent production predictions (with outcomes if available, or just inputs). Compute permutation importances on this recent slice.

(3) Detect drift: for each feature, check whether recent importance is outside the baseline CI. Use a statistical test: z-score = |recent_imp − baseline_mean| / baseline_std. Flag if z > 3.0.

(4) Also monitor: ranking changes (rank correlation between baseline and current importance vector), the proportion of predictions explained by each feature changing significantly.

(5) Alert triggers: a feature that was previously unimportant becomes important (possible data leakage or distribution shift). A previously important feature becomes unimportant (feature source broke, values now constant).

The comparison isn't straightforward without outcomes. If you don't have ground truth, compute importance with respect to model predictions (not true labels) — this detects input distribution shift, not performance degradation.

---

**Q15. A data scientist says: "I'll use SHAP values to build a feature selection system — drop any feature whose mean |SHAP| is below 0.01." What are the problems with this approach?**

*Topics: SHAP, correlated features, bias-variance, feature selection*

Several problems:

(1) Correlated features: a feature with mean |SHAP| = 0.005 may be highly correlated with a feature with mean |SHAP| = 0.08. The 0.005 feature looks unimportant — but dropping the 0.08 feature first might reveal it's actually critical. Individual SHAP values for correlated features are unreliable for selection decisions.

(2) Threshold is arbitrary: 0.01 is not theoretically motivated. It depends on the output scale of the model (log-odds vs probability vs raw regression output). A threshold that works for one model may be completely wrong for another.

(3) No confidence intervals: mean |SHAP| = 0.008 with std=0.010 means the CI spans [−0.012, 0.028] — the feature might have true importance 0.028 or true importance 0. A feature with mean |SHAP| = 0.012 and std=0.001 is clearly above zero; a feature with mean=0.012 and std=0.015 is not. The threshold should be applied to the lower bound of the CI, not the mean.

(4) Dropping features changes the model: after dropping features, you must retrain the model. The new model's SHAP values will be different — the process needs to be iterative, not a single-pass filter.

(5) Better approach: use drop-column importance or ablation studies to evaluate the performance impact of removing features, not just their SHAP values.

---

## 5. Causal Reasoning Questions

---

**Q16. Your model says age has high feature importance. A policy maker concludes that age causes income differences and proposes age-based programs. What's wrong with this reasoning?**

*Topics: causality, importance vs causality, Rashomon*

Feature importance is a measure of statistical association and model reliance, not causation. High importance for age means: the model uses age to predict income, and age's value correlates with income in the training data.

This is a statement about the model and the observed data — not about what causes income. Possible explanations for age's high importance that don't involve age causing income:

- Age correlates with experience, which causes income
- Age correlates with cohort effects (educational opportunities, labour market conditions when each cohort entered)
- Age correlates with occupation type (older workers in stable high-paying industries)
- Age is a proxy for many other correlated variables that the model wasn't given

To claim causality, you need: controlled experiments, instrumental variables, or causal graphs with do-calculus. A feature importance number from a supervised ML model is not sufficient. The Rashomon problem further weakens the claim — another equally accurate model might not use age at all, attributing the same predictive power to geography or education.

---

**Q17. A counterfactual explanation says: "If your income were $65k (currently $38k), you'd be approved for the loan." Is this a causal claim?**

*Topics: counterfactuals, causality*

No — not in the causal sense. This counterfactual describes model behaviour: "The model would approve you if it saw income=$65k." It says nothing about what would happen in the real world if your income increased.

In the real world: if your income genuinely increased to $65k, your spending patterns might change, your debt-to-income ratio might change, your credit utilisation might change — the model would see a different feature vector than just income=65k with everything else held constant.

The counterfactual is a ceteris paribus statement about the model's input-output function, not a real-world intervention. It answers "if I could magically change only this number in the model's input, what would happen?" not "if I worked harder and earned more, would I be approved?"

For truly actionable counterfactuals, you need: features that are actionable (income is somewhat actionable; age is not), feature interaction constraints (if reducing debt also improves credit score, the counterfactual should account for that), and domain knowledge about which feature changes are realistic.

---

## 6. Fairness and Regulation Questions

---

**Q18. A credit model shows gender has importance 0.02 in a feature importance analysis. Can you conclude the model is not using gender to discriminate?**

*Topics: Rashomon, correlated features, fairness, global vs local*

No — you cannot conclude this from importance 0.02 alone.

(1) Rashomon problem: this model shows 0.02, but an equally accurate model might show 0.08. The low importance is a property of this specific model, not the data.

(2) Correlated features: gender may be correlated with zip_code, occupation, income, or other features. If those features carry gender's predictive signal, gender's direct importance will be low — but the model is effectively using gender indirectly through its proxies. This is called proxy discrimination and is not captured by looking at gender's importance alone.

(3) Global vs local: even if mean |SHAP| for gender is 0.02 globally, there may be specific subgroups where gender SHAP values are systematically large (e.g., gender matters a lot for young applicants but not older ones). The global average hides this.

(4) SHAP interaction values: gender may have near-zero main effect but large interaction values with other features. Check the SHAP interaction matrix.

For proper fairness analysis: compute outcome disparities (approval rates by gender). Compute SHAP values per subgroup. Check for proxy features correlated with gender. Use fairness-specific metrics (demographic parity, equalised odds), not just feature importance.

---

**Q19. Under GDPR, a customer is entitled to an explanation of why they were denied credit. You provide a LIME explanation. What are the risks?**

*Topics: LIME instability, faithfulness, counterfactuals, regulation*

LIME has several properties that make it legally risky:

(1) Instability: running LIME twice produces different explanations. If the customer asks for a re-explanation, they may receive a different answer — inconsistency that undermines the credibility of the explanation and potentially violates the requirement for a consistent, deterministic process.

(2) No self-verification: LIME's coefficients don't sum to the prediction minus baseline. You cannot verify the explanation is correct. If the customer or a regulator audits it, there's no mathematical proof of correctness.

(3) Low fidelity risk: if local R² is 0.5, the explanation is largely fabricated noise. A regulator asking "how do you know this is what drove the decision?" would have no satisfying answer.

(4) Not actionable by default: "debt_ratio contributed +0.32" doesn't tell the customer what to do.

Better alternatives: TreeSHAP waterfall (exact, verifiable, with efficiency check). Counterfactuals (actionable: "if X, you'd be approved"). Anchors (rule-based: "your debt and income pattern placed you in this category"). These are defensible, stable, and explainable.

---

## 7. Design Questions — "How Would You..."

---

**Q20. You're building a feature selection pipeline for a high-stakes medical diagnosis model. Walk me through your approach.**

*Topics: correlated features, permutation importance, bias-variance, train vs test*

Step 1 — Correlation analysis: compute pairwise correlations and VIF for all features. Group features with |r| > 0.7. Correlated features cannot be reliably evaluated individually.

Step 2 — Model selection: train several model types (RF, GBM, Logistic) with equal performance. Check whether feature importance rankings are consistent across model types (Rashomon check). Features that are important across all models are robust candidates to keep.

Step 3 — Group importance: use grouped permutation importance on a held-out test set. This gives the true combined importance of each correlated group, fixing the substitute problem.

Step 4 — Within-group selection: for each high-importance group, use drop-column importance or ablation to find the single best representative feature. Prefer clinically interpretable features over proxy features.

Step 5 — Confidence intervals: compute importance ± CI for each feature (or group). Only drop features whose CI upper bound is below a threshold — don't drop features where uncertainty is high.

Step 6 — Retrain and validate: retrain with the selected features. Verify performance doesn't drop significantly (< 2% AUC). If it does, re-examine the features flagged for removal.

Step 7 — Document: report the correlation structure, VIF values, group importances, and confidence intervals. For high-stakes medical use, every feature inclusion decision should be auditable.

---

**Q21. A business stakeholder wants a simple dashboard showing "why the model made each decision." How do you design it?**

*Topics: global vs local, SHAP plots, counterfactuals, surrogate*

The dashboard needs two layers: model-level (global) and decision-level (local).

**Global layer (model overview):**
- Feature importance bar chart (mean |SHAP|) — sorted by importance
- ALE/PDP curves for the top 3–5 features — shows the shape of each relationship
- These answer: "What does the model generally care about?"

**Local layer (per-decision explanation):**
- SHAP waterfall plot for the specific decision — shows exact feature contributions
- Top 3 positive and top 3 negative features with plain-English labels
- Counterfactual: "The minimum change that would have resulted in a different outcome"
- These answer: "Why this specific decision?"

**UI considerations:**
- Non-technical users: show only the top features (K=3–5), in natural language
- Technical users: full SHAP waterfall with all features
- For any denied application: always show a counterfactual — it's actionable and legally defensible
- Flag when the model's confidence is low (prediction near the decision boundary) — explanations for uncertain predictions should be clearly marked as uncertain

**What NOT to show:**
- Raw SHAP values without labels (meaningless to most users)
- MDI/Gini importance (biased — use SHAP instead)
- LIME explanations without stability information

---

## 8. Trap Questions — Common Wrong Answers

---

**Q22. "SHAP values sum to the model's prediction." True or false?**

*Topics: SHAP axioms*

False — or at best incomplete. SHAP values sum to the difference between the model's prediction and the baseline (average prediction):

`Σ φⱼ = f(x) − E[f(X)]`

Not `Σ φⱼ = f(x)`.

If someone says "SHAP values sum to the prediction" they're forgetting the baseline. The baseline φ₀ = E[f(X)] is also part of the decomposition: `f(x) = φ₀ + Σ φⱼ`. The full sum including the baseline equals the prediction. But the SHAP values alone sum to the deviation from baseline.

This matters: if your model has a high baseline (e.g., 70% average churn rate), SHAP values explain the deviation from 70%, not from 0%.

---

**Q23. "A high SHAP value for age means age causes this outcome." True or false?**

*Topics: causality*

False. A high SHAP value for age means the model used age to push this prediction above the baseline. It reflects the model's statistical association between age and the outcome in the training data. This is a statement about the model, not about the world.

Age's high SHAP could reflect: age genuinely influences the outcome; age is a proxy for experience, tenure, or cohort effects; the model overfitted a spurious correlation between age and the outcome in the training sample; age is correlated with another causal feature that wasn't included.

Causality requires controlled experiments or causal inference methods, not feature importance values.

---

**Q24. "More features in my LIME explanation (higher K) means a better explanation." True or false?**

*Topics: LIME bias-variance*

False. Increasing K (the number of features in the LIME explanation) increases the complexity of the linear model. This reduces bias (more flexible approximation) but increases variance (more coefficients to estimate from the same N perturbed samples). The right K is the one that minimises total error.

In practice, K=5–10 is usually appropriate. Very small K (K=2) misses important features. Very large K (K=30+) produces unstable coefficients because each coefficient is estimated from insufficient data.

The right test: as you increase K, does the local R² improve significantly? If R² barely changes going from K=5 to K=10, the extra features aren't adding fidelity — just adding noise to the explanation.

---

**Q25. "If two models have the same test accuracy, their feature importance rankings should be the same." True or false?**

*Topics: Rashomon set*

False. This is the Rashomon problem. Two models can achieve identical test accuracy while using completely different features or combinations of features. The feature importance is a property of the specific model, not of the data or the generative process.

This is especially true for datasets with correlated features: one model might rely heavily on feature A (using it as a proxy for the correlated B), while another relies on B. Both achieve the same accuracy but have opposite importance rankings for A and B.

The correct interpretation: if importances agree across multiple equally-good models, that's a robust signal. If they disagree, the importance is model-dependent and should be reported as a range.

---

**Q26. "Permutation importance with n_repeats=100 is very reliable." True or false?**

*Topics: permutation importance, bias-variance*

Partially true, partially false. High n_repeats reduces variance — with 100 repeats, the importance estimate is very stable. But variance is only one source of error.

Bias is unaffected by n_repeats: (1) If you're using the training set, overfit bias persists regardless of repeats. (2) If features are correlated, the substitute problem underestimates correlated features regardless of repeats. (3) These biases don't shrink with more repeats — they're systematic.

So: n_repeats=100 gives a very precisely measured biased estimate if the underlying setup is flawed. High precision on a biased estimator is not reliability — it's precisely wrong.

True reliability requires: correct data split (test set), handling of correlated features (grouped permutation), and appropriate n_repeats to reduce variance. All three matter.

---

## Final Checklist — What Every Senior Candidate Should Know Cold

```
Theory
  ☐ SHAP efficiency axiom: Σφⱼ = f(x) − E[f(X)]  (not f(x))
  ☐ SHAP is unique: the ONLY method satisfying all 4 axioms
  ☐ Rashomon set: feature importance is model-specific, not data-specific
  ☐ Importance ≠ causality — ever

Train vs Test
  ☐ Always use test set for importance (training = overfit bias)
  ☐ Train−test importance gap = overfitting diagnostic per feature
  ☐ Underfitting: all importances collapse to zero

Correlated Features
  ☐ Permutation: substitute problem → both features underestimated
  ☐ MDI: credit absorption → arbitrary split by random seed
  ☐ SHAP marginal: unrealistic coalitions → wrong within-group split
  ☐ Fix: grouped permutation importance, conditional/interventional SHAP

Method Selection
  ☐ Tree model → TreeSHAP (never LIME or KernelSHAP if avoidable)
  ☐ Correlated features → ALE not PDP; grouped permutation not individual
  ☐ LIME for text/image OK; LIME for tabular → high variance, verify stability
  ☐ Surrogate: always report fidelity R²; R² < 0.75 → don't trust the rules

Local vs Global
  ☐ Never explain an individual decision with global importance
  ☐ Individual explanation → SHAP waterfall, counterfactual, anchor
  ☐ Model audit → surrogate, PDP/ALE, SHAP beeswarm

Fairness
  ☐ Low importance for protected attribute ≠ no discrimination
  ☐ Proxy features can carry discrimination even with zero direct importance
  ☐ SHAP interaction values needed for complete fairness audit
  ☐ Rashomon: apparent fairness is model-dependent
```
