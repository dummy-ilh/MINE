# Chapter 8 — Global vs. Local, and Aggregating Local Explanations

Chapter 1 introduced global vs. local as one of three taxonomy axes. This chapter goes deeper on that specific axis — how local explanations (SHAP, LIME) get turned into a global ranking, what's genuinely lost in that translation, and how to check whether an importance ranking (of any kind) is trustworthy or just noise.

## 8.1 Native global methods vs. natively local methods

Recall from Chapter 1's table: MDI, linear coefficients, and (in its native form) permutation importance are **natively global** — they were never computed "per prediction" in the first place; they answer a whole-dataset question directly. SHAP and LIME are **natively local** — every SHAP value or LIME coefficient is computed with respect to one specific example, and any "global" statement about them requires an explicit aggregation step on top.

This distinction matters because natively-global methods have no local counterpart at all — you can't ask MDI "how important was this feature for this one specific prediction," since MDI was never defined per-example to begin with. Natively-local methods, conversely, require an extra step to answer the global question, and that extra step is where information gets lost.

## 8.2 Aggregating SHAP values into a global ranking

**The standard aggregation:** compute the SHAP value for every feature, for every example in your dataset, then take the **mean absolute SHAP value** for each feature across all examples:

Global importance of feature j = (1/N) · Σᵢ |SHAP value of feature j for example i|

**Why absolute value, specifically, and not just the mean:** a feature can push predictions up for some examples and down for others (e.g., `age` might increase the predicted outcome for some people and decrease it for others, depending on other context) — if you averaged the *signed* SHAP values directly, these opposite-direction contributions could largely cancel out, making a genuinely influential feature look unimportant on average, purely because its effect isn't consistently one-directional. Taking the absolute value first ensures you're measuring "how much does this feature move the prediction, in either direction," rather than "does this feature consistently move the prediction in one particular direction."

**What's lost in this aggregation, precisely:** the same heterogeneity problem you saw with PDP vs. ICE in Chapter 7 — the mean absolute SHAP value tells you a feature matters "a lot, on average," but doesn't tell you *whether it matters consistently across all examples* or *matters hugely for one subgroup and not at all for another*. Two features can have identical mean absolute SHAP values while one affects every example moderately and the other affects a small subgroup enormously and everyone else not at all — the aggregated number can't distinguish these genuinely different situations. A SHAP summary plot (Chapter 6's earlier material) partially addresses this by showing the *distribution* of SHAP values across examples for each feature, not just the aggregated mean — which is exactly why the summary plot, not just a bar chart of mean absolute SHAP values, is the standard way to report SHAP results.

## 8.3 Aggregating LIME explanations into a global picture

**Why this is harder for LIME than for SHAP:** LIME's coefficients are only meaningful *locally*, within the specific neighborhood around one example (Chapter 6, §6.2–6.3) — there's no guarantee that a coefficient computed near example A and a coefficient computed near example B, even for the same feature, are measuring a comparable quantity, since each was fit against a different local neighborhood with its own perturbation sample and its own locally-linear approximation. Simply averaging LIME coefficients across many examples, the way you'd average absolute SHAP values, is on much shakier ground than the SHAP aggregation in §8.2, precisely because SHAP's axioms (Chapter 5, §5.3) give the underlying per-example values a consistent, well-defined meaning that LIME's locally-fit coefficients don't share.

**A common workaround: SP-LIME (submodular pick).** Rather than averaging every example's LIME explanation, SP-LIME selects a small, deliberately diverse subset of examples to explain individually (chosen to collectively cover as much of the different kinds of model behavior as possible, using a submodular optimization criterion to pick a representative set rather than a random or exhaustive one), and presents those individual explanations side by side as a stand-in for a "global picture" — an explicit acknowledgment that LIME doesn't aggregate cleanly, so the workaround is to pick a good sample of local stories rather than to force a single global number out of it.

## 8.4 Stability: how much does an importance ranking change across resamples or retraining?

**The question this section answers:** if you retrained your model on a slightly different sample of the same underlying data (a different train/test split, a different random seed, a bootstrap resample), would you get roughly the same importance ranking, or would it shuffle around substantially? An importance ranking that's highly sensitive to exactly which data happened to be sampled is a ranking you should trust much less than one that's stable across resamples.

**How to measure this in practice:**
1. Generate multiple bootstrap resamples of your training data (or retrain with several different random seeds, if your model has stochastic elements like a random forest's feature/sample subsampling).
2. Retrain the model (or recompute the importance measure on the same trained model, for methods like permutation importance where the shuffle itself is the random element) on each resample.
3. Compute the importance ranking each time, and measure how much the rankings agree across resamples — common approaches include tracking whether the same top-k features consistently appear across resamples, or computing a rank correlation (e.g., Spearman's) between pairs of resample-derived rankings.

**What instability tells you, and what to do about it:** an unstable ranking is often — though not always — a symptom of one of the correlated-feature issues covered throughout this topic (Chapter 4's masking, Chapter 3's VIF-flagged multicollinearity): when two features are highly redundant, which one "wins" the importance contest can flip essentially at random across different resamples, purely because the split of credit between them is inherently unstable, not because either feature's importance is genuinely changing. Seeing instability in your rankings is itself a signal to go check for exactly this kind of redundancy (§8.2's grouped-SHAP-style thinking, or Chapter 4's grouped permutation importance) rather than simply reporting whichever single ranking happened to come out of one particular training run.

## 8.5 Quick self-check before Chapter 9

- Can you explain precisely why mean absolute (rather than mean signed) SHAP value is the standard global aggregation, using a concrete example where signed averaging would mislead?
- Can you explain why averaging LIME coefficients across examples is on shakier theoretical ground than averaging SHAP values, tracing it back to Chapter 5's axioms?
- Given an unstable importance ranking across bootstrap resamples, what's the first thing you'd check, and why?

---

**Next: Chapter 9 — Pitfalls and Gotchas**, pulling together every recurring failure mode from this topic — correlated-feature masking and extrapolation, high-cardinality bias, leakage risk, and the predictive-vs-causal distinction — into one unified, interview-focused chapter.
