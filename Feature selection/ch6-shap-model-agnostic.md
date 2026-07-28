# Chapter 6 — SHAP and Other Model-Agnostic Importance Methods

Chapter 5's methods (MDI, permutation importance) work for any model in principle, but MDI specifically only makes sense for tree-based models. This chapter covers the two most commonly cited **model-agnostic** explanation methods — SHAP and LIME — which work identically whether the underlying model is a random forest, a neural network, or anything else.

## 6.1 Shapley values — the intuition before any formula

SHAP ("SHapley Additive exPlanations") is built on **Shapley values**, a concept from cooperative game theory that predates machine learning entirely — it was originally designed to answer a fairness question about splitting a payout among collaborators, and only later got repurposed for explaining model predictions.

**The original game-theory question:** imagine a group of players who cooperate to earn some total payout, where different subsets (coalitions) of players earn different amounts depending on who's in the coalition. How do you fairly split the total payout among the individual players, given that the value of adding any one player depends on *who else is already in the coalition*?

**The Shapley value's answer:** a player's fair share is their **average marginal contribution**, averaged over every possible order in which players could have joined the coalition. Concretely: consider every possible ordering in which players could be added one at a time; for each ordering, measure how much the total payout increases at the exact moment this particular player joins; then average that marginal contribution across all possible orderings.

**Why average over all orderings, not just one?** Because a player's apparent contribution depends heavily on who's already present. If player X only adds value once player Y is already in the coalition (they're complementary), then X's contribution looks huge in orderings where Y goes first, and small in orderings where X goes first. Averaging across every ordering is what makes the resulting split fair and not dependent on an arbitrary choice of "who happened to join first."

## 6.2 Mapping this onto feature importance

**The translation:** treat each **feature** as a "player," and the model's prediction (for one specific example) as the "payout" being cooperatively produced. A feature's Shapley value for that one prediction is its average marginal contribution to the prediction, averaged over every possible order in which features could be "added" (i.e., every possible subset of other features that could already be "known" when this feature's information is introduced).

**Concretely, for one specific loan applicant:** start from a baseline prediction (what the model would predict knowing nothing about this specific person — essentially the average prediction across the whole training population). Then ask: across every possible order of revealing this person's `credit_score`, `income`, `age`, etc. to the model, how much does the prediction change, on average, at the exact moment `credit_score` gets revealed? That average shift **is** `credit_score`'s Shapley value for this specific prediction. Sum every feature's Shapley value together with the baseline, and you get back exactly this person's actual predicted score — the Shapley values are guaranteed to add up perfectly to "baseline + total contribution = final prediction," which is where the "Additive" in SHAP's name comes from.

**Why this is more principled than MDI or even permutation importance:** MDI and permutation importance both give you a *single, global* importance score per feature across the whole dataset. Shapley values give you a **per-prediction, per-feature** contribution — you can explain not just "credit_score matters a lot overall" but "for this specific applicant, credit_score pushed their predicted approval probability up by 0.12, while their zip code pushed it down by 0.03" — a much more granular, individually-actionable explanation.

## 6.3 What SHAP actually computes in practice

Computing the *exact* Shapley value requires averaging over every possible ordering of features — for n features, that's n! orderings, which is computationally infeasible for anything beyond a handful of features. **SHAP is a set of efficient approximation algorithms** for estimating Shapley values without the full exponential computation — different SHAP variants (TreeSHAP for tree-based models, KernelSHAP as a fully model-agnostic but slower fallback, DeepSHAP for neural networks) each exploit structure specific to a model type to make the approximation tractable and fast.

**What a SHAP summary plot shows:** typically, one row per feature, with every training example plotted as a dot along that row — the dot's horizontal position shows that example's Shapley value (how much that feature pushed *this specific* prediction up or down), and the dot's color typically encodes the feature's actual value (e.g., red for high credit_score, blue for low). This lets you see, at a glance, both which features matter most overall (rows sorted by average absolute Shapley value) **and** the direction of the relationship (e.g., "high credit_score values, in red, cluster on the positive/right side — high credit_score pushes predictions up").

**What a SHAP force plot shows:** for one single prediction, a visual breakdown of exactly which features pushed the prediction up (in one color) and which pushed it down (in another), stacked so their sum lands exactly on the model's actual output for that example — directly useful for explaining one specific decision to a customer, auditor, or regulator (this connects directly to your Fairness & Responsible AI prep's Model Card requirement to be able to explain individual decisions, not just aggregate metrics).

## 6.4 LIME — a simpler, local approximation alternative

**The idea:** LIME ("Local Interpretable Model-agnostic Explanations") takes a very different approach to the same goal — instead of a principled game-theory allocation, it explains one specific prediction by **fitting a simple, interpretable model (usually a linear model) locally around that one prediction**, and reading off that simple model's coefficients as the explanation.

**The procedure:**
1. Pick the one specific prediction you want to explain.
2. Generate many slightly perturbed versions of that example (small random changes to its feature values).
3. Get the real (complex) model's prediction on each perturbed version.
4. Fit a simple linear model to these perturbed examples and their predictions, weighting perturbed examples that are closer to the original example more heavily.
5. That simple linear model's coefficients, restricted to this local neighborhood, are the explanation — "in the vicinity of this specific example, increasing credit_score by one unit was associated with this much change in the prediction."

**How LIME differs from SHAP, and why it matters which you pick:**
- LIME is **local by construction and by name** — it only claims to approximate the model's behavior *in the neighborhood of one specific example*, with no guarantee its explanation generalizes even slightly beyond that neighborhood.
- SHAP's Shapley-value foundation gives it certain guaranteed mathematical properties (the contributions genuinely sum to the actual prediction; a feature that never actually matters gets an exact zero, not just an approximately-small value) that LIME's locally-fit linear approximation doesn't guarantee.
- LIME is generally **cheaper to compute** than SHAP's more principled approximations, since it's "just" fitting one simple linear regression per explanation rather than approximating an exponential-order game-theoretic quantity.

## 6.5 When to reach for which

- **Need a rigorous, additive, per-prediction explanation, and can afford the compute** (especially for a tree-based model, where TreeSHAP is fast) → SHAP.
- **Need a quick, cheap, "roughly what's driving this one prediction" answer**, or are working with an unusual model type SHAP doesn't have an efficient variant for → LIME.
- **Just need an overall, dataset-wide ranking of which features matter most, not per-prediction detail** → permutation importance (Chapter 5) is usually simpler and cheaper than either SHAP or LIME for that specific, coarser question — reach for SHAP/LIME specifically when you need the *local*, per-example story, not just a global ranking.

---

**Next: Chapter 7 — Feature Importance for Linear Models**, returning to the simpler case of a linear/logistic regression model — why raw coefficient magnitude is meaningless without standardization, and how multicollinearity destabilizes coefficient-based importance.
