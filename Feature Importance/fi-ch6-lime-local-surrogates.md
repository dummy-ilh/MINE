# Chapter 6 — LIME and Other Local Surrogate Methods

Chapter 5 gave SHAP its full formal treatment. This chapter covers the other major local explanation approach — LIME — going deeper than the earlier intuitive pass: the exact sampling and weighting mechanics that make it work, precisely why those same mechanics make it somewhat unstable, and Anchors as a rule-based alternative that sidesteps some of LIME's issues.

## 6.1 The full LIME procedure, mechanically

**Goal:** explain one specific prediction f(x) for one specific example x, by fitting a simple, interpretable model that approximates f's behavior **in the local neighborhood around x**.

**Step-by-step:**
1. **Choose the example x to explain**, and the complex model f you want to explain (this can be any black-box model — a random forest, a neural network, anything).
2. **Generate a set of perturbed samples** around x: for tabular data, this typically means sampling new points by randomly varying x's feature values (e.g., for continuous features, adding random noise around x's original values; for categorical/text/image data, methods differ but the idea is the same — generate many "nearby" variants of x).
3. **Get the black-box model's prediction f(z) for every perturbed sample z.** This is the only place the complex model gets used at all — everything after this step only touches these input/output pairs, treating f as a pure black box.
4. **Weight each perturbed sample by its proximity to the original x**, using a kernel function (commonly a Gaussian/RBF-style kernel: closer perturbed samples to x get higher weight, farther ones get lower weight) — this is what makes the resulting fit "local" rather than a global approximation of f across the whole input space.
5. **Fit a simple, interpretable model** (almost always a weighted linear regression, sometimes a shallow decision tree) to the perturbed samples' (feature values, f-prediction) pairs, using the proximity weights from step 4.
6. **Read off the simple model's coefficients** as the explanation — "in the neighborhood immediately around this example, increasing feature j by one unit was associated with this much change in the prediction."

## 6.2 The choices that determine LIME's explanation, and why each one matters

Every step above involves a **choice** the practitioner makes, and each choice directly shapes the resulting explanation — this is the key thing that distinguishes LIME from SHAP's more principled, axiom-derived uniqueness (Chapter 5, §5.3).

- **The perturbation/sampling strategy** (how exactly you generate "nearby" points) determines what "local" even means for this specific data type — and there's no single canonically correct choice, unlike SHAP's mathematically-derived weighting.
- **The kernel width** (how quickly the proximity weight falls off with distance from x) directly controls how "local" the explanation really is — a very narrow kernel produces an explanation that's extremely specific to x but potentially noisy (few perturbed samples get meaningful weight); a very wide kernel produces a smoother but less genuinely "local" explanation that starts to blur into a more global approximation of f.
- **The choice of interpretable surrogate model family** (linear vs. a shallow tree) determines what *shape* of local explanation is even expressible — a linear surrogate can only express a locally-linear relationship, even if f's true local behavior is more complex.

**Why this matters as the central critique of LIME, stated precisely:** because none of these choices are derived from a set of provably-necessary axioms (contrast Chapter 5's four Shapley axioms, which *uniquely* determine the formula), **two reasonable practitioners can make different, both-defensible choices and get genuinely different explanations for the same prediction** — and even the same practitioner, rerunning LIME twice with a different random seed for the perturbation sampling, can get noticeably different coefficient estimates, since the whole procedure depends on a randomly-sampled set of perturbed points.

## 6.3 Where LIME can genuinely mislead: non-smooth local behavior

**The core assumption LIME makes, stated explicitly:** that f behaves *approximately linearly* in the neighborhood immediately around x — smoothly and gradually, without sharp transitions.

**Where this assumption breaks down:** many models — trees and tree ensembles especially — have genuinely **non-smooth** decision boundaries, with sharp changes in prediction right at a split threshold. If x sits very close to one of these sharp transitions, LIME's perturbed samples will straddle the transition, and a linear surrogate fit across that boundary will produce a coefficient that reflects an *average* of two genuinely different local behaviors on either side of the sharp transition — potentially misrepresenting what's actually happening at x specifically, rather than genuinely capturing the local behavior LIME claims to explain.

**The practical consequence:** LIME's explanations should be treated with more caution specifically for examples that plausibly sit close to a model's genuine decision boundaries or split thresholds — precisely where a locally-linear approximation is least likely to hold. This is a structural limitation, not a bug that better implementation choices in §6.2 can fully fix — it's inherent to approximating an arbitrary function locally with a linear model.

## 6.4 Anchors — a rule-based alternative

**The idea (Ribeiro, Singh & Guestrin, 2018 — the same authors as LIME, proposing a complementary approach):** instead of a linear surrogate, explain a prediction using an **if-then rule** — a minimal set of feature-value conditions ("anchor") such that, whenever those conditions hold, the model's prediction stays essentially the same with high probability, **regardless of what the other, unanchored features take on**.

**Concrete illustration:** rather than LIME's "increasing credit_score by one unit changes the log-odds by 0.02 in this neighborhood," an Anchor explanation might say: *"IF credit_score > 720 AND debt_to_income < 0.3, THEN the model predicts approval with 97% probability, regardless of any other feature."* This is a precise, verifiable, human-readable rule rather than a locally-linear coefficient.

**Why this sidesteps LIME's non-smoothness problem:** an Anchor rule doesn't assume smooth, locally-linear behavior at all — it directly searches for a region of the input space (defined by a small set of feature-value conditions) where the prediction is provably stable with high probability, which handles sharp decision boundaries gracefully, since the rule can simply be scoped tightly enough to stay entirely on one side of any such boundary.

**The tradeoff:** Anchors can be more computationally expensive to find (searching for a good rule involves testing many candidate condition combinations and estimating each one's "precision" — how consistently the prediction holds under that rule — often via repeated sampling), and the resulting rule may cover only a narrow slice of the input space, meaning it doesn't tell you anything about how the prediction would change if you moved outside that specific rule's conditions — unlike LIME's coefficients, which at least gesture at a direction and rate of change, even if potentially misleading near a sharp boundary.

## 6.5 When to reach for LIME, SHAP, or Anchors

- **Need a fast, rough, "what's roughly driving this prediction" answer, and the model's local behavior is plausibly smooth** (e.g., far from any sharp decision threshold) → LIME, cheaply.
- **Need a rigorous, axiom-guaranteed, additive explanation**, especially for a tree-based model where TreeSHAP is both exact and fast → SHAP.
- **Need a human-readable, verifiable rule** — especially useful for a compliance/audit context where "explain the exact conditions under which this decision is guaranteed" is more valuable than a coefficient — → Anchors.
- **Suspect the example sits near a sharp decision boundary** (a common situation right at a threshold-based business decision, like a loan cutoff) → prefer SHAP or Anchors over LIME specifically, since LIME's core linearity assumption is least trustworthy exactly there.

## 6.6 Quick self-check before Chapter 7

- Can you name the three practitioner choices in LIME's procedure that have no single "correct" answer, and explain why that's a meaningful critique compared to SHAP?
- Can you explain, mechanistically, why a sharp decision boundary specifically breaks LIME's core assumption?
- Given a compliance/audit scenario, could you explain why an Anchor rule might be preferred over both LIME and SHAP for that specific use case?

---

**Next: Chapter 7 — Partial Dependence & Individual Conditional Expectation (PDP/ICE)**, shifting from per-prediction attribution methods to a different lens entirely — how the model's predictions change as one feature varies, and why correlated features create the same extrapolation problem here that you saw in permutation importance (Chapter 4).
