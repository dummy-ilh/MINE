# Chapter 3 — Counterfactual Explanations

Chapter 2 was the interpretable-by-design branch. This chapter starts the post-hoc branch with something genuinely different from every attribution method in your Feature Importance syllabus: instead of asking "how much did each feature contribute to this prediction," counterfactual explanations ask **"what would need to change about this input for the model to predict something different?"**

## 3.1 What a counterfactual explanation actually is

**The definition:** for a specific denied prediction, a counterfactual explanation is a **modified version of the same input** — as similar as possible to the original — that the model would have predicted differently (typically, the favorable outcome instead of the unfavorable one).

**Concrete example, continuing the loan scenario from your Feature Importance syllabus:** Applicant X was denied a loan. A SHAP explanation (attribution) tells you *why* — "debt-to-income ratio contributed −25 points, credit score contributed −8 points." A counterfactual explanation instead tells you **what to do about it** — *"if your annual income were $8,000 higher, with everything else about your application unchanged, the model would have approved you."*

**Why this is a genuinely different question, not just a rephrasing of attribution:** attribution decomposes an *existing* prediction into contributions from each feature, using the input exactly as given. A counterfactual instead **searches for a different, nearby input** that produces a different output — it's fundamentally a search/optimization problem over the input space, not a decomposition of one fixed input. This distinction matters practically: attribution tells someone why they were denied; a counterfactual tells them what they could actually change to get a different outcome next time — a much more directly actionable piece of information for the person on the receiving end of the decision.

## 3.2 How counterfactuals are generated

**The core optimization problem, stated conceptually:** find a new input x' that (a) the model predicts differently from the original x (flips the decision), while (b) staying as close as possible to x by some distance measure, and (c) ideally changing as few features as possible.

**A basic version of this as an optimization objective:**

minimize [ distance(x, x') + λ · (does the model's prediction on x' still match the original, undesired outcome?) ]

— search over candidate x' values, penalizing both how far x' strays from the original input and whether it still fails to flip the prediction, until you find an x' that's both close to the original and successfully flips the outcome. In practice this is often solved with gradient-based optimization (for differentiable models) or search/genetic-algorithm-style methods (for non-differentiable models like tree ensembles), since the "did the prediction flip" condition is a hard, non-smooth constraint that plain gradient descent alone can't handle cleanly.

**Worked conceptual walkthrough:** for Applicant X, denied primarily due to debt-to-income ratio and credit score (per the SHAP explanation from Chapter 10 of your Feature Importance syllabus), a counterfactual search might explore: increasing income (which lowers debt-to-income), paying down existing debt (which also lowers debt-to-income directly), or waiting for credit score to improve. The search would evaluate many candidate combinations of small changes, checking at each step whether the modified input flips the model's prediction, and return whichever combination achieves a flip with the smallest overall change from the original application.

## 3.3 Desirable properties of a good counterfactual

A "successful" counterfactual (one that flips the prediction) isn't automatically a *good* one — several additional properties determine whether it's actually useful to the person receiving it.

- **Proximity:** the counterfactual should be as close as possible to the original input — a counterfactual that changes ten features drastically is technically valid but not a useful, focused explanation of "what would need to change."
- **Sparsity:** related to proximity but specifically about the *number* of features changed, not just the total magnitude of change — a counterfactual that changes only one or two features is far easier for a person to act on and understand than one that adjusts many features by small amounts each.
- **Plausibility/actionability:** the suggested change needs to be realistic and within the person's actual control. A counterfactual suggesting "if you were 10 years younger" is useless — age isn't something a person can act on. Similarly, a counterfactual should ideally respect real-world feature relationships (recall the `age`/`years_of_work_experience` correlation from your Feature Importance syllabus's Chapter 4 — a counterfactual suggesting a combination that's realistically impossible given how correlated features actually relate to each other is not a genuinely actionable suggestion, even if it's mathematically valid input to the model).
- **Diversity:** rather than returning a single counterfactual, returning **several different paths** to a flipped prediction (e.g., "either increase income by $8,000, OR pay down $5,000 of existing debt, OR wait 6 months for credit score improvement") gives the person more than one potential route, since not every suggested change will be equally feasible for every individual.

**Why these properties can trade off against each other:** the single closest possible counterfactual (maximizing proximity) might not be very actionable (e.g., "have a 15-point-higher credit score" is close in raw distance terms but not something someone can just decide to have tomorrow) — a genuinely useful counterfactual generation method has to balance closeness against real-world actionability, not just minimize distance blindly.

## 3.4 Counterfactuals vs. SHAP/LIME — different questions, different use cases

**The clean distinction to have ready:** attribution methods (SHAP, LIME, and everything in your Feature Importance syllabus) answer **"why did the model predict this"** — decomposing the existing prediction into per-feature contributions. Counterfactuals answer **"what would need to be different for the model to predict something else"** — a contrastive, forward-looking, action-oriented question rather than a backward-looking, decompositional one.

**When each is the right tool:**
- A compliance audit asking "which factors drove this decision" → attribution (SHAP), since the axiom-guaranteed decomposition (Chapter 5 of your Feature Importance syllabus) gives a complete, verifiable accounting of the existing prediction.
- An applicant asking "what can I do differently next time" → counterfactual, since it directly answers the actionable question attribution doesn't address at all — SHAP will tell them debt-to-income mattered, but won't tell them how much their debt-to-income would need to change to flip the outcome.
- A regulator specifically requiring actionable recourse to be provided to affected individuals (some interpretations of GDPR's "right to explanation," covered further in Chapter 8) → counterfactual, since "recourse" is inherently a forward-looking, actionable concept that attribution alone doesn't satisfy.

**They're complementary, not competing, in most real deployments:** a mature explanation system for a high-stakes decision often provides both — an attribution-style breakdown for audit/compliance purposes, and a counterfactual for the affected individual's own actionable understanding — since they answer genuinely different questions that both matter.

## 3.5 A caveat worth flagging explicitly: counterfactuals inherit the predictive-vs-causal problem too

Just as Chapter 9 of your Feature Importance syllabus flagged that attribution methods only measure predictive/associational importance, not causal effect, **counterfactual explanations have the exact same limitation, in a slightly different guise.** A counterfactual search finds an input the *model* would treat differently — it does not guarantee that actually making that change in the real world would produce that outcome, if the model's relationship between that feature and the target is itself confounded rather than causal. Suggesting "increase your income by $8,000" as actionable advice implicitly assumes income has a genuine causal effect on creditworthiness in the way the model's association suggests — which may or may not hold, exactly the same caveat as the ice-cream/drowning pattern from earlier material, just now showing up as a recommendation for real-world action rather than just an explanation of a prediction.

## 3.6 Quick self-check before Chapter 4

- Can you state, in one sentence, the core difference between what attribution methods and counterfactual methods each answer?
- Given a specific denied loan application, can you sketch what a good counterfactual (satisfying proximity, sparsity, and plausibility) would look like, versus a technically-valid-but-useless one?
- Can you explain why a counterfactual explanation carries the same causal-vs-predictive caveat as attribution methods, even though it's framed as actionable advice?

---

**Next: Chapter 4 — Explaining Deep Learning Models: Saliency & Gradient-Based Methods**, moving into explanation techniques specifically built for models where the input isn't naturally tabular — starting with the simplest gradient-based saliency maps and their known fragility, then Integrated Gradients and Grad-CAM as more principled alternatives.
