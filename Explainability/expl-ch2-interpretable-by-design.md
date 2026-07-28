# Chapter 2 — Interpretable-by-Design Models

Chapter 1 laid out the taxonomy. This chapter is the entire "interpretable-by-design" branch — models whose own structure is the explanation, with no separate post-hoc step needed at all. We'll go from the classic baselines through the two techniques worth knowing in real depth: Generalized Additive Models and rule-based scoring systems.

## 2.1 The classic baselines, and precisely what interpretability they buy you

**Linear/logistic regression:** a prediction is a weighted sum of features (your Feature Importance syllabus's Chapter 3 covered exactly how to read these coefficients correctly — standardized, with VIF checked). What makes this interpretable isn't that the *numbers* are simple — it's that the **functional form** is fully known and fixed in advance: every feature's effect is assumed linear and additive, with no hidden interactions the model might be exploiting that you haven't explicitly included. Reading the model *is* reading the coefficients — there's no approximation gap between "what the model does" and "what you're looking at," which is exactly the property post-hoc methods (Chapters 3–6) can never fully guarantee for a black box.

**Shallow decision trees:** a small number of if-then splits, each directly traceable — you can literally follow the path an example takes from root to leaf and read off exactly why it received its prediction. The interpretability cost of a tree scales with depth: a tree with 3–4 levels is genuinely easy for a human to trace by eye; a tree with 20 levels and hundreds of nodes is technically still "a decision tree" but has lost essentially all of the interpretability benefit a shallow tree provides — depth is the dial that trades interpretability for the flexibility to capture more complex patterns.

**The shared limitation both baselines have:** linear models can't naturally capture non-linear relationships (recall the y=x² example from your Feature Selection syllabus, where a linear model's coefficient would show approximately nothing despite a perfect underlying relationship) without manual feature engineering (adding x² as its own feature, which you'd have to think to do). Shallow trees can capture non-linearity and interactions, but only crudely, and lose interpretability quickly as they grow deeper to capture more nuance. Generalized Additive Models exist specifically to solve the linear model's non-linearity limitation without giving up interpretability.

## 2.2 Generalized Additive Models (GAMs)

**The core idea:** instead of a single coefficient per feature, fit a separate, flexible **function** per feature, and add them together:

y = f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ) + b

Each fᵢ can be any shape — a smooth curve, a step function, whatever best fits that one feature's real relationship with the target — while the model as a whole stays **additive**: there's no interaction between features baked in by default (each feature's contribution can be computed and plotted entirely independently of what the other features are doing), which is exactly what preserves interpretability even though each individual function fᵢ can now be non-linear.

**Why this fixes the linear model's blind spot without sacrificing interpretability:** a GAM fitted on the y=x² example would learn f(x) as a U-shaped curve — correctly capturing the true non-linear relationship — and you can **directly plot fᵢ(xᵢ)** for any feature and see exactly what the model believes that feature's effect looks like, the same way you'd read a linear coefficient, just as a curve instead of a single number. This is the single biggest practical advantage of a GAM over a plain linear model for interpretability purposes: you get non-linearity without losing the "just look at the function" property that makes linear models interpretable in the first place.

**How the individual functions are actually fit, at a conceptual level:** commonly via splines (piecewise polynomial curves, smoothly joined) or, in more modern implementations, small per-feature neural networks or shallow tree ensembles (in an "Explainable Boosting Machine," one popular modern GAM variant) — the fitting mechanics can get sophisticated, but the interpretability property is preserved regardless of which fitting method is used underneath, precisely because the *additive, no-hidden-interactions* structure is what matters for interpretability, not the specific curve-fitting technique.

**The limitation, stated honestly:** a plain GAM still assumes no interactions between features by default — if the true relationship genuinely depends on a *combination* of two features (e.g., age and income interact in a way neither one alone captures), a plain GAM will miss it, the same blind spot linear models have for interactions. GA²M (GAM with pairwise interaction terms) extends the additive structure to include a limited, still-interpretable set of two-way interaction terms fᵢⱼ(xᵢ, xⱼ) — a reasonable middle ground, since a two-feature interaction can still be visualized as a 2D heatmap and remains directly interpretable, whereas allowing arbitrary higher-order interactions would eventually collapse back into an uninterpretable black box.

## 2.3 Rule-based models: decision lists, RuleFit, and scoring systems

**Decision lists:** an ordered sequence of if-then rules, evaluated top to bottom — "IF condition 1, predict Y1; ELSE IF condition 2, predict Y2; ELSE predict default" — a strict, human-readable ordering that's arguably even more directly legible than a tree, since there's no branching to visually trace, just a linear list read from top to bottom.

**RuleFit:** a technique that generates a large pool of candidate rules (often extracted from an initial tree ensemble, converting each path from root to leaf into a rule) and then fits a sparse linear model (Lasso, from your Feature Selection syllabus's Chapter 4) over these rules plus the original features — the L1 penalty drives most candidate rules' coefficients to exactly zero, leaving a small, interpretable set of surviving rules with weights indicating each one's contribution.

**Scoring systems (point-based risk scores):** widely used in medicine and criminal justice specifically because of their extreme interpretability — each risk factor contributes a small integer number of points (e.g., "+2 points if age > 65," "+1 point if prior condition present"), and the final prediction is simply the sum of points compared against a threshold. A clinician can compute a patient's score by hand in seconds, with zero computational tooling required at all — this is about as far toward "the model IS the explanation" as it's possible to get, and it's exactly why these systems remain in active clinical use even in an era of far more powerful models: the interpretability isn't a nice-to-have, it's often a hard requirement for adoption by the people who'll actually use the tool under time pressure.

**How a scoring system like this is actually built, conceptually:** typically starts from a fitted logistic regression, then rounds and rescales the coefficients to small integers in a way that approximately preserves the model's discriminative performance (techniques like the Supersparse Linear Integer Model, or SLIM, formalize this rounding step as its own optimization problem, directly trading off a small amount of accuracy for coefficients simple enough to compute by hand).

## 2.4 The strongest version of "just use an interpretable model" (Rudin's argument)

**The argument, stated in its strongest form:** for many high-stakes decisions (especially ones with structured, tabular input data — recall Chapter 1, §1.3's point that the accuracy gap is often small for exactly this data type), there is frequently **no meaningful accuracy sacrifice** in choosing an interpretable model over a black box — meaning the common justification for reaching for a black box plus a post-hoc explanation ("we need the accuracy") often doesn't actually hold up empirically for the specific problem at hand. Combined with Chapter 1's point that post-hoc explanations carry a real risk of being unfaithful to the underlying black box's true reasoning, the conclusion is: **for high-stakes decisions, try genuinely hard to build an interpretable-by-design model first, and only reach for a black box plus post-hoc explanation if you've actually measured a real, meaningful accuracy gap that justifies the added risk.**

**Where this argument is weaker, and worth acknowledging honestly:** for problem types where black-box models have a genuine, substantial structural advantage — vision, language, and other high-dimensional, unstructured data — an interpretable-by-design model (even a GAM) often cannot come close to a deep network's accuracy, and the tradeoff becomes real and unavoidable rather than illusory. This is exactly why Chapters 4–6 of this syllabus exist at all: post-hoc explanation of black boxes remains necessary and important specifically in the domains where interpretable-by-design alternatives genuinely can't compete.

**The interview-ready synthesis:** *"For structured, tabular, high-stakes decisions, I'd try a strong interpretable baseline — a GAM or a well-built scoring system — first, and only justify a black box plus post-hoc explanation with a measured, meaningful accuracy gap. For vision, language, or other high-dimensional unstructured data, the interpretable-by-design alternatives genuinely can't compete, and post-hoc explanation of a black box becomes the realistic path."*

## 2.5 Quick self-check before Chapter 3

- Can you explain, precisely, why a GAM preserves interpretability even when each individual feature function fᵢ is non-linear?
- Can you explain what a plain GAM still misses, and how GA²M's pairwise interaction terms address it while staying interpretable?
- Given Rudin's argument, can you state one problem type where it applies strongly and one where it applies weakly, and why the difference exists?

---

**Next: Chapter 3 — Counterfactual Explanations**, moving to the post-hoc branch and a genuinely different kind of question than attribution: not "why did the model predict this," but "what would need to change for the model to predict something else."
